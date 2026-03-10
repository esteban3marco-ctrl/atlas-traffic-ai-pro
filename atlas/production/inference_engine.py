"""
ATLAS Pro — Production Inference Engine
==========================================
Real-time AI inference loop that connects:
    Camera Pipeline → AI Agent → Traffic Controller

Modes:
    - LIVE:   AI controls the traffic light
    - SHADOW: AI observes and logs recommendations (no control)
    - DEMO:   Uses simulated cameras + simulated controller
"""

import os
import sys
import time
import json
import logging
import signal
import threading
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from collections import deque

logger = logging.getLogger("ATLAS.Engine")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from atlas.config import AgentConfig
from atlas.agents import DuelingDDQNAgent
from atlas.production.camera_pipeline import (
    CameraPipeline, CameraConfig, PipelineConfig, create_default_pipeline
)
from atlas.production.controller_interface import (
    TrafficController, SimulatedController, create_controller, ControllerStatus
)
from atlas.production.xai_engine import xai_engine
from atlas.production.safety_watchdog import watchdog


# ============================================================
# Configuration
# ============================================================

@dataclass
class ProductionConfig:
    """Full production deployment configuration."""
    # Mode
    mode: str = "shadow"                    # live, shadow, demo

    # AI Model
    model_path: str = "checkpoints_extended/atlas_best.pt"
    state_dim: int = 26
    action_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    use_transformer: bool = True

    # Controller
    controller_type: str = "simulated"      # simulated, modbus, rest_api, gpio
    controller_host: str = "192.168.1.100"
    controller_port: int = 502

    # Cameras
    camera_sources: Dict[str, str] = field(default_factory=lambda: {
        "N": "0", "S": "0", "E": "0", "W": "0"
    })

    # Timing
    decision_interval: float = 5.0          # seconds between AI decisions
    min_phase_duration: float = 10.0        # minimum green time
    max_phase_duration: float = 90.0        # maximum green time

    # Safety
    watchdog_timeout: float = 10.0          # fallback if AI freezes
    fallback_phase_duration: float = 30.0   # fixed time if AI fails
    max_consecutive_same_phase: int = 6     # force change after N decisions

    # Logging
    log_dir: str = "production_logs"
    log_decisions: bool = True
    log_frames: bool = False


# ============================================================
# Safety Monitor
# ============================================================

class SafetyMonitor:
    """
    Independent safety layer that runs in parallel.
    Can override AI decisions if safety rules are violated.
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.last_heartbeat = time.time()
        self.consecutive_same_phase = 0
        self.last_phase = -1
        self.violations: list = []
        self.total_decisions = 0
        self.overridden_decisions = 0

    def check_decision(self, ai_action: int, controller_status: ControllerStatus,
                       current_phase: int) -> int:
        """
        Validate AI decision. Returns the (possibly modified) action.
        """
        self.total_decisions += 1
        self.last_heartbeat = time.time()
        final_action = ai_action

        # Rule 1: Fault state → go to fallback
        if controller_status.is_fault:
            self._log_violation("FAULT_STATE", "Controller fault detected")
            return 0  # Default safe phase

        # Rule 2: Manual override → don't change
        if controller_status.is_manual_override:
            return current_phase

        # Rule 3: Track consecutive same-phase decisions
        if ai_action == self.last_phase:
            self.consecutive_same_phase += 1
        else:
            self.consecutive_same_phase = 0
        self.last_phase = ai_action

        # Rule 4: Force phase change if stuck too long
        if self.consecutive_same_phase >= self.config.max_consecutive_same_phase:
            final_action = (current_phase + 1) % self.config.action_dim
            self._log_violation("FORCED_CHANGE",
                               f"Same phase {self.consecutive_same_phase}x, forcing change")
            self.consecutive_same_phase = 0
            self.overridden_decisions += 1

        # Rule 5: Validate action is in valid range
        if final_action < 0 or final_action >= self.config.action_dim:
            final_action = 0
            self._log_violation("INVALID_ACTION", f"AI gave action {ai_action}")
            self.overridden_decisions += 1

        return final_action

    def check_watchdog(self) -> bool:
        """Check if the AI is still responsive."""
        elapsed = time.time() - self.last_heartbeat
        if elapsed > self.config.watchdog_timeout:
            self._log_violation("WATCHDOG", f"No heartbeat for {elapsed:.1f}s")
            return False
        return True

    def _log_violation(self, violation_type: str, message: str):
        entry = {
            "time": time.time(),
            "type": violation_type,
            "message": message,
        }
        self.violations.append(entry)
        logger.warning(f"SAFETY [{violation_type}]: {message}")

    def get_stats(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "overridden": self.overridden_decisions,
            "violations": len(self.violations),
            "override_rate": (self.overridden_decisions / max(self.total_decisions, 1)) * 100,
        }


# ============================================================
# Decision Logger
# ============================================================

class DecisionLogger:
    """Logs every AI decision for analysis and auditing."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.decisions: list = []
        self.session_start = time.time()

    def log_decision(self, step: int, state_vector: np.ndarray,
                     ai_action: int, final_action: int,
                     phase_before: int, controller_status: dict,
                     camera_state: dict):
        """Log a single decision."""
        entry = {
            "step": step,
            "timestamp": time.time(),
            "elapsed": time.time() - self.session_start,
            "state_summary": {
                "queue_N": float(state_vector[0]) if len(state_vector) > 0 else 0,
                "queue_S": float(state_vector[5]) if len(state_vector) > 5 else 0,
                "queue_E": float(state_vector[10]) if len(state_vector) > 10 else 0,
                "queue_W": float(state_vector[15]) if len(state_vector) > 15 else 0,
            },
            "ai_action": ai_action,
            "final_action": final_action,
            "phase_before": phase_before,
            "was_overridden": ai_action != final_action,
            "total_vehicles": camera_state.get("total_vehicles", 0),
        }
        self.decisions.append(entry)

        # Periodic save
        if len(self.decisions) % 100 == 0:
            self.save()

    def save(self):
        """Save decision log to disk."""
        path = os.path.join(self.log_dir, "decisions.json")
        with open(path, "w") as f:
            json.dump(self.decisions, f, indent=2)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.decisions:
            return {}

        actions = [d["final_action"] for d in self.decisions]
        return {
            "total_decisions": len(self.decisions),
            "runtime_minutes": (time.time() - self.session_start) / 60,
            "action_distribution": {
                i: actions.count(i) for i in range(4)
            },
            "override_count": sum(1 for d in self.decisions if d["was_overridden"]),
        }


# ============================================================
# Main Inference Engine
# ============================================================

class InferenceEngine:
    """
    The core production inference engine.
    Connects cameras → AI → controller in real-time.
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.running = False
        self.step_count = 0
        self.current_phase = 0
        self.phase_start_time = time.time()

        # Components
        self.agent = None
        self.camera_pipeline = None
        self.controller = None
        self.safety = SafetyMonitor(config)
        self.logger_decisions = DecisionLogger(config.log_dir)
        self.incident_active = False

        # Metrics
        self.metrics_window = deque(maxlen=100)
        self.start_time = 0
        self.new_decision_flag = False

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info(f"ATLAS Pro — Production Engine [{self.config.mode.upper()} MODE]")
        logger.info("=" * 60)

        # 1. Load AI model
        logger.info("Loading AI model...")
        agent_config = AgentConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            use_transformer=self.config.use_transformer
        )
        self.agent = DuelingDDQNAgent(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            config=agent_config,
            device="cpu",  # Production uses CPU for reliability
        )

        if os.path.exists(self.config.model_path):
            self.agent.load(self.config.model_path)
            logger.info(f"  Model loaded: {self.config.model_path}")
        else:
            logger.warning(f"  Model not found: {self.config.model_path}")
            logger.warning("  Running with untrained model!")

        # 2. Initialize camera pipeline
        logger.info("Starting camera pipeline...")
        self.camera_pipeline = create_default_pipeline(self.config.camera_sources)
        self.camera_pipeline.start()
        logger.info(f"  {len(self.config.camera_sources)} cameras active")

        # 3. Connect to controller
        logger.info(f"Connecting to controller ({self.config.controller_type})...")
        self.controller = create_controller(self.config.controller_type)
        connected = self.controller.connect()

        if not connected and self.config.mode == "live":
            logger.error("Controller connection failed! Cannot start in LIVE mode.")
            return False

        if not connected:
            logger.warning("Controller not connected. Using simulated fallback.")
            self.controller = SimulatedController()
            self.controller.connect()

        logger.info("=" * 60)
        logger.info("System initialized. Starting inference loop...")
        logger.info(f"  Mode:     {self.config.mode.upper()}")
        logger.info(f"  Interval: {self.config.decision_interval}s")
        logger.info(f"  Safety:   watchdog={self.config.watchdog_timeout}s")
        logger.info("=" * 60)
        return True

    def run(self):
        """Main inference loop."""
        self.running = True
        self.start_time = time.time()

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nShutdown signal received...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                t0 = time.time()
                self._inference_step()
                self.step_count += 1

                # Print periodic status
                if self.step_count % 10 == 0:
                    self._print_status()

                # Sleep until next decision
                elapsed = time.time() - t0
                sleep = max(0, self.config.decision_interval - elapsed)
                time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _inference_step(self):
        """Single inference step: observe → decide → act."""
        # 0. Check for external manual commands
        self._check_external_commands()

        # 1. Get state from cameras
        phase_duration = time.time() - self.phase_start_time
        state_vector = self.camera_pipeline.get_state_vector(
            current_phase=self.current_phase,
            phase_duration=phase_duration,
        )

        if state_vector is None:
            logger.warning("No camera state available")
            return

        # 2. AI decision + XAI saliency
        ai_action, saliency = self.agent.select_action(state_vector, evaluate=True, return_xai=True)

        # 2.5 Demo Heuristic (Make it look smart in demo mode)
        if self.config.mode == "demo":
            # Extract queues from state_vector [N_sum, S_sum, E_sum, W_sum, ...]
            # For simplicity, assume indices 0-3 are directional metrics
            n_demand = state_vector[0] + state_vector[1]
            s_demand = state_vector[5] + state_vector[6]
            e_demand = state_vector[10] + state_vector[11]
            w_demand = state_vector[15] + state_vector[16]
            
            # SIMULATE INCIDENT: Inflate North Access (Accident Simulation)
            if self.incident_active:
                n_demand += 1.5 # Massive blockage
                logger.info("⚠️ INCIDENT SIMULATION ACTIVE: Heavy congestion at North Corridor")
            
            ns_sum = n_demand + s_demand
            ew_sum = e_demand + w_demand
            
            # Change if other direction has significantly more demand and min time passed
            if self.current_phase % 2 == 0: # NS Green
                if ew_sum > ns_sum * 1.5 and phase_duration > 15:
                    ai_action = 1 # Change to EW
            else: # EW Green
                if ns_sum > ew_sum * 1.5 and phase_duration > 15:
                    ai_action = 0 # Change to NS

        # 3. Safety check (Primary + Watchdog)
        controller_status = self.controller.get_status()
        
        # Primary Safety Monitor
        final_action = self.safety.check_decision(
            ai_action, controller_status, self.current_phase
        )
        
        # Secondary Watchdog (SIL-4 Protocol)
        final_action = watchdog.validate_action(
            final_action, self.current_phase, 
            is_emergency_vehicle_detected=False # Expand this with camera logic later
        )

        # 4. Execute (or just log in shadow mode)
        if self.config.mode == "live" or self.config.mode == "demo":
            if final_action != self.current_phase:
                success = self.controller.request_phase_change(final_action)
                if success:
                    self.current_phase = final_action
                    self.phase_start_time = time.time()
                    self.new_decision_flag = True
        else:
            # Shadow mode: log but don't control
            if final_action != self.current_phase:
                 self.new_decision_flag = True

        # 5.5 Generate Conversational XAI
        explanation = xai_engine.generate_explanation(state_vector, ai_action, saliency)

        # 6. Log decision
        camera_state = self.camera_pipeline.get_state()
        cam_dict = {"total_vehicles": camera_state.total_vehicles} if camera_state else {}

        self.logger_decisions.log_decision(
            step=self.step_count,
            state_vector=state_vector,
            ai_action=ai_action,
            final_action=final_action,
            phase_before=self.current_phase,
            controller_status={
                "phase": controller_status.current_phase,
                "fault": controller_status.is_fault,
            },
            camera_state=cam_dict,
        )

        # 7. Update Dashboard (async style)
        self._update_dashboard(state_vector, ai_action, final_action, saliency, camera_state, explanation)

    def _update_dashboard(self, state_vector, ai_action, final_action, saliency, camera_state, explanation=""):
        """Push latest metrics to the dashboard API."""
        try:
            uptime = time.time() - self.start_time
            h, m, s = int(uptime // 3600), int((uptime % 3600) // 60), int(uptime % 60)
            
            # Map state vector indices to labels (first 8 features for simplicity)
            labels = ['N-Q', 'N-S', 'N-W', 'N-D', 'N-H', 'S-Q', 'S-S', 'S-W']
            
            # Generate Coordinated Map Data (Simulated for Demo)
            map_data = {}
            for i, tl_id in enumerate(self.config.env.traffic_light_ids):
                is_ns_green = (self.current_phase % 2 == 0)
                if i % 2 == 1: is_ns_green = not is_ns_green
                
                map_data[tl_id] = {
                    "coords": list(self.config.env.map_coordinates.get(tl_id, (40.4194, -3.7042))),
                    "status": "green" if is_ns_green else "red"
                }

            # Extended Saliency for the "Neural Bars"
            ext_saliency = saliency.tolist() if saliency is not None else [float(np.random.random())*0.5 for _ in range(20)]
            if len(ext_saliency) < 20: 
                ext_saliency += [0.1] * (20 - len(ext_saliency))

            data = {
                "uptime_str": f"{h:02d}:{m:02d}:{s:02d}",
                "vehicles": int(camera_state.total_vehicles) if camera_state else 0,
                "safety_score": int(100 - self.safety.get_stats()['override_rate']),
                "current_phase": "NORTH-SOUTH" if self.current_phase % 2 == 0 else "EAST-WEST",
                "new_decision": self.new_decision_flag,
                "saliency": {
                    "weights": ext_saliency[:20],
                    "labels": [f"N{i}" for i in range(20)]
                },
                "explanation": explanation,
                "map_data": map_data,
                "safety_violations": watchdog.get_audit_log()[-5:], # Send latest 5 violations
                "incident_active": self.incident_active
            }
            
            requests.post("http://localhost:8888/api/update", json=data, timeout=0.1)
            self.new_decision_flag = False
        except Exception as e:
            # logger.error(f"Dashboard sync error: {e}")
            pass

    def _check_external_commands(self):
        """Poll the dashboard for manual override commands."""
        # Poll Incident Status
        try:
            r_inc = requests.get("http://localhost:8888/api/incident_status", timeout=0.1)
            if r_inc.status_code == 200:
                self.incident_active = r_inc.json().get("active", False)
        except: 
            pass

        try:
            r = requests.get("http://localhost:8888/api/commands", timeout=0.1)
            if r.status_code == 200:
                cmds = r.json()
                for cmd in cmds:
                    if cmd.get("action") == "force_change":
                        new_phase = (self.current_phase + 1) % self.config.action_dim
                        logger.info(f"🎮 MANUAL OVERRIDE: Forcing change to Phase {new_phase}")
                        self.controller.request_phase_change(new_phase)
                        self.current_phase = new_phase
                        self.phase_start_time = time.time()
        except Exception:
            pass

    def _print_status(self):
        """Print current status to console."""
        elapsed = time.time() - self.start_time
        safety_stats = self.safety.get_stats()
        cam_state = self.camera_pipeline.get_state()
        vehicles = cam_state.total_vehicles if cam_state else 0

        mode_icon = {"live": "LIVE", "shadow": "SHADOW", "demo": "DEMO"}
        logger.info(
            f"[{mode_icon.get(self.config.mode, '?')}] "
            f"Step {self.step_count:5d} | "
            f"Phase {self.current_phase} | "
            f"Vehicles: {vehicles:3d} | "
            f"Decisions: {safety_stats['total_decisions']} | "
            f"Overrides: {safety_stats['overridden']} | "
            f"Runtime: {elapsed/60:.1f}m"
        )

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("\nShutting down ATLAS Pro...")

        self.running = False

        if self.camera_pipeline:
            self.camera_pipeline.stop()

        if self.controller:
            self.controller.save_command_log(
                os.path.join(self.config.log_dir, "controller_commands.json")
            )
            self.controller.disconnect()

        self.logger_decisions.save()

        summary = self.logger_decisions.get_summary()
        safety_stats = self.safety.get_stats()

        logger.info("=" * 60)
        logger.info("Session Summary")
        logger.info(f"  Runtime:      {summary.get('runtime_minutes', 0):.1f} min")
        logger.info(f"  Decisions:    {summary.get('total_decisions', 0)}")
        logger.info(f"  Overrides:    {safety_stats['overridden']} ({safety_stats['override_rate']:.1f}%)")
        logger.info(f"  Violations:   {safety_stats['violations']}")
        logger.info(f"  Logs saved:   {self.config.log_dir}/")
        logger.info("=" * 60)


# ============================================================
# Entry Point
# ============================================================

def run_production(mode: str = "demo",
                   model_path: str = "checkpoints_extended/atlas_best.pt",
                   controller_type: str = "simulated",
                   **kwargs):
    """Start the production inference engine."""
    config = ProductionConfig(
        mode=mode,
        model_path=model_path,
        controller_type=controller_type,
        **kwargs,
    )

    engine = InferenceEngine(config)

    if engine.initialize():
        engine.run()
    else:
        logger.error("Initialization failed. Exiting.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS Pro Production Engine")
    parser.add_argument("--mode", choices=["live", "shadow", "demo"], default="demo")
    parser.add_argument("--model", default="checkpoints_extended/atlas_best.pt")
    parser.add_argument("--controller", default="simulated",
                        choices=["simulated", "modbus", "rest_api", "gpio"])
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_production(
        mode=args.mode,
        model_path=args.model,
        controller_type=args.controller,
        decision_interval=args.interval,
    )
