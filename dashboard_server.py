"""
ATLAS Pro v4.0 — Real-Time Dashboard Backend
═════════════════════════════════════════════

FastAPI server that bridges the Python AI engine with the web dashboard.
Exposes:
  - REST endpoints for configuration and manual control
  - WebSocket /ws/live  → streaming real-time telemetry to dashboard
  - Control API        → manual overrides, phase changes, start/stop sim

Run:
    pip install fastapi uvicorn[standard] numpy
    python dashboard_server.py
    Open: http://localhost:8000
"""

import asyncio
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel

# ── ATLAS Imports ──────────────────────────────────────────────────────
try:
    from src.agent.ddqn_agent import ATLASAgent, HyperParams, IntersectionObservation, compute_reward
    from src.muse.muse_engine import MUSEEngine
    from src.simulation.intersection_sim import IntersectionSimulator, SimConfig
    from src.coordination.qmix_coordinator import QMIXCoordinator
    ATLAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ATLAS modules not found, using built-in simulation: {e}")
    ATLAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("atlas.server")

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="ATLAS Pro Dashboard", version="4.0")

# Serve static files (dashboard HTML/CSS/JS)
STATIC_DIR = Path(__file__).parent / "dashboard_static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Simulation State ───────────────────────────────────────────────────
INTERSECTION_IDS = ["INT_01", "INT_02", "INT_03", "INT_04", "INT_05", "INT_06"]
PHASE_NAMES = {
    0: "N-S Through",
    1: "E-W Through",
    2: "N-S Left Turn",
    3: "E-W Left Turn",
}

class SimulationEngine:
    """Wraps ATLAS Python modules or falls back to built-in simulation."""

    def __init__(self):
        self.running = False
        self.paused = False
        self.speed = 1.0          # Simulation speed multiplier
        self.step = 0
        self.sim_time = 28800     # Start at 08:00 (seconds from midnight)
        self.clients: list[WebSocket] = []
        self.manual_overrides: dict[str, int] = {}  # iid → forced phase
        self.event_active: dict[str, bool] = {iid: False for iid in INTERSECTION_IDS}

        # Per-intersection state
        self.states: dict[str, dict] = {
            iid: self._make_initial_state(iid, i)
            for i, iid in enumerate(INTERSECTION_IDS)
        }
        self.decisions: list[dict] = []   # Last N MUSE decisions
        self.metrics_history: list[dict] = []
        self.total_vehicles_served = 0
        self.total_co2_saved_kg = 0.0

        # Try to load real ATLAS modules
        if ATLAS_AVAILABLE:
            hp = HyperParams()
            self._sims = {
                iid: IntersectionSimulator(iid, SimConfig(), seed=i)
                for i, iid in enumerate(INTERSECTION_IDS)
            }
            self._agents = {
                iid: ATLASAgent(iid, hp)
                for iid in INTERSECTION_IDS
            }
            self._muse = {
                iid: MUSEEngine(iid, audit_log_path=f"data/audit/{iid}.jsonl")
                for iid in INTERSECTION_IDS
            }
            self._hp = hp
            logger.info("✓ ATLAS Python modules loaded")
        else:
            self._sims = None
            self._agents = None
            self._muse = None
            logger.info("⚠ Using built-in simulation (ATLAS modules not available)")

    def _make_initial_state(self, iid: str, idx: int) -> dict:
        base_lat = 40.4168 + (idx // 3) * 0.005
        base_lon = -3.7038 + (idx % 3) * 0.007
        return {
            "id": iid,
            "name": f"Intersection {iid}",
            "lat": base_lat,
            "lon": base_lon,
            "phase": 0,
            "phase_duration": 0,
            "queue": [0.0, 0.0, 0.0, 0.0],
            "occupancy": [0.0, 0.0, 0.0, 0.0],
            "speed": [0.8, 0.8, 0.8, 0.8],
            "throughput": 0.0,
            "wait_time_avg": 0.0,
            "ai_confidence": 0.85,
            "q_values": {"phase_0": 0.0, "phase_1": 0.0, "phase_2": 0.0, "phase_3": 0.0},
            "anomaly": False,
            "anomaly_desc": "",
            "vehicles_served": 0,
            "status": "active",
            "manual_override": False,
        }

    # ── Simulation Loop ────────────────────────────────────────────────

    async def run_loop(self):
        self.running = True
        logger.info("ATLAS simulation loop started")
        while self.running:
            if not self.paused:
                await self._tick()
                await self._broadcast()
            dt = 1.0 / max(0.1, self.speed)
            await asyncio.sleep(dt)

    async def _tick(self):
        self.step += 1
        self.sim_time += 1
        hour = (self.sim_time // 3600) % 24

        if ATLAS_AVAILABLE and self._sims:
            await self._tick_atlas(hour)
        else:
            await self._tick_builtin(hour)

        # Global metrics
        if self.step % 10 == 0:
            self._update_global_metrics(hour)

    async def _tick_atlas(self, hour: int):
        """Use real ATLAS Python modules for simulation."""
        for iid in INTERSECTION_IDS:
            sim = self._sims[iid]
            agent = self._agents[iid]
            muse = self._muse[iid]
            state = self.states[iid]

            # Get current observation from simulator
            obs, reward, done = sim.step(state["phase"])

            # Get AI action
            if iid in self.manual_overrides:
                action = self.manual_overrides[iid]
                state["manual_override"] = True
            else:
                action = agent.act(obs, explore=True)
                state["manual_override"] = False

            # MUSE explanation
            q_vals = agent.get_q_values(obs)
            record = muse.explain(q_vals, obs, action, agent.epsilon, reward)

            # Update visual state
            state["phase"] = action
            state["phase_duration"] = int(obs.phase_duration_s)
            state["queue"] = [round(q, 3) for q in obs.queue_lengths]
            state["occupancy"] = [round(o, 3) for o in obs.occupancies]
            state["speed"] = [round(s, 3) for s in obs.avg_speeds]
            q_arr = list(q_vals.values())
            q_max = max(q_arr) if q_arr else 0
            e_q = [math.exp((q - q_max)*1.5) for q in q_arr]
            state["ai_confidence"] = round(max(e_q) / (sum(e_q) + 1e-6), 3)
            state["q_values"] = {k: round(v, 4) for k, v in q_vals.items()}
            state["anomaly"] = record.anomaly_flag
            state["anomaly_desc"] = record.anomaly_description
            served = int(sum(obs.queue_lengths) * 8)
            state["vehicles_served"] += served
            state["throughput"] = round(sum(obs.queue_lengths) * 60, 1)
            state["wait_time_avg"] = round(sum(obs.queue_lengths) * 45, 1)
            self.total_vehicles_served += served

            # Store last decision
            if record.anomaly_flag or self.step % 20 == 0:
                self.decisions.insert(0, {
                    "time": self._format_sim_time(),
                    "intersection": iid,
                    "phase": PHASE_NAMES.get(action, f"Phase {action}"),
                    "reasoning": record.reasoning[:120],
                    "anomaly": record.anomaly_flag,
                    "confidence": state["ai_confidence"],
                })
                if len(self.decisions) > 50:
                    self.decisions.pop()

            # Trigger event occasionally
            state["event"] = sim._event_active

    async def _tick_builtin(self, hour: int):
        """Built-in simulation when ATLAS modules are unavailable."""
        # Time-of-day arrival intensity
        if 7 <= hour < 9 or 17 <= hour < 19:
            intensity = 0.85
        elif 9 <= hour < 17:
            intensity = 0.45
        else:
            intensity = 0.15

        for iid in INTERSECTION_IDS:
            state = self.states[iid]
            is_override = iid in self.manual_overrides

            # Poisson arrivals per lane
            arrivals = [max(0.0, intensity + random.gauss(0, 0.08)) for _ in range(4)]

            # Green lanes based on phase
            green = {0: [0, 1], 1: [2, 3], 2: [0], 3: [2]}.get(state["phase"], [0, 1])

            new_queue = []
            new_occ = []
            new_speed = []
            served = 0
            for i in range(4):
                prev_q = state["queue"][i]
                inflow = arrivals[i] * 0.05
                if i in green:
                    outflow = min(prev_q, 0.45)
                    served += int(outflow * 10)
                else:
                    outflow = 0.0
                q = max(0.0, min(1.0, prev_q + inflow - outflow + random.gauss(0, 0.01)))
                occ = min(1.0, q * 0.85 + inflow * 0.15)
                spd = max(0.0, 1.0 - occ * 0.9) if i in green else max(0.0, 0.2 - q * 0.2)
                new_queue.append(round(q, 3))
                new_occ.append(round(occ, 3))
                new_speed.append(round(spd, 3))

            state["queue"] = new_queue
            state["occupancy"] = new_occ
            state["speed"] = new_speed
            state["phase_duration"] = state.get("phase_duration", 0) + 1
            state["vehicles_served"] += served
            state["throughput"] = round(sum(new_queue) * 55 + random.gauss(0, 2), 1)
            state["wait_time_avg"] = round(sum(new_queue) * 42, 1)
            self.total_vehicles_served += served
            self.total_co2_saved_kg += served * 0.00012

            # Event
            if random.random() < 0.0005:
                self.event_active[iid] = True
            elif random.random() < 0.003:
                self.event_active[iid] = False
            state["event"] = self.event_active[iid]

            # AI phase logic (simulated)
            if not is_override:
                state["manual_override"] = False
                if state["phase_duration"] > random.randint(20, 60):
                    # Switch to highest-queue phase
                    pairs = [(new_queue[0] + new_queue[1], 0), (new_queue[2] + new_queue[3], 1)]
                    state["phase"] = max(pairs)[1]
                    state["phase_duration"] = 0
            else:
                state["phase"] = self.manual_overrides[iid]
                state["manual_override"] = True

            # Simulated AI Q-values
            q = new_queue
            state["q_values"] = {
                "phase_0": round(-(q[0] + q[1]) + random.gauss(0, 0.02), 4),
                "phase_1": round(-(q[2] + q[3]) + random.gauss(0, 0.02), 4),
                "phase_2": round(-q[0] * 1.2 + random.gauss(0, 0.02), 4),
                "phase_3": round(-q[2] * 1.2 + random.gauss(0, 0.02), 4),
            }
            best_q = max(state["q_values"].values())
            total_q = sum(abs(v) for v in state["q_values"].values()) + 1e-6
            state["ai_confidence"] = round(min(0.99, 0.5 + abs(best_q) / total_q * 0.5), 3)

            # Anomaly detection
            if sum(new_occ) > 3.0 or max(new_occ) > 0.92:
                state["anomaly"] = True
                state["anomaly_desc"] = f"High occupancy detected on {['N','S','E','W'][new_occ.index(max(new_occ))]} lane"
            else:
                state["anomaly"] = False
                state["anomaly_desc"] = ""

            # MUSE decision logs
            if state["anomaly"] or self.step % 25 == 0:
                phase_name = PHASE_NAMES.get(state["phase"], "?")
                best_lane = ["N", "S", "E", "W"][new_queue.index(max(new_queue))]
                self.decisions.insert(0, {
                    "time": self._format_sim_time(),
                    "intersection": iid,
                    "phase": phase_name,
                    "reasoning": f"{'⚠️ ANOMALY – ' if state['anomaly'] else ''}"
                                 f"Selected [{phase_name}]. "
                                 f"Highest load on {best_lane} lane ({int(max(new_queue)*100)}%). "
                                 f"Confidence: {int(state['ai_confidence']*100)}%",
                    "anomaly": state["anomaly"],
                    "confidence": state["ai_confidence"],
                })
                if len(self.decisions) > 50:
                    self.decisions.pop()

    def _update_global_metrics(self, hour: int):
        avg_wait = sum(s["wait_time_avg"] for s in self.states.values()) / len(self.states)
        avg_occ = sum(sum(s["occupancy"]) / 4 for s in self.states.values()) / len(self.states)
        self.metrics_history.append({
            "time": self._format_sim_time(),
            "avg_wait": round(avg_wait, 1),
            "avg_occupancy": round(avg_occ * 100, 1),
            "total_served": self.total_vehicles_served,
            "co2_saved": round(self.total_vehicles_served * 0.00012, 2),
        })
        if len(self.metrics_history) > 200:
            self.metrics_history.pop(0)

    def _format_sim_time(self) -> str:
        h = (self.sim_time // 3600) % 24
        m = (self.sim_time % 3600) // 60
        s = self.sim_time % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── WebSocket Broadcast ────────────────────────────────────────────

    async def _broadcast(self):
        if not self.clients:
            return
        payload = json.dumps(self._build_payload())
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)

    def _build_payload(self) -> dict:
        return {
            "type": "tick",
            "sim_time": self._format_sim_time(),
            "step": self.step,
            "speed": self.speed,
            "paused": self.paused,
            "intersections": list(self.states.values()),
            "decisions": self.decisions[:20],
            "metrics_history": self.metrics_history[-60:],
            "global": {
                "total_vehicles": self.total_vehicles_served,
                "co2_saved_kg": round(self.total_vehicles_served * 0.00012, 2),
                "avg_wait": round(sum(s["wait_time_avg"] for s in self.states.values()) / len(self.states), 1),
                "avg_occupancy": round(sum(sum(s["occupancy"])/4 for s in self.states.values()) / len(self.states) * 100, 1),
                "anomalies_active": sum(1 for s in self.states.values() if s["anomaly"]),
                "atlas_modules": ATLAS_AVAILABLE,
            }
        }


# ── Singleton ─────────────────────────────────────────────────────────
engine = SimulationEngine()


# ── REST API ───────────────────────────────────────────────────────────

class PhaseOverride(BaseModel):
    intersection_id: str
    phase: int           # 0-3, or -1 to release override

class SpeedControl(BaseModel):
    speed: float         # 0.25 to 8.0


@app.on_event("startup")
async def startup():
    asyncio.create_task(engine.run_loop())
    logger.info("ATLAS Pro dashboard server started — http://localhost:8000")


@app.get("/")
async def serve_dashboard():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/api/status")
async def get_status():
    return {
        "running": engine.running,
        "paused": engine.paused,
        "step": engine.step,
        "sim_time": engine._format_sim_time(),
        "intersections": len(INTERSECTION_IDS),
        "atlas_modules": ATLAS_AVAILABLE,
    }


@app.get("/api/intersections")
async def get_intersections():
    return list(engine.states.values())


@app.get("/api/decisions")
async def get_decisions():
    return engine.decisions[:30]


@app.get("/api/metrics")
async def get_metrics():
    return engine.metrics_history[-100:]


@app.post("/api/pause")
async def toggle_pause():
    engine.paused = not engine.paused
    return {"paused": engine.paused}


@app.post("/api/speed")
async def set_speed(body: SpeedControl):
    engine.speed = max(0.1, min(10.0, body.speed))
    return {"speed": engine.speed}


@app.post("/api/override")
async def set_override(body: PhaseOverride):
    if body.phase == -1:
        engine.manual_overrides.pop(body.intersection_id, None)
        return {"status": "released", "intersection": body.intersection_id}
    if 0 <= body.phase <= 3:
        engine.manual_overrides[body.intersection_id] = body.phase
        return {"status": "override_set", "intersection": body.intersection_id, "phase": body.phase}
    return {"error": "Invalid phase (0-3 or -1 to release)"}


@app.post("/api/reset")
async def reset_simulation():
    engine.step = 0
    engine.sim_time = 28800
    engine.total_vehicles_served = 0
    engine.total_co2_saved_kg = 0
    engine.manual_overrides.clear()
    engine.decisions.clear()
    engine.metrics_history.clear()
    for i, iid in enumerate(INTERSECTION_IDS):
        engine.states[iid] = engine._make_initial_state(iid, i)
    return {"status": "reset"}


@app.get("/api/export")
async def export_report():
    """Generates a CSV report with all current data for offline analysis."""
    lines = []
    lines.append("ATLAS PRO - SYSTEM TELEMETRY REPORT")
    lines.append(f"Generated at Step: {engine.step} / Sim Time: {engine._format_sim_time()}")
    lines.append("")
    
    # 1. Global Metrics
    lines.append("--- GLOBAL METRICS ---")
    lines.append(f"Total Vehicles Served,{engine.total_vehicles_served}")
    lines.append(f"CO2 Saved (kg),{round(engine.total_co2_saved_kg, 2)}")
    lines.append("")
    
    # 2. Node States
    lines.append("--- INTERSECTION NODE STATES ---")
    lines.append("ID,Phase,WaitTime(s),Throughput,OccupancyAVG,Confidence,Anomaly")
    for s in engine.states.values():
        occ_avg = sum(s["occupancy"])/4
        lines.append(f'{s["id"]},P{s["phase"]},{s["wait_time_avg"]},{s["throughput"]},{round(occ_avg*100,1)}%,{round(s["ai_confidence"]*100,1)}%,{s["anomaly"]}')
    lines.append("")
    
    # 3. MUSE Decisions
    lines.append("--- RECENT AI DECISION LOG (MUSE) ---")
    lines.append("Time,Node,Phase,Confidence,Anomaly,Reasoning")
    for d in engine.decisions:
        r_clean = d['reasoning'].replace(',', ';')  # Format safe for CSV
        lines.append(f"{d['time']},{d['intersection']},{d['phase']},{round(d['confidence']*100,1)}%,{d['anomaly']},{r_clean}")
        
    csv_data = "\n".join(lines)
    
    return PlainTextResponse(
        content=csv_data,
        headers={"Content-Disposition": f"attachment; filename=atlas_report_{int(time.time())}.csv"}
    )


# ── WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    engine.clients.append(ws)
    logger.info(f"Dashboard client connected ({len(engine.clients)} total)")
    try:
        # Send initial state immediately
        await ws.send_text(json.dumps(engine._build_payload()))
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            # Handle client commands via WebSocket too
            if msg.get("type") == "override":
                iid = msg.get("intersection_id")
                phase = msg.get("phase", -1)
                if phase == -1:
                    engine.manual_overrides.pop(iid, None)
                else:
                    engine.manual_overrides[iid] = phase
            elif msg.get("type") == "pause":
                engine.paused = not engine.paused
            elif msg.get("type") == "speed":
                engine.speed = max(0.1, min(10.0, float(msg.get("speed", 1.0))))
    except WebSocketDisconnect:
        engine.clients.remove(ws)
        logger.info(f"Dashboard client disconnected ({len(engine.clients)} remaining)")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
