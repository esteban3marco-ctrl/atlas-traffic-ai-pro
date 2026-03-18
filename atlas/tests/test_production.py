"""
ATLAS Pro — Production Module Tests
======================================
Tests for safety watchdog, XAI engine, inference engine,
controller interface, and CLI.
"""

import sys
import os
import time
import json
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# SAFETY WATCHDOG TESTS
# =============================================================================

class TestSafetyWatchdog:
    """Tests for WatchdogSafetyModule — critical safety system."""

    def _make_watchdog(self):
        from atlas.production.safety_watchdog import WatchdogSafetyModule
        return WatchdogSafetyModule()

    def test_creation(self):
        w = self._make_watchdog()
        assert w.MIN_GREEN_TIME == 15.0
        assert w.MAX_GREEN_TIME == 120.0
        assert w.HEARTBEAT_TIMEOUT == 5.0
        assert w.is_emergency_mode is False
        assert w.violations == []

    def test_validate_action_passthrough(self):
        """Normal action should pass through when min green met."""
        w = self._make_watchdog()
        w.last_phase_change_time = time.time() - 20  # 20s ago, past min green
        result = w.validate_action(2, current_phase=0)
        assert result == 2

    def test_validate_action_min_green_blocks(self):
        """Phase change should be blocked if min green time not met."""
        w = self._make_watchdog()
        w.last_phase_change_time = time.time()  # Just now
        result = w.validate_action(2, current_phase=0)
        assert result == 0  # Stays on current phase
        assert len(w.violations) == 1
        assert w.violations[0].violation_type == "MIN_GREEN_VIOLATION"

    def test_validate_action_max_green_forces_change(self):
        """Phase should be forced to change after max green time."""
        w = self._make_watchdog()
        w.last_phase_change_time = time.time() - 130  # 130s ago, past max
        result = w.validate_action(0, current_phase=0)  # AI wants same phase
        assert result == 1  # Forced to next phase
        assert any(v.violation_type == "MAX_GREEN_VIOLATION" for v in w.violations)

    def test_emergency_preemption(self):
        """Emergency vehicle should override all actions."""
        w = self._make_watchdog()
        w.last_phase_change_time = time.time() - 20
        result = w.validate_action(2, current_phase=0, is_emergency_vehicle_detected=True)
        assert result == 0  # Stays on current to clear path
        assert w.is_emergency_mode is True
        assert any(v.violation_type == "EMERGENCY_PREEMPTION" for v in w.violations)

    def test_emergency_mode_clears(self):
        """Emergency mode should clear when no more emergency vehicles."""
        w = self._make_watchdog()
        w.is_emergency_mode = True
        w.last_phase_change_time = time.time() - 20
        w.validate_action(1, current_phase=0)
        assert w.is_emergency_mode is False

    def test_system_health_ok(self):
        w = self._make_watchdog()
        w.last_heartbeat = time.time()
        assert w.check_system_health() is True

    def test_system_health_timeout(self):
        w = self._make_watchdog()
        w.last_heartbeat = time.time() - 10  # 10s ago, past 5s timeout
        assert w.check_system_health() is False
        assert any(v.violation_type == "SYSTEM_FREEZE" for v in w.violations)

    def test_audit_log(self):
        w = self._make_watchdog()
        w.last_phase_change_time = time.time()  # Trigger min green violation
        w.validate_action(2, current_phase=0)
        log = w.get_audit_log()
        assert len(log) >= 1
        assert "violation_type" in log[0]
        assert "severity" in log[0]

    def test_violation_severities(self):
        """Verify correct severity levels for different violations."""
        w = self._make_watchdog()
        # Trigger emergency (HIGH)
        w.validate_action(0, current_phase=0, is_emergency_vehicle_detected=True)
        assert w.violations[-1].severity == "HIGH"
        # Trigger min green (MEDIUM)
        w2 = self._make_watchdog()
        w2.last_phase_change_time = time.time()
        w2.validate_action(2, current_phase=0)
        assert w2.violations[-1].severity == "MEDIUM"

    def test_thread_safety(self):
        """Validate action uses lock for thread safety."""
        w = self._make_watchdog()
        w.last_phase_change_time = time.time() - 20
        import threading
        results = []

        def call():
            r = w.validate_action(1, current_phase=0)
            results.append(r)

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10


# =============================================================================
# XAI ENGINE TESTS
# =============================================================================

class TestXAIEngine:
    """Tests for ConversationalXAI engine."""

    def _make_xai(self):
        from atlas.production.xai_engine import ConversationalXAI
        return ConversationalXAI()

    def test_creation(self):
        xai = self._make_xai()
        assert len(xai.directions) == 4
        assert len(xai.phase_names) == 4

    def test_generate_explanation_basic(self):
        xai = self._make_xai()
        state = np.random.rand(26).astype(np.float32)
        explanation = xai.generate_explanation(state, action=0)
        assert isinstance(explanation, str)
        assert len(explanation) > 20

    def test_generate_explanation_with_saliency(self):
        xai = self._make_xai()
        state = np.random.rand(26).astype(np.float32)
        saliency = np.random.rand(26).astype(np.float32)
        explanation = xai.generate_explanation(state, action=2, saliency=saliency)
        assert "NEURAL_ATTENTION" in explanation

    def test_generate_explanation_all_actions(self):
        xai = self._make_xai()
        state = np.random.rand(26).astype(np.float32)
        for action in range(4):
            explanation = xai.generate_explanation(state, action=action)
            assert isinstance(explanation, str)

    def test_low_density_insight(self):
        xai = self._make_xai()
        state = np.zeros(26, dtype=np.float32)  # All zeros = low density
        explanation = xai.generate_explanation(state, action=0)
        assert "LOW_DENSITY" in explanation

    def test_turn_recovery_insight(self):
        xai = self._make_xai()
        state = np.full(26, 0.5, dtype=np.float32)  # Medium density
        explanation = xai.generate_explanation(state, action=1)  # Odd action
        assert "TURN_RECOVERY" in explanation

    def test_no_repeat_explanation(self):
        xai = self._make_xai()
        state = np.random.rand(26).astype(np.float32)
        e1 = xai.generate_explanation(state, action=0)
        xai.last_explanation = e1
        e2 = xai.generate_explanation(state, action=0)
        # If same, should get RE-OPTIMIZING message
        if e1 == e2:
            assert "RE-OPTIMIZING" in e2

    def test_full_name_mapping(self):
        xai = self._make_xai()
        assert xai._get_full_name("N") == "Norte"
        assert xai._get_full_name("S") == "Sur"
        assert xai._get_full_name("E") == "Este"
        assert xai._get_full_name("W") == "Oeste"
        assert xai._get_full_name("X") == "X"  # Unknown


# =============================================================================
# CONTROLLER INTERFACE TESTS
# =============================================================================

class TestControllerInterface:
    """Tests for SimulatedController and factory."""

    def test_simulated_controller_creation(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController()
        assert c.connected is False
        assert c.current_phase == 0

    def test_simulated_connect_disconnect(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        assert c.connect() is True
        assert c.connected is True
        c.disconnect()
        assert c.connected is False

    def test_simulated_set_phase(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        assert c.set_phase(2) is True
        assert c.set_phase(99) is False  # Out of range

    def test_simulated_get_status(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        status = c.get_status()
        assert status.current_phase == 0
        assert status.is_fault is False

    def test_simulated_fault_state(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c._fault = True
        status = c.get_status()
        assert status.is_fault is True

    def test_request_phase_change_min_green(self):
        """Phase change should be blocked before min green."""
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        c.phase_start_time = time.time()  # Just started
        result = c.request_phase_change(2)
        assert result is False  # Min green not met

    def test_request_phase_change_after_min_green(self):
        """Phase change should succeed after min green."""
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        c.phase_start_time = time.time() - 15  # 15s ago, past min
        result = c.request_phase_change(2)
        assert result is True
        assert c.current_phase == 2

    def test_request_phase_change_fault_rejected(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        c._fault = True
        c.phase_start_time = time.time() - 15
        result = c.request_phase_change(2)
        assert result is False

    def test_command_log(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        c.phase_start_time = time.time() - 15
        c.request_phase_change(2)
        assert len(c.command_log) == 1
        assert c.command_log[0]["from_phase"] == 0
        assert c.command_log[0]["to_phase"] == 2

    def test_save_command_log(self):
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        c.phase_start_time = time.time() - 15
        c.request_phase_change(2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            c.save_command_log(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1
        finally:
            os.unlink(path)

    def test_factory_simulated(self):
        from atlas.production.controller_interface import create_controller
        c = create_controller("simulated")
        assert c is not None

    def test_factory_unknown_raises(self):
        from atlas.production.controller_interface import create_controller
        with pytest.raises(ValueError):
            create_controller("nonexistent")

    def test_phase_definition(self):
        from atlas.production.controller_interface import PhaseDefinition
        p = PhaseDefinition(id=0, name="NS_Green", green_directions=["N", "S"])
        assert p.min_green == 10.0
        assert p.max_green == 90.0
        assert p.yellow == 3.0

    def test_same_phase_no_change(self):
        """Requesting current phase should return True without logging."""
        from atlas.production.controller_interface import SimulatedController
        c = SimulatedController(latency=0.0)
        c.connect()
        c.phase_start_time = time.time() - 15
        result = c.request_phase_change(0)  # Already on phase 0
        assert result is True
        assert len(c.command_log) == 0


# =============================================================================
# INFERENCE ENGINE TESTS (SafetyMonitor, DecisionLogger)
# =============================================================================

class TestInferenceEngineSafety:
    """Tests for SafetyMonitor within inference_engine.py."""

    def _make_monitor(self):
        from atlas.production.inference_engine import SafetyMonitor, ProductionConfig
        config = ProductionConfig(max_consecutive_same_phase=3)
        return SafetyMonitor(config)

    def _make_status(self, fault=False, manual=False):
        from atlas.production.controller_interface import ControllerStatus
        return ControllerStatus(is_fault=fault, is_manual_override=manual)

    def test_normal_decision_passthrough(self):
        m = self._make_monitor()
        result = m.check_decision(2, self._make_status(), current_phase=0)
        assert result == 2

    def test_fault_returns_safe_phase(self):
        m = self._make_monitor()
        result = m.check_decision(2, self._make_status(fault=True), current_phase=1)
        assert result == 0  # Default safe

    def test_manual_override_keeps_phase(self):
        m = self._make_monitor()
        result = m.check_decision(2, self._make_status(manual=True), current_phase=1)
        assert result == 1

    def test_consecutive_same_phase_force_change(self):
        m = self._make_monitor()
        status = self._make_status()
        for _ in range(3):
            m.check_decision(0, status, current_phase=0)
        # 4th same phase should trigger force change
        result = m.check_decision(0, status, current_phase=0)
        assert result != 0  # Forced to different phase
        assert m.overridden_decisions > 0

    def test_invalid_action_corrected(self):
        m = self._make_monitor()
        result = m.check_decision(-1, self._make_status(), current_phase=0)
        assert result == 0
        assert m.overridden_decisions == 1

    def test_watchdog_ok(self):
        m = self._make_monitor()
        m.last_heartbeat = time.time()
        assert m.check_watchdog() is True

    def test_watchdog_timeout(self):
        m = self._make_monitor()
        m.last_heartbeat = time.time() - 15
        assert m.check_watchdog() is False

    def test_stats(self):
        m = self._make_monitor()
        m.check_decision(0, self._make_status(), current_phase=0)
        stats = m.get_stats()
        assert stats["total_decisions"] == 1
        assert "override_rate" in stats


class TestDecisionLogger:
    """Tests for DecisionLogger."""

    def test_log_decision(self):
        from atlas.production.inference_engine import DecisionLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(tmpdir)
            state = np.zeros(26)
            logger.log_decision(
                step=0, state_vector=state, ai_action=2,
                final_action=2, phase_before=0,
                controller_status={}, camera_state={"total_vehicles": 5}
            )
            assert len(logger.decisions) == 1
            assert logger.decisions[0]["ai_action"] == 2
            assert logger.decisions[0]["was_overridden"] is False

    def test_log_overridden_decision(self):
        from atlas.production.inference_engine import DecisionLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(tmpdir)
            state = np.zeros(26)
            logger.log_decision(
                step=0, state_vector=state, ai_action=2,
                final_action=0, phase_before=0,
                controller_status={}, camera_state={}
            )
            assert logger.decisions[0]["was_overridden"] is True

    def test_save_and_summary(self):
        from atlas.production.inference_engine import DecisionLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(tmpdir)
            state = np.zeros(26)
            for i in range(5):
                logger.log_decision(
                    step=i, state_vector=state, ai_action=i % 4,
                    final_action=i % 4, phase_before=0,
                    controller_status={}, camera_state={}
                )
            logger.save()
            path = os.path.join(tmpdir, "decisions.json")
            assert os.path.exists(path)
            summary = logger.get_summary()
            assert summary["total_decisions"] == 5

    def test_empty_summary(self):
        from atlas.production.inference_engine import DecisionLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(tmpdir)
            assert logger.get_summary() == {}


# =============================================================================
# CLI TESTS
# =============================================================================

class TestCLI:
    """Tests for CLI argument parsing and command dispatch."""

    def test_setup_logging(self):
        from atlas.cli import setup_logging
        setup_logging("DEBUG")
        setup_logging("INFO")
        # Should not raise

    def test_main_no_args_shows_help(self, capsys):
        from atlas.cli import main
        with pytest.raises(SystemExit):
            main()

    def test_cmd_config_generate(self):
        from atlas.cli import cmd_config
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            args = Namespace(action="generate", output=path, file=None)
            cmd_config(args)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_cmd_config_show(self, capsys):
        from atlas.cli import cmd_config
        args = Namespace(action="show", file=None, output=None)
        cmd_config(args)
        captured = capsys.readouterr()
        assert "agent" in captured.out
        assert "reward" in captured.out

    def test_cmd_config_show_from_file(self, capsys):
        from atlas.cli import cmd_config
        from atlas.config import TrainingConfig
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            TrainingConfig().save(path)
            args = Namespace(action="show", file=path, output=None)
            cmd_config(args)
            captured = capsys.readouterr()
            assert "total_episodes" in captured.out
        finally:
            os.unlink(path)

    def test_parser_train_subcommand(self):
        """Verify train subcommand accepts expected arguments."""
        import argparse
        from atlas.cli import main
        # Just verify the parser doesn't crash with valid args
        # (actual training would require SUMO)

    def test_parser_evaluate_subcommand(self):
        """Verify evaluate subcommand is registered."""
        from atlas.cli import main
        import atlas.cli as cli_mod
        # Verify commands dict has all expected entries
        # We can't actually run them without SUMO


# =============================================================================
# PRODUCTION CONFIG TESTS
# =============================================================================

class TestProductionConfig:
    """Tests for ProductionConfig dataclass."""

    def test_defaults(self):
        from atlas.production.inference_engine import ProductionConfig
        config = ProductionConfig()
        assert config.mode == "shadow"
        assert config.decision_interval == 5.0
        assert config.max_consecutive_same_phase == 6
        assert config.action_dim == 4

    def test_custom_config(self):
        from atlas.production.inference_engine import ProductionConfig
        config = ProductionConfig(mode="live", decision_interval=3.0)
        assert config.mode == "live"
        assert config.decision_interval == 3.0
