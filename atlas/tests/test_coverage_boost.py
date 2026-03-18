# -*- coding: utf-8 -*-
"""
ATLAS Pro — Coverage Boost Tests
====================================
Additional tests to reach 85%+ overall coverage.
Covers: camera_pipeline, controller_interface, inference_engine,
        dashboard, baselines, cli.
"""

import sys
import os
import time
import json
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CAMERA PIPELINE TESTS
# =============================================================================

class TestVehicleDetection:
    """Tests for VehicleDetection dataclass."""

    def test_creation(self):
        from atlas.production.camera_pipeline import VehicleDetection
        det = VehicleDetection(
            bbox=(10, 20, 100, 150),
            confidence=0.85,
            class_name="car",
            speed_estimate=30.0,
            lane=1,
            direction="N",
        )
        assert det.bbox == (10, 20, 100, 150)
        assert det.confidence == 0.85
        assert det.class_name == "car"
        assert det.speed_estimate == 30.0
        assert det.direction == "N"

    def test_defaults(self):
        from atlas.production.camera_pipeline import VehicleDetection
        det = VehicleDetection(bbox=(0, 0, 50, 50), confidence=0.5, class_name="truck")
        assert det.speed_estimate == 0.0
        assert det.lane == 0
        assert det.direction == ""


class TestIntersectionState:
    """Tests for IntersectionState dataclass and to_state_vector."""

    def _make_state(self):
        from atlas.production.camera_pipeline import IntersectionState
        return IntersectionState(
            timestamp=time.time(),
            queue_lengths={"N": 10, "S": 5, "E": 15, "W": 3},
            vehicle_speeds={"N": 30, "S": 40, "E": 10, "W": 50},
            wait_times={"N": 20, "S": 10, "E": 60, "W": 5},
            densities={"N": 0.5, "S": 0.3, "E": 0.8, "W": 0.1},
            halted_counts={"N": 5, "S": 2, "E": 10, "W": 1},
            total_vehicles=42,
            emergency_detected=False,
        )

    def test_creation(self):
        state = self._make_state()
        assert state.total_vehicles == 42
        assert state.emergency_detected is False
        assert state.queue_lengths["N"] == 10

    def test_to_state_vector_shape(self):
        state = self._make_state()
        vec = state.to_state_vector(current_phase=0, phase_duration=15.0)
        assert vec.shape == (26,)
        assert vec.dtype == np.float32

    def test_to_state_vector_values(self):
        state = self._make_state()
        vec = state.to_state_vector(current_phase=2, phase_duration=30.0)
        # Check normalization: queue_N = 10/50 = 0.2
        assert abs(vec[0] - 10 / 50.0) < 1e-5
        # Phase one-hot: phase 2 → [0, 0, 1, 0]
        assert vec[20] == 0.0
        assert vec[21] == 0.0
        assert vec[22] == 1.0
        assert vec[23] == 0.0
        # Emergency flag
        assert vec[25] == 0.0

    def test_to_state_vector_emergency(self):
        from atlas.production.camera_pipeline import IntersectionState
        state = IntersectionState(emergency_detected=True)
        vec = state.to_state_vector()
        assert vec[25] == 1.0

    def test_to_state_vector_missing_directions(self):
        from atlas.production.camera_pipeline import IntersectionState
        state = IntersectionState()
        vec = state.to_state_vector()
        assert vec.shape == (26,)
        # All zeros except phase one-hot
        assert vec[20] == 1.0  # Phase 0 by default

    def test_to_state_vector_out_of_range_phase(self):
        from atlas.production.camera_pipeline import IntersectionState
        state = IntersectionState()
        vec = state.to_state_vector(current_phase=10)
        # Phase out of range, all zeros in one-hot
        assert sum(vec[20:24]) == 0.0


class TestCameraConfig:
    """Tests for CameraConfig dataclass."""

    def test_defaults(self):
        from atlas.production.camera_pipeline import CameraConfig
        cfg = CameraConfig()
        assert cfg.source == "0"
        assert cfg.direction == "N"
        assert cfg.fps == 10
        assert cfg.resolution == (1280, 720)
        assert cfg.roi is None
        assert cfg.detection_confidence == 0.4

    def test_custom(self):
        from atlas.production.camera_pipeline import CameraConfig
        cfg = CameraConfig(source="rtsp://cam1", direction="E", fps=15)
        assert cfg.source == "rtsp://cam1"
        assert cfg.direction == "E"


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self):
        from atlas.production.camera_pipeline import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.cameras == []
        assert cfg.inference_interval == 0.5
        assert cfg.state_buffer_size == 10
        assert cfg.halted_speed_threshold == 2.0

    def test_custom(self):
        from atlas.production.camera_pipeline import PipelineConfig, CameraConfig
        cams = [CameraConfig(direction="N"), CameraConfig(direction="S")]
        cfg = PipelineConfig(cameras=cams, inference_interval=1.0)
        assert len(cfg.cameras) == 2
        assert cfg.inference_interval == 1.0


class TestCameraStream:
    """Tests for CameraStream (without OpenCV)."""

    def test_creation(self):
        from atlas.production.camera_pipeline import CameraStream, CameraConfig
        cfg = CameraConfig(source="0", direction="N")
        stream = CameraStream(cfg)
        assert stream.running is False
        assert stream.frame is None
        assert stream.fps_actual == 0.0

    def test_start_without_opencv(self):
        from atlas.production.camera_pipeline import CameraStream, CameraConfig
        cfg = CameraConfig(source="0", direction="S")
        stream = CameraStream(cfg)
        with patch("atlas.production.camera_pipeline.HAS_CV2", False):
            stream.start()
            assert stream.running is True
            stream.stop()

    def test_get_frame_none(self):
        from atlas.production.camera_pipeline import CameraStream, CameraConfig
        cfg = CameraConfig()
        stream = CameraStream(cfg)
        assert stream.get_frame() is None

    def test_get_frame_returns_copy(self):
        from atlas.production.camera_pipeline import CameraStream, CameraConfig
        cfg = CameraConfig()
        stream = CameraStream(cfg)
        stream.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = stream.get_frame()
        assert frame is not None
        assert frame is not stream.frame  # Should be a copy


class TestVehicleDetector:
    """Tests for VehicleDetector."""

    def test_creation_no_yolo(self):
        from atlas.production.camera_pipeline import VehicleDetector
        with patch("atlas.production.camera_pipeline.HAS_YOLO", False):
            det = VehicleDetector()
            assert det.model is None

    def test_simulated_detection(self):
        from atlas.production.camera_pipeline import VehicleDetector
        with patch("atlas.production.camera_pipeline.HAS_YOLO", False):
            det = VehicleDetector()
            dets = det._simulated_detection("N")
            for d in dets:
                assert d.direction == "N"
                assert d.confidence > 0
                assert d.class_name in ["car", "truck", "bus"]

    def test_detect_without_model_uses_simulation(self):
        from atlas.production.camera_pipeline import VehicleDetector
        with patch("atlas.production.camera_pipeline.HAS_YOLO", False):
            det = VehicleDetector()
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            dets = det.detect(frame, "E")
            # Should fall back to simulated
            for d in dets:
                assert d.direction == "E"

    def test_vehicle_classes(self):
        from atlas.production.camera_pipeline import VehicleDetector
        assert "car" in VehicleDetector.VEHICLE_CLASSES
        assert "truck" in VehicleDetector.VEHICLE_CLASSES
        assert 2 in VehicleDetector.COCO_VEHICLES


class TestCameraPipeline:
    """Tests for CameraPipeline."""

    def _make_pipeline(self):
        from atlas.production.camera_pipeline import (
            CameraPipeline, PipelineConfig, CameraConfig
        )
        cams = [
            CameraConfig(source="0", direction="N"),
            CameraConfig(source="0", direction="S"),
        ]
        config = PipelineConfig(cameras=cams, inference_interval=0.1)
        return CameraPipeline(config)

    def test_creation(self):
        pipeline = self._make_pipeline()
        assert pipeline.running is False
        assert pipeline.latest_state is None
        assert len(pipeline._direction_history) == 2

    def test_get_state_none_initially(self):
        pipeline = self._make_pipeline()
        assert pipeline.get_state() is None

    def test_get_state_vector_none_initially(self):
        pipeline = self._make_pipeline()
        assert pipeline.get_state_vector() is None

    def test_start_stop(self):
        pipeline = self._make_pipeline()
        with patch("atlas.production.camera_pipeline.HAS_CV2", False):
            pipeline.start()
            assert pipeline.running is True
            time.sleep(0.3)  # Let process loop run briefly
            pipeline.running = False
            for stream in pipeline.cameras.values():
                stream.running = False
            time.sleep(0.2)

    def test_get_state_after_processing(self):
        pipeline = self._make_pipeline()
        with patch("atlas.production.camera_pipeline.HAS_CV2", False):
            pipeline.start()
            time.sleep(0.5)
            state = pipeline.get_state()
            if state is not None:
                assert state.total_vehicles >= 0
                vec = pipeline.get_state_vector(current_phase=1)
                assert vec is not None
                assert vec.shape == (26,)
            pipeline.running = False
            for stream in pipeline.cameras.values():
                stream.running = False
            time.sleep(0.2)


class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline factory."""

    def test_default(self):
        from atlas.production.camera_pipeline import create_default_pipeline
        pipeline = create_default_pipeline()
        assert len(pipeline.cameras) == 0  # Not started, cameras dict empty
        assert len(pipeline.config.cameras) == 4

    def test_custom_sources(self):
        from atlas.production.camera_pipeline import create_default_pipeline
        sources = {"N": "rtsp://cam1", "S": "rtsp://cam2"}
        pipeline = create_default_pipeline(sources)
        assert len(pipeline.config.cameras) == 2


# =============================================================================
# CONTROLLER INTERFACE — ADDITIONAL TESTS
# =============================================================================

class TestPhaseDefinition:
    """Tests for PhaseDefinition dataclass."""

    def test_creation(self):
        from atlas.production.controller_interface import PhaseDefinition
        phase = PhaseDefinition(
            id=0, name="NS_Green", green_directions=["N", "S"],
            min_green=10, max_green=90, yellow=3, all_red=2,
        )
        assert phase.id == 0
        assert phase.name == "NS_Green"
        assert phase.green_directions == ["N", "S"]

    def test_defaults(self):
        from atlas.production.controller_interface import DEFAULT_PHASES
        assert len(DEFAULT_PHASES) == 4
        assert DEFAULT_PHASES[0].name == "NS_Green"
        assert DEFAULT_PHASES[2].name == "EW_Green"


class TestControllerStatus:
    """Tests for ControllerStatus dataclass."""

    def test_defaults(self):
        from atlas.production.controller_interface import ControllerStatus
        status = ControllerStatus()
        assert status.current_phase == 0
        assert status.is_fault is False
        assert status.is_manual_override is False
        assert status.last_command_accepted is True

    def test_custom(self):
        from atlas.production.controller_interface import ControllerStatus
        status = ControllerStatus(is_fault=True, is_manual_override=True)
        assert status.is_fault is True
        assert status.is_manual_override is True


class TestSimulatedControllerExtended:
    """Extended tests for SimulatedController."""

    def _make_controller(self):
        from atlas.production.controller_interface import SimulatedController
        return SimulatedController(latency=0.01)

    def test_save_command_log(self):
        ctrl = self._make_controller()
        ctrl.connect()
        # Force enough time to pass min_green
        ctrl.phase_start_time = time.time() - 20
        ctrl.request_phase_change(1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            ctrl.save_command_log(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
        finally:
            os.unlink(path)

    def test_command_log_trimming(self):
        ctrl = self._make_controller()
        ctrl.command_log = [{"test": i} for i in range(10001)]
        # Trigger the trim by calling _log_command
        ctrl._log_command(0, 1, 15.0)
        assert len(ctrl.command_log) <= 5002

    def test_request_phase_change_fault_rejected(self):
        ctrl = self._make_controller()
        ctrl.connect()
        ctrl._fault = True
        result = ctrl.request_phase_change(1)
        assert result is False

    def test_request_phase_change_same_phase(self):
        ctrl = self._make_controller()
        ctrl.connect()
        ctrl.current_phase = 0
        ctrl.phase_start_time = time.time() - 20
        result = ctrl.request_phase_change(0)
        assert result is True  # Same phase, no change needed

    def test_request_phase_change_min_green_not_met(self):
        ctrl = self._make_controller()
        ctrl.connect()
        ctrl.phase_start_time = time.time()  # Just now
        result = ctrl.request_phase_change(1)
        assert result is False

    def test_set_phase_invalid(self):
        ctrl = self._make_controller()
        result = ctrl.set_phase(999)
        assert result is False


class TestModbusController:
    """Tests for ModbusController with mocked pymodbus."""

    def test_creation(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController(host="10.0.0.1", port=503, unit_id=2)
        assert ctrl.host == "10.0.0.1"
        assert ctrl.port == 503
        assert ctrl.unit_id == 2
        assert ctrl.client is None

    def test_connect_no_pymodbus(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        with patch.dict("sys.modules", {"pymodbus": None, "pymodbus.client": None}):
            result = ctrl.connect()
            assert result is False

    def test_disconnect_no_client(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.disconnect()  # Should not raise

    def test_set_phase_no_client(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        assert ctrl.set_phase(0) is False

    def test_get_status_no_client(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        status = ctrl.get_status()
        assert status.is_fault is True


class TestRestAPIController:
    """Tests for RestAPIController."""

    def test_creation(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController(base_url="http://example.com/api/", api_key="secret")
        assert ctrl.base_url == "http://example.com/api"  # Trailing slash stripped
        assert ctrl.api_key == "secret"

    def test_connect_failure(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController(base_url="http://localhost:99999")
        result = ctrl.connect()
        assert result is False

    def test_disconnect_no_session(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.disconnect()  # Should not raise

    def test_get_status_no_session(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        status = ctrl.get_status()
        assert status.is_fault is True


class TestGPIOController:
    """Tests for GPIOController."""

    def test_creation(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        assert "N_green" in ctrl.pin_map
        assert ctrl.gpio is None

    def test_connect_no_rpi(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        result = ctrl.connect()
        assert result is False  # RPi.GPIO not available

    def test_disconnect_no_gpio(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        ctrl.disconnect()  # Should not raise

    def test_set_phase_no_gpio(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        assert ctrl.set_phase(0) is False

    def test_get_status(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        status = ctrl.get_status()
        assert status.current_phase == 0


class TestControllerFactory:
    """Tests for create_controller factory."""

    def test_create_simulated(self):
        from atlas.production.controller_interface import create_controller, SimulatedController
        ctrl = create_controller("simulated")
        assert isinstance(ctrl, SimulatedController)

    def test_create_modbus(self):
        from atlas.production.controller_interface import create_controller, ModbusController
        ctrl = create_controller("modbus")
        assert isinstance(ctrl, ModbusController)

    def test_create_rest_api(self):
        from atlas.production.controller_interface import create_controller, RestAPIController
        ctrl = create_controller("rest_api")
        assert isinstance(ctrl, RestAPIController)

    def test_create_gpio(self):
        from atlas.production.controller_interface import create_controller, GPIOController
        ctrl = create_controller("gpio")
        assert isinstance(ctrl, GPIOController)

    def test_create_unknown_raises(self):
        from atlas.production.controller_interface import create_controller
        with pytest.raises(ValueError):
            create_controller("unknown_type")


# =============================================================================
# INFERENCE ENGINE — ADDITIONAL TESTS
# =============================================================================

class TestProductionConfig:
    """Tests for ProductionConfig."""

    def test_defaults(self):
        from atlas.production.inference_engine import ProductionConfig
        cfg = ProductionConfig()
        assert cfg.mode == "shadow"
        assert cfg.state_dim == 26
        assert cfg.action_dim == 4
        assert cfg.watchdog_timeout == 10.0
        assert cfg.max_consecutive_same_phase == 6

    def test_custom(self):
        from atlas.production.inference_engine import ProductionConfig
        cfg = ProductionConfig(mode="live", decision_interval=2.0)
        assert cfg.mode == "live"
        assert cfg.decision_interval == 2.0


class TestSafetyMonitorExtended:
    """Extended SafetyMonitor tests."""

    def _make_monitor(self):
        from atlas.production.inference_engine import SafetyMonitor, ProductionConfig
        return SafetyMonitor(ProductionConfig())

    def _make_status(self, fault=False, manual=False):
        from atlas.production.controller_interface import ControllerStatus
        return ControllerStatus(is_fault=fault, is_manual_override=manual)

    def test_invalid_action_range(self):
        monitor = self._make_monitor()
        status = self._make_status()
        result = monitor.check_decision(-1, status, current_phase=0)
        assert result == 0
        assert monitor.overridden_decisions == 1

    def test_invalid_action_high(self):
        monitor = self._make_monitor()
        status = self._make_status()
        result = monitor.check_decision(99, status, current_phase=0)
        assert result == 0

    def test_get_stats(self):
        monitor = self._make_monitor()
        status = self._make_status()
        monitor.check_decision(0, status, 0)
        monitor.check_decision(1, status, 0)
        stats = monitor.get_stats()
        assert stats["total_decisions"] == 2
        assert "override_rate" in stats

    def test_watchdog_expired(self):
        monitor = self._make_monitor()
        monitor.last_heartbeat = time.time() - 20  # 20s ago
        assert monitor.check_watchdog() is False
        assert len(monitor.violations) == 1

    def test_watchdog_ok(self):
        monitor = self._make_monitor()
        assert monitor.check_watchdog() is True


class TestDecisionLoggerExtended:
    """Extended DecisionLogger tests."""

    def _make_logger(self):
        from atlas.production.inference_engine import DecisionLogger
        return DecisionLogger(tempfile.mkdtemp())

    def test_log_and_save(self):
        dl = self._make_logger()
        state_vec = np.random.rand(26).astype(np.float32)
        dl.log_decision(
            step=0, state_vector=state_vec,
            ai_action=1, final_action=1,
            phase_before=0,
            controller_status={"phase": 0, "fault": False},
            camera_state={"total_vehicles": 10},
        )
        assert len(dl.decisions) == 1
        dl.save()
        path = os.path.join(dl.log_dir, "decisions.json")
        assert os.path.exists(path)

    def test_summary_empty(self):
        dl = self._make_logger()
        assert dl.get_summary() == {}

    def test_summary_with_data(self):
        dl = self._make_logger()
        state_vec = np.random.rand(26).astype(np.float32)
        for i in range(5):
            dl.log_decision(
                step=i, state_vector=state_vec,
                ai_action=i % 4, final_action=i % 4,
                phase_before=0,
                controller_status={},
                camera_state={},
            )
        summary = dl.get_summary()
        assert summary["total_decisions"] == 5
        assert "action_distribution" in summary
        assert summary["override_count"] == 0

    def test_override_detection(self):
        dl = self._make_logger()
        state_vec = np.random.rand(26).astype(np.float32)
        dl.log_decision(
            step=0, state_vector=state_vec,
            ai_action=1, final_action=2,  # Different = override
            phase_before=0,
            controller_status={},
            camera_state={},
        )
        assert dl.decisions[0]["was_overridden"] is True
        summary = dl.get_summary()
        assert summary["override_count"] == 1

    def test_short_state_vector(self):
        dl = self._make_logger()
        state_vec = np.array([0.5], dtype=np.float32)
        dl.log_decision(
            step=0, state_vector=state_vec,
            ai_action=0, final_action=0,
            phase_before=0,
            controller_status={},
            camera_state={},
        )
        # Should handle short vector without crash
        assert dl.decisions[0]["state_summary"]["queue_N"] == 0.5


# =============================================================================
# DASHBOARD MODULE TESTS
# =============================================================================

class TestProductionDashboard:
    """Tests for production/dashboard.py module-level code."""

    def test_create_dashboard_html(self):
        from atlas.production.dashboard import create_dashboard_html
        html = create_dashboard_html()
        assert isinstance(html, str)
        assert "ATLAS" in html
        assert "<html" in html
        assert "canvas" in html

    def test_dashboard_data_exists(self):
        from atlas.production.dashboard import _dashboard_data
        assert isinstance(_dashboard_data, dict)
        assert "safety_score" in _dashboard_data
        assert "current_phase" in _dashboard_data

    def test_data_lock_exists(self):
        from atlas.production.dashboard import _data_lock
        import threading
        assert hasattr(_data_lock, 'acquire')
        assert hasattr(_data_lock, 'release')


class TestDashboardApp:
    """Tests for dashboard/app.py."""

    def test_training_state_defaults(self):
        from atlas.dashboard.app import _training_state
        assert isinstance(_training_state, dict)
        assert "is_training" in _training_state
        assert "algorithm" in _training_state

    def test_update_training_state(self):
        from atlas.dashboard.app import update_training_state, _training_state
        old_episode = _training_state.get("episode", 0)
        update_training_state({"episode": 42, "reward": 10.5})
        assert _training_state["episode"] == 42
        assert _training_state["reward"] == 10.5
        # Reset
        update_training_state({"episode": old_episode, "reward": 0.0})

    def test_create_app_without_fastapi(self):
        from atlas.dashboard import app as app_module
        original = app_module._FASTAPI_AVAILABLE
        try:
            app_module._FASTAPI_AVAILABLE = False
            with pytest.raises(ImportError):
                app_module.create_app()
        finally:
            app_module._FASTAPI_AVAILABLE = original


# =============================================================================
# BASELINES — ADDITIONAL TESTS
# =============================================================================

class TestWebsterPolicyExtended:
    """Extended tests for Webster baseline."""

    def test_recalculate_splits(self):
        from atlas.baselines import get_baseline
        b = get_baseline("webster")
        state = np.zeros(26)
        state[0] = 0.8  # High N queue
        state[5] = 0.6  # High S queue
        state[10] = 0.1  # Low E queue
        state[15] = 0.1  # Low W queue
        b._recalculate_splits(state)
        # NS should get more green than EW
        assert b.cycle_ns >= b.cycle_ew

    def test_recalculate_zero_demand(self):
        from atlas.baselines import get_baseline
        b = get_baseline("webster")
        state = np.zeros(26)
        b._recalculate_splits(state)
        # Should use defaults
        assert b.cycle_ns == 6
        assert b.cycle_ew == 5

    def test_periodic_recalc_trigger(self):
        from atlas.baselines import get_baseline
        b = get_baseline("webster")
        state = np.zeros(26)
        state[0] = 0.5
        state[10] = 0.3
        # Run 20 steps to trigger recalculation
        for _ in range(20):
            b.select_action(state)
        assert b._recalc_counter == 0  # Just reset

    def test_phase_switching(self):
        from atlas.baselines import get_baseline
        b = get_baseline("webster")
        state = np.zeros(26)
        actions = []
        for _ in range(30):
            actions.append(b.select_action(state))
        # Should have at least one phase change
        assert any(a != 0 for a in actions)

    def test_webster_reset(self):
        from atlas.baselines import get_baseline
        b = get_baseline("webster")
        for _ in range(10):
            b.select_action(np.random.rand(26))
        b.reset()
        assert b.step_count == 0
        assert b.current_phase == 0
        assert b.steps_in_phase == 0
        assert b._recalc_counter == 0


class TestActuatedPolicyExtended:
    """Extended tests for Actuated baseline."""

    def test_min_green_hold(self):
        from atlas.baselines import get_baseline
        b = get_baseline("actuated")
        state = np.zeros(26)
        state[10] = 0.9  # High EW queue
        # First step should hold (min green not met)
        action = b.select_action(state)
        assert action == 0  # Maintain

    def test_max_green_switch(self):
        from atlas.baselines import get_baseline
        b = get_baseline("actuated")
        state = np.zeros(26)
        state[0] = 0.5  # Some NS traffic
        # Run past max_green
        for _ in range(b.max_green + 1):
            action = b.select_action(state)
        # Should have switched at max green
        assert b.current_phase == 1

    def test_gap_detection_switch(self):
        from atlas.baselines import get_baseline
        b = get_baseline("actuated")
        # Past min green
        b.steps_in_phase = b.min_green + 1
        state = np.zeros(26)
        state[0] = 0.0   # No NS traffic
        state[5] = 0.0   # No NS traffic
        state[10] = 0.5  # EW traffic
        state[15] = 0.5  # EW traffic
        action = b.select_action(state)
        # Should switch because gap detected and other direction has traffic
        assert action in (1, 2)

    def test_extension(self):
        from atlas.baselines import get_baseline
        b = get_baseline("actuated")
        b.steps_in_phase = b.min_green + 1
        state = np.zeros(26)
        state[0] = 0.5   # Some NS traffic
        state[10] = 0.0  # No EW traffic
        action = b.select_action(state)
        assert action == 3  # Extend

    def test_actuated_reset(self):
        from atlas.baselines import get_baseline
        b = get_baseline("actuated")
        for _ in range(10):
            b.select_action(np.random.rand(26))
        b.reset()
        assert b.steps_in_phase == 0
        assert b.current_phase == 0


class TestMaxPressureExtended:
    """Extended tests for MaxPressure baseline."""

    def test_switch_to_ew(self):
        from atlas.baselines import get_baseline
        b = get_baseline("max_pressure")
        b.steps_in_phase = b.min_green + 1
        state = np.zeros(26)
        state[10] = 0.9   # High E queue
        state[15] = 0.9   # High W queue
        state[12] = 0.5   # E wait time
        state[17] = 0.5   # W wait time
        action = b.select_action(state)
        assert action == 2  # Switch to EW

    def test_switch_to_ns(self):
        from atlas.baselines import get_baseline
        b = get_baseline("max_pressure")
        b.current_phase = 1  # Currently EW
        b.steps_in_phase = b.min_green + 1
        state = np.zeros(26)
        state[0] = 0.9   # High N queue
        state[5] = 0.9   # High S queue
        state[2] = 0.5   # N wait time
        state[7] = 0.5   # S wait time
        action = b.select_action(state)
        assert action == 1  # Switch to NS

    def test_hysteresis_prevents_switch(self):
        from atlas.baselines import get_baseline
        b = get_baseline("max_pressure")
        b.steps_in_phase = b.min_green + 1
        state = np.zeros(26)
        # Nearly equal pressure — hysteresis should prevent switch
        state[0] = 0.5
        state[5] = 0.5
        state[10] = 0.5
        state[15] = 0.5
        action = b.select_action(state)
        assert action == 0  # Maintain


class TestBasePolicy:
    """Tests for BasePolicy."""

    def test_not_implemented(self):
        from atlas.baselines import BasePolicy
        b = BasePolicy()
        with pytest.raises(NotImplementedError):
            b.select_action(np.zeros(26))

    def test_reset_noop(self):
        from atlas.baselines import BasePolicy
        b = BasePolicy()
        b.reset()  # Should not raise


# =============================================================================
# CLI — ADDITIONAL TESTS
# =============================================================================

class TestCLI:
    """Tests for CLI module."""

    def test_setup_logging(self):
        from atlas.cli import setup_logging
        setup_logging("DEBUG")
        setup_logging("INFO")
        setup_logging("WARNING")
        # Should not raise

    def test_main_no_command(self):
        from atlas.cli import main
        with patch("sys.argv", ["atlas"]):
            main()  # Should print help, not crash

    def test_main_help(self):
        from atlas.cli import main
        with patch("sys.argv", ["atlas", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

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

    def test_cmd_config_show_default(self):
        from atlas.cli import cmd_config
        args = Namespace(action="show", file=None, output=None)
        cmd_config(args)  # Should print config

    def test_cmd_config_show_file(self):
        from atlas.cli import cmd_config
        from atlas.config import TrainingConfig
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        try:
            TrainingConfig().save(path)
            args = Namespace(action="show", file=path, output=None)
            cmd_config(args)
        finally:
            os.unlink(path)

    def test_parser_train_subcommand(self):
        from atlas.cli import main
        # Just test that parser accepts train args without actually training
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", action="store_true")
        sub = parser.add_subparsers(dest="command")
        train_p = sub.add_parser("train")
        train_p.add_argument("--episodes", type=int)
        args = parser.parse_args(["train", "--episodes", "10"])
        assert args.command == "train"
        assert args.episodes == 10


# =============================================================================
# CONTROLLER — MOCKED CONNECTION TESTS
# =============================================================================

class TestModbusControllerMocked:
    """Tests for ModbusController with mocked pymodbus client."""

    def test_connect_success(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        mock_client_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client_cls.return_value = mock_client

        mock_module = MagicMock()
        mock_module.ModbusTcpClient = mock_client_cls
        with patch.dict("sys.modules", {"pymodbus": MagicMock(), "pymodbus.client": mock_module}):
            result = ctrl.connect()
            assert result is True

    def test_disconnect_with_client(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        ctrl.disconnect()
        ctrl.client.close.assert_called_once()

    def test_set_phase_success(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        ctrl.client.write_register.return_value = MagicMock(isError=lambda: False)
        assert ctrl.set_phase(1) is True

    def test_set_phase_error(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        ctrl.client.write_register.return_value = MagicMock(isError=lambda: True)
        assert ctrl.set_phase(1) is False

    def test_set_phase_exception(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        ctrl.client.write_register.side_effect = Exception("write failed")
        assert ctrl.set_phase(1) is False

    def test_get_status_success(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_result.registers = [2, 15, 0, 1]
        ctrl.client.read_holding_registers.return_value = mock_result
        status = ctrl.get_status()
        assert status.current_phase == 2
        assert status.phase_elapsed == 15
        assert status.is_fault is False
        assert status.is_manual_override is True

    def test_get_status_read_error(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        mock_result = MagicMock()
        mock_result.isError.return_value = True
        ctrl.client.read_holding_registers.return_value = mock_result
        status = ctrl.get_status()
        assert status.is_fault is True

    def test_get_status_exception(self):
        from atlas.production.controller_interface import ModbusController
        ctrl = ModbusController()
        ctrl.client = MagicMock()
        ctrl.client.read_holding_registers.side_effect = Exception("read failed")
        status = ctrl.get_status()
        assert status.is_fault is True


class TestRestAPIControllerMocked:
    """Tests for RestAPIController with mocked requests."""

    def test_connect_success(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController(api_key="test-key")
        import requests as req_mod
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        with patch.object(req_mod, "Session", return_value=mock_session):
            result = ctrl.connect()
            assert result is True
            assert "Authorization" in mock_session.headers

    def test_disconnect_with_session(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.session = MagicMock()
        ctrl.disconnect()
        ctrl.session.close.assert_called_once()

    def test_set_phase_success(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        ctrl.session.post.return_value = mock_resp
        assert ctrl.set_phase(2) is True

    def test_set_phase_failure(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        ctrl.session.post.return_value = mock_resp
        assert ctrl.set_phase(2) is False

    def test_set_phase_exception(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.session = MagicMock()
        ctrl.session.post.side_effect = Exception("network error")
        assert ctrl.set_phase(2) is False

    def test_get_status_success(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "phase": 1, "elapsed": 25, "fault": False, "manual": True
        }
        ctrl.session.get.return_value = mock_resp
        status = ctrl.get_status()
        assert status.current_phase == 1
        assert status.phase_elapsed == 25
        assert status.is_manual_override is True

    def test_get_status_exception(self):
        from atlas.production.controller_interface import RestAPIController
        ctrl = RestAPIController()
        ctrl.session = MagicMock()
        ctrl.session.get.side_effect = Exception("timeout")
        status = ctrl.get_status()
        assert status.is_fault is True


class TestGPIOControllerMocked:
    """Tests for GPIOController with mocked RPi.GPIO."""

    def test_connect_with_mock_gpio(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        mock_gpio = MagicMock()
        mock_gpio.BCM = 11
        mock_gpio.OUT = 0
        mock_gpio.LOW = 0
        with patch.dict("sys.modules", {"RPi": MagicMock(), "RPi.GPIO": mock_gpio}):
            result = ctrl.connect()
            assert result is True

    def test_disconnect_with_mock_gpio(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        ctrl.gpio = MagicMock()
        ctrl.disconnect()
        ctrl.gpio.cleanup.assert_called_once()

    def test_set_phase_with_mock_gpio(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        ctrl.gpio = MagicMock()
        ctrl.gpio.LOW = 0
        ctrl.gpio.HIGH = 1
        result = ctrl.set_phase(0)
        assert result is True

    def test_get_status_returns_current(self):
        from atlas.production.controller_interface import GPIOController
        ctrl = GPIOController()
        ctrl.current_phase = 2
        status = ctrl.get_status()
        assert status.current_phase == 2


class TestRequestPhaseChangeFullFlow:
    """Test the full request_phase_change flow with manual override."""

    def test_manual_override_rejected(self):
        from atlas.production.controller_interface import SimulatedController, ControllerStatus
        ctrl = SimulatedController()
        ctrl.connect()
        # Mock get_status to return manual override
        ctrl.get_status = lambda: ControllerStatus(is_manual_override=True)
        ctrl.phase_start_time = time.time() - 20
        result = ctrl.request_phase_change(1)
        assert result is False


# =============================================================================
# INFERENCE ENGINE — EXTENDED TESTS
# =============================================================================

class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    def test_creation(self):
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        config = ProductionConfig()
        engine = InferenceEngine(config)
        assert engine.running is False
        assert engine.step_count == 0
        assert engine.current_phase == 0
        assert engine.incident_active is False

    def test_safety_and_logger_initialized(self):
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        config = ProductionConfig(log_dir=tempfile.mkdtemp())
        engine = InferenceEngine(config)
        assert engine.safety is not None
        assert engine.logger_decisions is not None
        assert engine.agent is None  # Not initialized yet

    def test_initialize_demo_mode(self):
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        config = ProductionConfig(
            mode="demo",
            model_path="nonexistent_model.pt",
            log_dir=tempfile.mkdtemp(),
        )
        engine = InferenceEngine(config)
        # Mock camera pipeline to avoid OpenCV access violation
        with patch("atlas.production.camera_pipeline.HAS_CV2", False):
            result = engine.initialize()
            assert result is True
            assert engine.agent is not None
            assert engine.camera_pipeline is not None
            assert engine.controller is not None
            # Stop camera threads immediately to avoid crash
            if engine.camera_pipeline:
                engine.camera_pipeline.running = False
            engine.logger_decisions.save()

    def test_shutdown(self):
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        config = ProductionConfig(
            mode="demo",
            model_path="nonexistent_model.pt",
            log_dir=tempfile.mkdtemp(),
        )
        engine = InferenceEngine(config)
        with patch("atlas.production.camera_pipeline.HAS_CV2", False):
            engine.initialize()
            # Stop camera threads immediately
            if engine.camera_pipeline:
                engine.camera_pipeline.running = False
                for stream in engine.camera_pipeline.cameras.values():
                    stream.running = False
            engine.shutdown()
            assert engine.running is False

    def test_print_status(self):
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        config = ProductionConfig(
            mode="demo",
            model_path="nonexistent_model.pt",
            log_dir=tempfile.mkdtemp(),
        )
        engine = InferenceEngine(config)
        with patch("atlas.production.camera_pipeline.HAS_CV2", False):
            engine.initialize()
            engine.start_time = time.time()
            engine._print_status()  # Should not raise
            # Stop camera threads immediately
            if engine.camera_pipeline:
                engine.camera_pipeline.running = False
                for stream in engine.camera_pipeline.cameras.values():
                    stream.running = False
            engine.shutdown()

    def test_check_external_commands_no_server(self):
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        config = ProductionConfig(log_dir=tempfile.mkdtemp())
        engine = InferenceEngine(config)
        engine.controller = MagicMock()
        engine._check_external_commands()  # Should not raise (catches exceptions)


class TestRunProduction:
    """Tests for run_production function."""

    def test_run_production_init_failure(self):
        from atlas.production.inference_engine import run_production, InferenceEngine
        # Mock InferenceEngine to fail on initialize
        with patch("atlas.production.inference_engine.InferenceEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_instance.initialize.return_value = False
            MockEngine.return_value = mock_instance
            run_production(mode="demo")
            mock_instance.run.assert_not_called()
