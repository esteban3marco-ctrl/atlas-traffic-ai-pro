"""
ATLAS Pro — Real-Time Camera Pipeline
=========================================
Captures video from RTSP/USB cameras, runs YOLOv8 vehicle detection,
and converts detections into the 26-dim state vector for the AI agent.

Supports:
    - RTSP streams (IP cameras)
    - USB cameras (for testing)
    - Video files (for demo/replay)
    - Simulated feeds (for development)
"""

import os
import time
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("ATLAS.Camera")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available. Camera pipeline will use simulated feeds.")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("Ultralytics not available. Using simulated detection.")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class VehicleDetection:
    """Single detected vehicle."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str  # car, truck, bus, motorcycle, bicycle
    speed_estimate: float = 0.0
    lane: int = 0
    direction: str = ""  # N, S, E, W


@dataclass
class IntersectionState:
    """Full state of an intersection from camera analysis."""
    timestamp: float = 0.0
    queue_lengths: Dict[str, float] = field(default_factory=dict)
    vehicle_speeds: Dict[str, float] = field(default_factory=dict)
    wait_times: Dict[str, float] = field(default_factory=dict)
    densities: Dict[str, float] = field(default_factory=dict)
    halted_counts: Dict[str, int] = field(default_factory=dict)
    total_vehicles: int = 0
    emergency_detected: bool = False
    emergency_direction: str = ""
    detections: List[VehicleDetection] = field(default_factory=list)

    def to_state_vector(self, directions=("N", "S", "E", "W"),
                        current_phase=0, phase_duration=0,
                        num_phases=4) -> np.ndarray:
        """Convert to 26-dim state vector matching training format."""
        state = []
        for d in directions:
            state.append(self.queue_lengths.get(d, 0) / 50.0)    # normalize
            state.append(self.vehicle_speeds.get(d, 0) / 60.0)
            state.append(self.wait_times.get(d, 0) / 120.0)
            state.append(self.densities.get(d, 0))
            state.append(self.halted_counts.get(d, 0) / 20.0)

        # Phase one-hot (4 dims)
        phase_one_hot = [0.0] * num_phases
        if 0 <= current_phase < num_phases:
            phase_one_hot[current_phase] = 1.0
        state.extend(phase_one_hot)

        # Timing (2 dims)
        state.append(phase_duration / 90.0)
        state.append(1.0 if self.emergency_detected else 0.0)

        return np.array(state, dtype=np.float32)


# ============================================================
# Camera Pipeline
# ============================================================

@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    source: str = "0"                  # RTSP URL, device ID, or file path
    direction: str = "N"               # Which approach this camera covers
    fps: int = 10                      # Target FPS
    resolution: Tuple[int, int] = (1280, 720)
    roi: Optional[Tuple[int, int, int, int]] = None  # Region of Interest
    detection_confidence: float = 0.4
    model_path: str = "yolov8n.pt"


@dataclass
class PipelineConfig:
    """Configuration for the full camera pipeline."""
    cameras: List[CameraConfig] = field(default_factory=list)
    inference_interval: float = 0.5    # seconds between full inference
    state_buffer_size: int = 10        # smoothing buffer
    halted_speed_threshold: float = 2.0  # km/h
    queue_detection_method: str = "density"  # density or counting
    enable_tracking: bool = True
    save_frames: bool = False
    save_dir: str = "production_frames"


class CameraStream:
    """Thread-safe video capture from a single camera."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.fps_actual = 0.0
        self._frame_count = 0
        self._start_time = 0

    def start(self):
        """Start capturing in background thread."""
        if not HAS_CV2:
            logger.warning(f"OpenCV unavailable. Camera {self.config.direction} in sim mode.")
            self.running = True
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.config.direction} started: {self.config.source}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    def _capture_loop(self):
        """Continuous frame capture."""
        source = self.config.source
        if source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open camera: {self.config.source}")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])

        self._start_time = time.time()
        target_delay = 1.0 / self.config.fps

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Try reconnecting for RTSP
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(source if isinstance(source, int) else self.config.source)
                continue

            if self.config.roi:
                x1, y1, x2, y2 = self.config.roi
                frame = frame[y1:y2, x1:x2]

            with self.lock:
                self.frame = frame
                self._frame_count += 1

            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self.fps_actual = self._frame_count / elapsed

            time.sleep(target_delay)

        cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (thread-safe)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class VehicleDetector:
    """YOLOv8-based vehicle detector with tracking."""

    VEHICLE_CLASSES = {
        "car": 2, "truck": 7, "bus": 5,
        "motorcycle": 3, "bicycle": 1,
    }
    # Reverse mapping from COCO class IDs
    COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        self.confidence = confidence
        self.model = None

        if HAS_YOLO:
            try:
                self.model = YOLO(model_path)
                logger.info(f"YOLO model loaded: {model_path}")
            except Exception as e:
                logger.warning(f"YOLO load failed: {e}. Using simulated detection.")

    def detect(self, frame: np.ndarray, direction: str = "N") -> List[VehicleDetection]:
        """Detect vehicles in a frame."""
        if self.model is None:
            return self._simulated_detection(direction)

        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.COCO_VEHICLES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det = VehicleDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        class_name=self.COCO_VEHICLES[cls_id],
                        direction=direction,
                    )
                    detections.append(det)

        return detections

    def _simulated_detection(self, direction: str) -> List[VehicleDetection]:
        """Generate realistic simulated detections for testing."""
        n_vehicles = np.random.poisson(5)
        detections = []
        for _ in range(n_vehicles):
            det = VehicleDetection(
                bbox=(np.random.randint(0, 1000), np.random.randint(0, 700),
                      np.random.randint(50, 200), np.random.randint(40, 150)),
                confidence=np.random.uniform(0.5, 0.99),
                class_name=np.random.choice(["car", "car", "car", "truck", "bus"]),
                speed_estimate=np.random.uniform(0, 50),
                direction=direction,
            )
            detections.append(det)
        return detections


class CameraPipeline:
    """
    Full camera pipeline: captures frames, detects vehicles,
    and produces intersection state vectors for the AI agent.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cameras: Dict[str, CameraStream] = {}
        self.detector = VehicleDetector()
        self.state_buffer = deque(maxlen=config.state_buffer_size)
        self.latest_state: Optional[IntersectionState] = None
        self.running = False
        self._lock = threading.Lock()

        # Direction-specific trackers
        self._direction_history: Dict[str, deque] = {}
        for cam_cfg in config.cameras:
            self._direction_history[cam_cfg.direction] = deque(maxlen=30)

    def start(self):
        """Start all cameras and the processing loop."""
        for cam_cfg in self.config.cameras:
            stream = CameraStream(cam_cfg)
            stream.start()
            self.cameras[cam_cfg.direction] = stream

        self.running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        logger.info(f"Camera pipeline started with {len(self.cameras)} cameras")

    def stop(self):
        """Stop all cameras and processing."""
        self.running = False
        for stream in self.cameras.values():
            stream.stop()
        logger.info("Camera pipeline stopped")

    def get_state(self) -> Optional[IntersectionState]:
        """Get the latest intersection state (thread-safe)."""
        with self._lock:
            return self.latest_state

    def get_state_vector(self, current_phase: int = 0,
                         phase_duration: float = 0) -> Optional[np.ndarray]:
        """Get the latest state as a 26-dim vector for the AI agent."""
        state = self.get_state()
        if state is None:
            return None
        return state.to_state_vector(
            current_phase=current_phase,
            phase_duration=phase_duration,
        )

    def _process_loop(self):
        """Main processing loop."""
        while self.running:
            t0 = time.time()

            state = IntersectionState(timestamp=time.time())
            all_detections = []

            for direction, stream in self.cameras.items():
                frame = stream.get_frame()
                if frame is not None:
                    dets = self.detector.detect(frame, direction)
                else:
                    # Simulated if no frame available
                    dets = self.detector._simulated_detection(direction)

                all_detections.extend(dets)

                # Compute per-direction metrics
                n_vehicles = len(dets)
                n_halted = sum(1 for d in dets
                               if d.speed_estimate < self.config.halted_speed_threshold)
                avg_speed = (np.mean([d.speed_estimate for d in dets])
                             if dets else 0.0)

                state.queue_lengths[direction] = float(n_halted)
                state.vehicle_speeds[direction] = float(avg_speed)
                state.densities[direction] = min(n_vehicles / 20.0, 1.0)
                state.halted_counts[direction] = n_halted

                # Estimate wait time from history
                hist = self._direction_history[direction]
                hist.append(n_halted)
                if len(hist) > 1:
                    # Approximate wait = cumulative halted vehicles over time
                    state.wait_times[direction] = float(
                        sum(hist) * self.config.inference_interval
                    )
                else:
                    state.wait_times[direction] = 0.0

                # Emergency detection (look for ambulance/fire truck patterns)
                for d in dets:
                    if d.class_name in ("truck", "bus") and d.confidence > 0.7:
                        # In production, this would use siren audio or
                        # special vehicle classification
                        pass

            state.total_vehicles = len(all_detections)
            state.detections = all_detections

            with self._lock:
                self.latest_state = state
                self.state_buffer.append(state)

            # Sleep to maintain target inference rate
            elapsed = time.time() - t0
            sleep_time = max(0, self.config.inference_interval - elapsed)
            time.sleep(sleep_time)


def create_default_pipeline(
    camera_sources: Optional[Dict[str, str]] = None
) -> CameraPipeline:
    """Create a pipeline with default or provided camera sources."""
    if camera_sources is None:
        camera_sources = {
            "N": "0",  # Default webcam or simulated
            "S": "0",
            "E": "0",
            "W": "0",
        }

    cameras = [
        CameraConfig(source=src, direction=d)
        for d, src in camera_sources.items()
    ]

    config = PipelineConfig(cameras=cameras)
    return CameraPipeline(config)
