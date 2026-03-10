"""
ATLAS Pro — Traffic Controller Interface
=============================================
Adapters for communicating with real traffic light controllers.

Supports:
    - NTCIP (National Transportation Communications for ITS Protocol)
    - Modbus TCP/RTU (industrial controllers)
    - REST API (modern smart controllers)
    - GPIO (direct Raspberry Pi control for prototyping)
    - Simulated (for testing)
"""

import os
import time
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("ATLAS.Controller")


# ============================================================
# Phase Definitions
# ============================================================

@dataclass
class PhaseDefinition:
    """Defines a traffic signal phase."""
    id: int
    name: str
    green_directions: List[str]   # e.g. ["N", "S"]
    min_green: float = 10.0       # seconds
    max_green: float = 90.0       # seconds
    yellow: float = 3.0           # seconds
    all_red: float = 2.0          # seconds


DEFAULT_PHASES = [
    PhaseDefinition(0, "NS_Green", ["N", "S"], min_green=10, max_green=90),
    PhaseDefinition(1, "NS_Left",  ["N", "S"], min_green=8,  max_green=30),
    PhaseDefinition(2, "EW_Green", ["E", "W"], min_green=10, max_green=90),
    PhaseDefinition(3, "EW_Left",  ["E", "W"], min_green=8,  max_green=30),
]


@dataclass
class ControllerStatus:
    """Current status from the traffic controller."""
    current_phase: int = 0
    phase_elapsed: float = 0.0
    is_yellow: bool = False
    is_all_red: bool = False
    is_fault: bool = False
    is_manual_override: bool = False
    controller_time: float = 0.0
    last_command_accepted: bool = True


# ============================================================
# Base Controller Interface
# ============================================================

class TrafficController(ABC):
    """Abstract base class for traffic controller adapters."""

    def __init__(self, phases: List[PhaseDefinition] = None):
        self.phases = phases or DEFAULT_PHASES
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.command_log: List[dict] = []
        self._lock = threading.Lock()

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the controller."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection."""
        pass

    @abstractmethod
    def set_phase(self, phase_id: int) -> bool:
        """Command the controller to switch to a specific phase."""
        pass

    @abstractmethod
    def get_status(self) -> ControllerStatus:
        """Read current status from the controller."""
        pass

    def request_phase_change(self, new_phase: int) -> bool:
        """
        Safely request a phase change with all safety checks.
        This is the primary method the AI agent should call.
        """
        with self._lock:
            status = self.get_status()

            # Safety checks
            if status.is_fault:
                logger.error("Controller in FAULT state. Phase change rejected.")
                return False

            if status.is_manual_override:
                logger.warning("Manual override active. Phase change rejected.")
                return False

            phase_def = self.phases[self.current_phase]
            elapsed = time.time() - self.phase_start_time

            # Check minimum green time
            if elapsed < phase_def.min_green:
                logger.debug(
                    f"Min green not met: {elapsed:.1f}s < {phase_def.min_green}s"
                )
                return False

            # Execute phase change
            if new_phase != self.current_phase:
                success = self.set_phase(new_phase)
                if success:
                    self._log_command(self.current_phase, new_phase, elapsed)
                    self.current_phase = new_phase
                    self.phase_start_time = time.time()
                    logger.info(
                        f"Phase changed: {phase_def.name} -> "
                        f"{self.phases[new_phase].name} (after {elapsed:.1f}s)"
                    )
                return success

            return True

    def _log_command(self, from_phase: int, to_phase: int, elapsed: float):
        """Log every command for auditing."""
        entry = {
            "timestamp": time.time(),
            "from_phase": from_phase,
            "to_phase": to_phase,
            "elapsed_seconds": elapsed,
            "from_name": self.phases[from_phase].name,
            "to_name": self.phases[to_phase].name,
        }
        self.command_log.append(entry)

        # Keep log manageable
        if len(self.command_log) > 10000:
            self.command_log = self.command_log[-5000:]

    def save_command_log(self, path: str):
        """Save command log to JSON file."""
        with open(path, "w") as f:
            json.dump(self.command_log, f, indent=2)


# ============================================================
# Simulated Controller (for testing)
# ============================================================

class SimulatedController(TrafficController):
    """Simulated controller for testing without hardware."""

    def __init__(self, phases=None, latency: float = 0.05):
        super().__init__(phases)
        self.connected = False
        self.latency = latency
        self._fault = False

    def connect(self) -> bool:
        self.connected = True
        logger.info("Simulated controller connected")
        return True

    def disconnect(self):
        self.connected = False
        logger.info("Simulated controller disconnected")

    def set_phase(self, phase_id: int) -> bool:
        time.sleep(self.latency)  # Simulate hardware latency
        if 0 <= phase_id < len(self.phases):
            return True
        return False

    def get_status(self) -> ControllerStatus:
        return ControllerStatus(
            current_phase=self.current_phase,
            phase_elapsed=time.time() - self.phase_start_time,
            is_fault=self._fault,
        )


# ============================================================
# Modbus Controller (Industrial standard)
# ============================================================

class ModbusController(TrafficController):
    """
    Modbus TCP/RTU controller adapter.
    Common in European traffic systems (Spain, Germany).

    Register map (configurable):
        - Register 0: Current phase (read/write)
        - Register 1: Phase timer (read)
        - Register 2: Fault status (read)
        - Register 3: Manual override (read)
    """

    def __init__(self, host: str = "192.168.1.100", port: int = 502,
                 unit_id: int = 1, phases=None):
        super().__init__(phases)
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.client = None

    def connect(self) -> bool:
        try:
            from pymodbus.client import ModbusTcpClient
            self.client = ModbusTcpClient(self.host, port=self.port)
            connected = self.client.connect()
            if connected:
                logger.info(f"Modbus connected to {self.host}:{self.port}")
            return connected
        except ImportError:
            logger.error("pymodbus not installed. Run: pip install pymodbus")
            return False
        except Exception as e:
            logger.error(f"Modbus connection failed: {e}")
            return False

    def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Modbus disconnected")

    def set_phase(self, phase_id: int) -> bool:
        if not self.client:
            return False
        try:
            result = self.client.write_register(0, phase_id, unit=self.unit_id)
            return not result.isError()
        except Exception as e:
            logger.error(f"Modbus write error: {e}")
            return False

    def get_status(self) -> ControllerStatus:
        if not self.client:
            return ControllerStatus(is_fault=True)
        try:
            result = self.client.read_holding_registers(0, 4, unit=self.unit_id)
            if result.isError():
                return ControllerStatus(is_fault=True)
            return ControllerStatus(
                current_phase=result.registers[0],
                phase_elapsed=result.registers[1],
                is_fault=bool(result.registers[2]),
                is_manual_override=bool(result.registers[3]),
            )
        except Exception as e:
            logger.error(f"Modbus read error: {e}")
            return ControllerStatus(is_fault=True)


# ============================================================
# REST API Controller (Modern smart controllers)
# ============================================================

class RestAPIController(TrafficController):
    """
    REST API adapter for modern smart traffic controllers.
    Works with any controller that exposes an HTTP API.
    """

    def __init__(self, base_url: str = "http://localhost:8080/api",
                 api_key: str = "", phases=None):
        super().__init__(phases)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = None

    def connect(self) -> bool:
        try:
            import requests
            self.session = requests.Session()
            if self.api_key:
                self.session.headers["Authorization"] = f"Bearer {self.api_key}"
            resp = self.session.get(f"{self.base_url}/status", timeout=5)
            connected = resp.status_code == 200
            if connected:
                logger.info(f"REST API connected to {self.base_url}")
            return connected
        except Exception as e:
            logger.error(f"REST API connection failed: {e}")
            return False

    def disconnect(self):
        if self.session:
            self.session.close()

    def set_phase(self, phase_id: int) -> bool:
        try:
            resp = self.session.post(
                f"{self.base_url}/phase",
                json={"phase": phase_id},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"REST API phase change failed: {e}")
            return False

    def get_status(self) -> ControllerStatus:
        try:
            resp = self.session.get(f"{self.base_url}/status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return ControllerStatus(
                    current_phase=data.get("phase", 0),
                    phase_elapsed=data.get("elapsed", 0),
                    is_fault=data.get("fault", False),
                    is_manual_override=data.get("manual", False),
                )
        except Exception:
            pass
        return ControllerStatus(is_fault=True)


# ============================================================
# GPIO Controller (Raspberry Pi prototyping)
# ============================================================

class GPIOController(TrafficController):
    """
    Direct GPIO control for Raspberry Pi prototyping.
    Controls LED traffic light modules directly.

    Pin mapping (configurable):
        Phase 0 (NS Green):  GPIO 17, 18 (green), 22, 23 (red)
        Phase 1 (EW Green):  GPIO 17, 18 (red),   22, 23 (green)
    """

    def __init__(self, pin_map: Optional[Dict] = None, phases=None):
        super().__init__(phases)
        self.pin_map = pin_map or {
            "N_green": 17, "N_red": 27, "N_yellow": 22,
            "S_green": 5,  "S_red": 6,  "S_yellow": 13,
            "E_green": 19, "E_red": 26, "E_yellow": 21,
            "W_green": 20, "W_red": 16, "W_yellow": 12,
        }
        self.gpio = None

    def connect(self) -> bool:
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in self.pin_map.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            logger.info("GPIO controller connected")
            return True
        except ImportError:
            logger.error("RPi.GPIO not available. Not on Raspberry Pi?")
            return False

    def disconnect(self):
        if self.gpio:
            self.gpio.cleanup()

    def set_phase(self, phase_id: int) -> bool:
        if not self.gpio:
            return False

        # Turn all off first
        for pin in self.pin_map.values():
            self.gpio.output(pin, self.gpio.LOW)

        # Set appropriate green/red based on phase
        phase = self.phases[phase_id]
        all_dirs = ["N", "S", "E", "W"]

        for d in all_dirs:
            if d in phase.green_directions:
                self.gpio.output(self.pin_map[f"{d}_green"], self.gpio.HIGH)
            else:
                self.gpio.output(self.pin_map[f"{d}_red"], self.gpio.HIGH)

        return True

    def get_status(self) -> ControllerStatus:
        return ControllerStatus(
            current_phase=self.current_phase,
            phase_elapsed=time.time() - self.phase_start_time,
        )


# ============================================================
# Factory
# ============================================================

CONTROLLERS = {
    "simulated": SimulatedController,
    "modbus": ModbusController,
    "rest_api": RestAPIController,
    "gpio": GPIOController,
}


def create_controller(controller_type: str = "simulated", **kwargs) -> TrafficController:
    """Create a controller instance by type."""
    if controller_type not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {controller_type}. Options: {list(CONTROLLERS)}")
    return CONTROLLERS[controller_type](**kwargs)
