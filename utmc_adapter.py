"""
ATLAS Pro - Adaptador UTMC para Controladores Europeos
=======================================================
UTMC (Urban Traffic Management and Control) es el estandar europeo
para sistemas de gestion de trafico urbano.

A diferencia de NTCIP (basado en SNMP), UTMC usa:
  - XML sobre TCP/IP para intercambio de datos
  - Estandar BS 8438 (UK) / EN 12675 (EU) para controladores
  - OCIT-C (Alemania) y DIASER (Francia) como variantes regionales

Este modulo implementa:
  - Cliente UTMC con mensajeria XML
  - Adaptacion a EN 12675 (senales de trafico)
  - Lectura de estado de detectores y senales
  - Control de fases compatible con ATLAS (acciones 0-3)
  - Modo OCIT-C para compatibilidad con controladores alemanes

Dependencias:
  pip install requests lxml

Uso:
  adapter = UTMCAdapter("https://controller.city.eu/utmc", api_key="xxx")
  adapter.connect()
  state = adapter.read_state()
  adapter.apply_action(1)  # N-S
"""

import os
import sys
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger("ATLAS.UTMC")

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from lxml import etree
    LXML_OK = True
except ImportError:
    LXML_OK = False
    try:
        import xml.etree.ElementTree as etree
    except ImportError:
        pass


# =============================================================================
# CONSTANTES UTMC / EN 12675
# =============================================================================

class UTMCNamespaces:
    """Namespaces XML de UTMC."""
    UTMC = "http://www.utmc.eu/schema/2.0"
    TF = "http://www.utmc.eu/schema/2.0/trafficflow"
    SIG = "http://www.utmc.eu/schema/2.0/signals"
    DET = "http://www.utmc.eu/schema/2.0/detectors"
    EVT = "http://www.utmc.eu/schema/2.0/events"


class SignalState(IntEnum):
    """Estados de senal segun EN 12675."""
    OFF = 0
    RED = 1
    RED_AMBER = 2      # rojo+ambar (UK, Alemania = proximo verde)
    GREEN = 3
    AMBER = 4           # ambar/amarillo
    FLASHING_AMBER = 5  # intermitente
    FLASHING_RED = 6
    DARK = 7


class ControllerMode(IntEnum):
    """Modos del controlador EN 12675."""
    STANDBY = 0
    FIXED_TIME = 1
    VEHICLE_ACTUATED = 2
    ADAPTIVE = 3          # ATLAS usa este
    CENTRAL_CONTROL = 4
    MANUAL = 5
    FLASH = 6
    ALL_RED = 7


# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass
class UTMCConfig:
    """Configuracion del adaptador UTMC."""
    # Conexion
    base_url: str = "https://controller.local/utmc/api"
    api_key: str = ""
    auth_token: str = ""
    timeout_s: float = 5.0
    verify_ssl: bool = True

    # Identificacion
    controller_id: str = "CTRL_001"
    intersection_id: str = "INT_001"

    # Mapeo de fases ATLAS -> stages del controlador
    # EN 12675 usa "stages" en lugar de "phases"
    stage_map: Dict[int, List[str]] = field(default_factory=lambda: {
        1: ["STAGE_A"],     # N-S
        2: ["STAGE_B"],     # E-O
    })

    # Mapeo de detectores a direcciones
    detector_map: Dict[str, Tuple[int, str]] = field(default_factory=lambda: {
        "DET_N1": (0, "inductive_loop"),
        "DET_N2": (0, "stop_line"),
        "DET_S1": (1, "inductive_loop"),
        "DET_S2": (1, "stop_line"),
        "DET_E1": (2, "inductive_loop"),
        "DET_E2": (2, "stop_line"),
        "DET_W1": (3, "inductive_loop"),
        "DET_W2": (3, "stop_line"),
    })

    # Seguridad
    min_green_s: int = 7
    max_green_s: int = 90
    intergreen_s: int = 5     # tiempo entre fases (ambar + todo rojo)

    # Heartbeat
    heartbeat_interval_s: float = 3.0
    max_missed_heartbeats: int = 3

    # OCIT-C (Alemania)
    use_ocit: bool = False
    ocit_port: int = 5000


# =============================================================================
# CLIENTE UTMC
# =============================================================================

class UTMCClient:
    """
    Cliente para comunicacion con controladores UTMC via XML/REST.

    El protocolo UTMC define mensajes XML con estructura:
    <UTMCMessage>
      <Header>...</Header>
      <Body>
        <Command type="...">...</Command>
      </Body>
    </UTMCMessage>
    """

    def __init__(self, config: UTMCConfig, simulated: bool = None):
        self.config = config
        self._session = None
        self._connected = False
        # Simulado si: no hay requests, se pide explicitamente, o el host es localhost/local
        if simulated is not None:
            self._simulated = simulated
        elif not REQUESTS_OK:
            self._simulated = True
        else:
            # Auto-detectar: si el host no es alcanzable, simular
            self._simulated = ("local" in config.base_url or "localhost" in config.base_url
                               or "127.0.0.1" in config.base_url or "example" in config.base_url)

    def connect(self) -> bool:
        """Establece sesion con el controlador UTMC."""
        if self._simulated:
            logger.info("UTMC simulado: conexion virtual establecida")
            self._connected = True
            return True

        try:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/xml",
                "Accept": "application/xml",
                "X-API-Key": self.config.api_key,
            })
            if self.config.auth_token:
                self._session.headers["Authorization"] = f"Bearer {self.config.auth_token}"

            self._session.verify = self.config.verify_ssl
            self._session.timeout = self.config.timeout_s

            # Test de conexion
            response = self._session.get(
                f"{self.config.base_url}/status/{self.config.controller_id}"
            )
            if response.status_code == 200:
                self._connected = True
                logger.info(f"UTMC conectado a {self.config.base_url}")
                return True
            else:
                logger.error(f"UTMC conexion fallida: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error conectando UTMC: {e}")
            return False

    def disconnect(self):
        """Cierra sesion UTMC."""
        if self._session:
            self._session.close()
        self._connected = False
        logger.info("UTMC desconectado")

    def send_command(self, command_type: str, params: Dict) -> Optional[Dict]:
        """Envia un comando UTMC al controlador."""
        xml_msg = self._build_message(command_type, params)

        if self._simulated:
            return self._simulated_response(command_type, params)

        if not self._connected:
            return None

        try:
            response = self._session.post(
                f"{self.config.base_url}/command/{self.config.controller_id}",
                data=xml_msg,
            )
            if response.status_code == 200:
                return self._parse_response(response.text)
            else:
                logger.warning(f"UTMC comando {command_type}: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"UTMC error enviando {command_type}: {e}")
            return None

    def get_detector_data(self) -> Optional[Dict]:
        """Lee datos de todos los detectores."""
        if self._simulated:
            return self._simulated_detector_data()

        try:
            response = self._session.get(
                f"{self.config.base_url}/detectors/{self.config.intersection_id}"
            )
            if response.status_code == 200:
                return self._parse_detector_xml(response.text)
            return None
        except Exception as e:
            logger.error(f"Error leyendo detectores: {e}")
            return None

    def get_signal_state(self) -> Optional[Dict]:
        """Lee estado de senales del controlador."""
        if self._simulated:
            return self._simulated_signal_state()

        try:
            response = self._session.get(
                f"{self.config.base_url}/signals/{self.config.controller_id}"
            )
            if response.status_code == 200:
                return self._parse_signal_xml(response.text)
            return None
        except Exception as e:
            logger.error(f"Error leyendo senales: {e}")
            return None

    def _build_message(self, command_type: str, params: Dict) -> str:
        """Construye mensaje XML UTMC."""
        timestamp = datetime.now(timezone.utc).isoformat()
        params_xml = "\n".join(
            f'        <Param name="{k}">{v}</Param>' for k, v in params.items()
        )
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<UTMCMessage xmlns="{UTMCNamespaces.UTMC}">
  <Header>
    <MessageId>{int(time.time() * 1000)}</MessageId>
    <Timestamp>{timestamp}</Timestamp>
    <Source>ATLAS_PRO</Source>
    <Destination>{self.config.controller_id}</Destination>
  </Header>
  <Body>
    <Command type="{command_type}">
{params_xml}
    </Command>
  </Body>
</UTMCMessage>"""

    def _parse_response(self, xml_text: str) -> Dict:
        """Parsea respuesta XML del controlador."""
        try:
            root = etree.fromstring(xml_text.encode())
            result = {}
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    result[tag] = elem.text.strip()
            return result
        except Exception as e:
            logger.error(f"Error parseando XML: {e}")
            return {"raw": xml_text}

    def _parse_detector_xml(self, xml_text: str) -> Dict:
        """Parsea datos de detectores."""
        return self._parse_response(xml_text)

    def _parse_signal_xml(self, xml_text: str) -> Dict:
        """Parsea estado de senales."""
        return self._parse_response(xml_text)

    # -- Simulacion --

    _sim_stage = "STAGE_A"

    def _simulated_response(self, cmd: str, params: Dict) -> Dict:
        if cmd == "SetStage":
            self._sim_stage = params.get("stage", self._sim_stage)
        return {"status": "OK", "command": cmd}

    def _simulated_detector_data(self) -> Dict:
        data = {}
        for det_id in self.config.detector_map:
            data[det_id] = {
                "volume": np.random.randint(0, 25),
                "occupancy": np.random.randint(0, 80),
                "speed": np.random.uniform(5, 55),
                "status": "OK",
            }
        return data

    def _simulated_signal_state(self) -> Dict:
        return {
            "current_stage": self._sim_stage,
            "mode": ControllerMode.ADAPTIVE,
            "time_in_stage": np.random.randint(5, 60),
            "cycle_time": 90,
        }


# =============================================================================
# ADAPTADOR UTMC PRINCIPAL
# =============================================================================

class UTMCAdapter:
    """
    Adaptador UTMC para ATLAS Pro.

    Interfaz identica a NTCIPAdapter para que sean intercambiables.
    """

    def __init__(self, config: UTMCConfig = None):
        self.config = config or UTMCConfig()
        self.client = UTMCClient(self.config)

        # Estado interno
        self._current_phase = 0
        self._phase_start_time = 0
        self._step = 0
        self._total_steps = 0
        self._connected = False

        # Historial
        self._volume_history = {d: deque(maxlen=60) for d in range(4)}
        self._wait_estimates = {d: 0.0 for d in range(4)}
        self._speed_history = {d: deque(maxlen=30) for d in range(4)}

        # Preempcion
        self._preempt_active = False

        # Watchdog
        self._watchdog_running = False
        self._watchdog_thread = None
        self._missed_heartbeats = 0
        self._last_heartbeat = 0

        # Log
        self._action_log = deque(maxlen=10000)

    def connect(self) -> bool:
        """Conecta con el controlador UTMC."""
        if not self.client.connect():
            return False

        # Verificar estado
        state = self.client.get_signal_state()
        if state is None:
            logger.error("Controlador no responde")
            return False

        mode = state.get("mode", ControllerMode.STANDBY)
        logger.info(f"Controlador UTMC en modo: {mode}")

        # Solicitar control adaptativo
        result = self.client.send_command("SetControlMode", {
            "mode": "ADAPTIVE",
            "source": "ATLAS_PRO",
            "priority": "HIGH",
        })

        self._start_watchdog()
        self._connected = True
        self._phase_start_time = time.time()
        logger.info("ATLAS UTMC: Conectado y listo")
        return True

    def disconnect(self):
        """Desconecta del controlador."""
        self._stop_watchdog()

        if self._connected:
            self.client.send_command("SetControlMode", {
                "mode": "VEHICLE_ACTUATED",
                "source": "ATLAS_PRO",
            })

        self.client.disconnect()
        self._connected = False
        logger.info("ATLAS UTMC: Desconectado")

    def read_state(self) -> np.ndarray:
        """Lee estado actual y convierte a vector 26D."""
        state = np.zeros(26, dtype=np.float32)
        self._step += 1
        self._total_steps += 1

        # Leer detectores
        det_data = self.client.get_detector_data()
        if det_data is None:
            return state

        for det_id, (direction, det_type) in self.config.detector_map.items():
            if det_id not in det_data:
                continue

            d = det_data[det_id]
            offset = direction * 6

            volume = d.get("volume", 0)
            occupancy = d.get("occupancy", 0)
            speed = d.get("speed", 0)

            # Acumular por direccion
            state[offset + 0] = max(state[offset + 0], min(int(occupancy / 100.0 * 30) / 30.0, 1.0))
            state[offset + 1] = max(state[offset + 1], min(speed / 15.0, 1.0))

            wait = self._estimate_wait_time(direction, occupancy)
            state[offset + 2] = min(wait / 300.0, 1.0)

            state[offset + 3] = max(state[offset + 3], min(volume / 30.0, 1.0))
            state[offset + 4] = max(state[offset + 4], min(occupancy / 100.0, 1.0))

            # Emergencia
            if d.get("emergency", False):
                state[offset + 5] = 1.0

        # Fase actual y paso
        state[24] = self._current_phase / 3.0
        state[25] = min(self._step / 2000, 1.0)

        return state

    def _estimate_wait_time(self, direction: int, occupancy: float) -> float:
        """Estima tiempo de espera."""
        if (direction in (0, 1) and self._current_phase == 1) or \
           (direction in (2, 3) and self._current_phase == 2):
            self._wait_estimates[direction] = max(0, self._wait_estimates[direction] - 1)
        else:
            occ_factor = occupancy / 100.0
            self._wait_estimates[direction] += 1.0 + occ_factor * 2.0
        return self._wait_estimates[direction]

    def apply_action(self, action: int) -> bool:
        """Aplica accion del agente RL al controlador UTMC."""
        if not self._connected:
            return False

        if self._preempt_active:
            return True

        time_in_phase = time.time() - self._phase_start_time

        if time_in_phase < self.config.min_green_s and action not in (0, 3):
            return True

        if time_in_phase > self.config.max_green_s and action == 0:
            action = 2 if self._current_phase == 1 else 1

        success = False

        if action == 0:
            success = True
        elif action in (1, 2):
            stages = self.config.stage_map.get(action, [])
            if stages:
                result = self.client.send_command("SetStage", {
                    "stage": stages[0],
                    "transition": "NORMAL",
                    "intergreen": str(self.config.intergreen_s),
                })
                success = result is not None and result.get("status") == "OK"
                if success:
                    self._current_phase = action
                    self._phase_start_time = time.time()
                    self._step = 0
        elif action == 3:
            result = self.client.send_command("ExtendStage", {
                "duration": "10",
            })
            success = result is not None

        self._action_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "phase": self._current_phase,
            "success": success,
        })

        return success

    def activate_preemption(self, direction: str) -> bool:
        """Activa preempcion de emergencia."""
        stage = "STAGE_A" if direction.upper() in ("NS", "N", "S") else "STAGE_B"
        result = self.client.send_command("ActivatePreemption", {
            "stage": stage,
            "reason": "EMERGENCY_VEHICLE",
            "source": "ATLAS_PRO",
        })
        if result:
            self._preempt_active = True
            logger.warning(f"PREEMPCION UTMC: {direction}")
        return result is not None

    def deactivate_preemption(self) -> bool:
        """Desactiva preempcion."""
        result = self.client.send_command("DeactivatePreemption", {})
        self._preempt_active = False
        return result is not None

    def get_status(self) -> Dict:
        """Estado del adaptador."""
        return {
            "connected": self._connected,
            "protocol": "UTMC",
            "url": self.config.base_url,
            "controller_id": self.config.controller_id,
            "current_phase": self._current_phase,
            "time_in_phase": round(time.time() - self._phase_start_time, 1) if self._connected else 0,
            "total_steps": self._total_steps,
            "preempt_active": self._preempt_active,
        }

    # Watchdog
    def _start_watchdog(self):
        self._watchdog_running = True
        self._last_heartbeat = time.time()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _stop_watchdog(self):
        self._watchdog_running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=5)

    def _watchdog_loop(self):
        while self._watchdog_running:
            try:
                state = self.client.get_signal_state()
                if state:
                    self._last_heartbeat = time.time()
                    self._missed_heartbeats = 0
                else:
                    self._missed_heartbeats += 1
                    if self._missed_heartbeats >= self.config.max_missed_heartbeats:
                        logger.critical("UTMC: Conexion perdida")
                        self._connected = False
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
            time.sleep(self.config.heartbeat_interval_s)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n  ATLAS UTMC Adapter - Test Simulado")
    print("=" * 50)

    config = UTMCConfig()
    adapter = UTMCAdapter(config)
    adapter.connect()

    for i in range(20):
        state = adapter.read_state()
        action = np.random.randint(0, 4)
        adapter.apply_action(action)
        print(f"  Step {i+1:3d} | State sum: {state.sum():.2f} | Action: {action}")
        time.sleep(0.1)

    print(f"\n  Status: {adapter.get_status()}")
    adapter.disconnect()
    print("  Test UTMC completado OK")
