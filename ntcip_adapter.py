"""
ATLAS Pro - Adaptador NTCIP 1202 para Controladores de Trafico Reales
======================================================================
Protocolo NTCIP (National Transportation Communications for ITS Protocol)
es el estandar en USA y LATAM para comunicarse con controladores de semaforos.

Este modulo traduce:
  - Acciones ATLAS (0=Mantener, 1=NS, 2=EO, 3=Extender) -> comandos NTCIP
  - Datos de sensores NTCIP -> vector de estado 26D para el agente RL

Basado en NTCIP 1202 v03 (Object Definitions for Actuated Traffic Signal
Controller Units) y NTCIP 1103 (Transport Management Protocol - SNMP).

Soporta:
  - SNMP v1/v2c para GET/SET de objetos ASC
  - Cambio de fases via Phase Control Group
  - Lectura de detectores via Vehicle Detector Group
  - Monitoreo de estado del controlador
  - Modo manual/override para emergencias
  - Heartbeat y watchdog de conexion

Dependencias:
  pip install pysnmp==4.4.12 pysnmp-mibs

Uso:
  adapter = NTCIPAdapter("192.168.1.100", community="atlas")
  adapter.connect()
  state_26d = adapter.read_state()      # estado para el agente RL
  adapter.apply_action(1)                # cambiar a fase N-S
  adapter.disconnect()
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
from datetime import datetime

logger = logging.getLogger("ATLAS.NTCIP")


# =============================================================================
# CONSTANTES NTCIP 1202
# =============================================================================

class NTCIPObjects:
    """OIDs (Object Identifiers) de NTCIP 1202 v03 para ASC."""

    # Base MIB: 1.3.6.1.4.1.1206.4.2.1
    ASC_BASE = "1.3.6.1.4.1.1206.4.2.1"

    # --- Phase Control Group (1.2) ---
    PHASE_STATUS_GROUP_PHASE_STATUS = f"{ASC_BASE}.1.2.1.1.4"      # phaseStatusGroupPhaseStatus
    PHASE_CONTROL_GROUP_PHASE_OMIT = f"{ASC_BASE}.1.2.1.1.5"       # omitir fases
    PHASE_CONTROL_GROUP_PHASE_HOLD = f"{ASC_BASE}.1.2.1.1.6"       # mantener fase actual
    PHASE_CONTROL_GROUP_PHASE_FORCE_OFF = f"{ASC_BASE}.1.2.1.1.7"  # forzar fin de fase
    PHASE_CONTROL_GROUP_PHASE_CALL = f"{ASC_BASE}.1.2.1.1.8"       # solicitar fase

    # --- Phase Timing (1.2.2) ---
    PHASE_MIN_GREEN = f"{ASC_BASE}.1.2.2.1.4"     # minimo verde (segundos)
    PHASE_MAX_GREEN = f"{ASC_BASE}.1.2.2.1.7"     # maximo verde (segundos)
    PHASE_YELLOW = f"{ASC_BASE}.1.2.2.1.8"        # tiempo amarillo
    PHASE_RED_CLEAR = f"{ASC_BASE}.1.2.2.1.9"     # rojo de despeje

    # --- Vehicle Detector Group (4) ---
    DETECTOR_VOLUME = f"{ASC_BASE}.4.1.1.5"       # volumen (vehiculos)
    DETECTOR_OCCUPANCY = f"{ASC_BASE}.4.1.1.6"    # ocupacion (%)
    DETECTOR_SPEED = f"{ASC_BASE}.4.1.1.7"        # velocidad (km/h)
    DETECTOR_STATUS = f"{ASC_BASE}.4.1.1.8"       # estado detector

    # --- Pedestrian Detector Group (5) ---
    PED_DETECTOR_CALL = f"{ASC_BASE}.5.1.1.3"     # llamada peaton

    # --- Unit Control (6) ---
    UNIT_FLASH_STATUS = f"{ASC_BASE}.6.1.0"       # modo flash
    UNIT_ALARM_STATUS = f"{ASC_BASE}.6.2.0"       # alarmas
    UNIT_CONTROL_MODE = f"{ASC_BASE}.6.4.0"       # modo de control

    # --- Coordination (9) ---
    COORD_PATTERN = f"{ASC_BASE}.9.1.0"           # patron de coordinacion
    COORD_CYCLE_LENGTH = f"{ASC_BASE}.9.2.0"      # longitud de ciclo
    COORD_OFFSET = f"{ASC_BASE}.9.3.0"            # offset

    # --- Preempt (7) ---
    PREEMPT_CONTROL = f"{ASC_BASE}.7.1.1.2"       # control de preempcion
    PREEMPT_STATE = f"{ASC_BASE}.7.1.1.3"         # estado de preempcion


class ControlMode(IntEnum):
    """Modos de control del controlador ASC."""
    OTHER = 1
    FLASH = 2
    FREE = 3             # operacion libre (cada fase por demanda)
    COORDINATION = 4     # coordinado con otros controladores
    MANUAL = 5           # control manual (operador)
    ADAPTIVE = 6         # control adaptativo (ATLAS usa este)
    CENTRAL_OVERRIDE = 7 # override central


class PhaseStatus(IntEnum):
    """Estados posibles de una fase."""
    DARK = 0
    RED = 1
    YELLOW = 2
    GREEN_WALK = 3
    GREEN = 4
    GREEN_EXTEND = 5
    YELLOW_CLEAR = 6
    RED_CLEAR = 7
    PREEMPT = 8
    FLASH = 9


# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass
class NTCIPConfig:
    """Configuracion del adaptador NTCIP."""
    # Conexion
    host: str = "192.168.1.100"
    port: int = 161                     # puerto SNMP estandar
    community_read: str = "public"      # SNMP community de lectura
    community_write: str = "atlas"      # SNMP community de escritura
    snmp_version: int = 2               # 1 o 2 (v2c)
    timeout_ms: int = 1000              # timeout de respuesta
    retries: int = 2                    # reintentos

    # Mapeo de fases ATLAS -> fases del controlador
    # ATLAS: 0=Mantener, 1=NS, 2=EO, 3=Extender
    phase_map: Dict[int, List[int]] = field(default_factory=lambda: {
        1: [1, 5],     # Fase N-S -> fases 1 y 5 del controlador (tipicamente N y S)
        2: [3, 7],     # Fase E-O -> fases 3 y 7 del controlador (tipicamente E y O)
    })

    # Mapeo de detectores a direcciones
    # detector_id -> (direccion, tipo)
    # direcciones: N=0, S=1, E=2, W=3
    detector_map: Dict[int, Tuple[int, str]] = field(default_factory=lambda: {
        1: (0, "advance"),    # Detector 1 -> Norte, detector avanzado
        2: (0, "stop_line"),  # Detector 2 -> Norte, linea de parada
        3: (1, "advance"),    # Detector 3 -> Sur
        4: (1, "stop_line"),
        5: (2, "advance"),    # Detector 5 -> Este
        6: (2, "stop_line"),
        7: (3, "advance"),    # Detector 7 -> Oeste
        8: (3, "stop_line"),
    })

    # Limites de seguridad
    min_green_s: int = 7          # minimo verde en segundos
    max_green_s: int = 90         # maximo verde en segundos
    yellow_s: int = 3             # amarillo fijo
    all_red_s: int = 2            # todo rojo de despeje
    max_cycle_s: int = 180        # maximo ciclo en segundos

    # Watchdog
    heartbeat_interval_s: float = 2.0   # heartbeat cada 2 segundos
    connection_timeout_s: float = 10.0  # timeout de conexion
    max_missed_heartbeats: int = 3      # maximo heartbeats perdidos

    # Preempcion de emergencia
    emergency_vehicle_types: List[str] = field(default_factory=lambda: [
        "ambulancia", "policia", "bomberos", "emergencia"
    ])
    preempt_phase_ns: int = 1     # fase de preempcion para emergencia N-S
    preempt_phase_ew: int = 3     # fase de preempcion para emergencia E-O


# =============================================================================
# SNMP CLIENT
# =============================================================================

class SNMPClient:
    """
    Cliente SNMP para comunicacion con controladores NTCIP.

    Abstrae las operaciones GET/SET de SNMP para que el adaptador
    pueda leer/escribir objetos del controlador sin preocuparse
    del protocolo subyacente.
    """

    def __init__(self, config: NTCIPConfig):
        self.config = config
        self._connected = False
        self._engine = None

        # Intentar importar pysnmp
        try:
            from pysnmp.hlapi import (
                SnmpEngine, CommunityData, UdpTransportTarget,
                ContextData, ObjectType, ObjectIdentity,
                getCmd, setCmd, nextCmd
            )
            self._snmp_available = True
            self._snmp = {
                'SnmpEngine': SnmpEngine,
                'CommunityData': CommunityData,
                'UdpTransportTarget': UdpTransportTarget,
                'ContextData': ContextData,
                'ObjectType': ObjectType,
                'ObjectIdentity': ObjectIdentity,
                'getCmd': getCmd,
                'setCmd': setCmd,
                'nextCmd': nextCmd,
            }
        except ImportError:
            self._snmp_available = False
            logger.warning(
                "pysnmp no disponible. Usando modo simulado. "
                "Instalar: pip install pysnmp==4.4.12"
            )

    def connect(self) -> bool:
        """Establece conexion SNMP con el controlador."""
        if not self._snmp_available:
            logger.info("SNMP simulado: conexion virtual establecida")
            self._connected = True
            return True

        try:
            self._engine = self._snmp['SnmpEngine']()
            self._target = self._snmp['UdpTransportTarget'](
                (self.config.host, self.config.port),
                timeout=self.config.timeout_ms / 1000.0,
                retries=self.config.retries
            )
            self._read_auth = self._snmp['CommunityData'](
                self.config.community_read,
                mpModel=self.config.snmp_version - 1
            )
            self._write_auth = self._snmp['CommunityData'](
                self.config.community_write,
                mpModel=self.config.snmp_version - 1
            )
            self._connected = True
            logger.info(f"SNMP conectado a {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Error conectando SNMP: {e}")
            return False

    def disconnect(self):
        """Cierra conexion SNMP."""
        if self._engine:
            try:
                self._engine.transportDispatcher.closeDispatcher()
            except Exception:
                pass
        self._connected = False
        logger.info("SNMP desconectado")

    def get(self, oid: str, index: int = 0) -> Optional[int]:
        """Lee un valor entero de un OID NTCIP."""
        full_oid = f"{oid}.{index}" if index else oid

        if not self._snmp_available:
            return self._simulated_get(full_oid)

        if not self._connected:
            return None

        try:
            result = next(self._snmp['getCmd'](
                self._engine,
                self._read_auth,
                self._target,
                self._snmp['ContextData'](),
                self._snmp['ObjectType'](
                    self._snmp['ObjectIdentity'](full_oid)
                )
            ))
            error_indication, error_status, _, var_binds = result
            if error_indication or error_status:
                logger.warning(f"SNMP GET error [{full_oid}]: {error_indication or error_status}")
                return None
            _, value = var_binds[0]
            return int(value)
        except Exception as e:
            logger.error(f"SNMP GET exception [{full_oid}]: {e}")
            return None

    def set(self, oid: str, value: int, index: int = 0) -> bool:
        """Escribe un valor entero en un OID NTCIP."""
        full_oid = f"{oid}.{index}" if index else oid

        if not self._snmp_available:
            return self._simulated_set(full_oid, value)

        if not self._connected:
            return False

        try:
            from pysnmp.hlapi import Integer32
            result = next(self._snmp['setCmd'](
                self._engine,
                self._write_auth,
                self._target,
                self._snmp['ContextData'](),
                self._snmp['ObjectType'](
                    self._snmp['ObjectIdentity'](full_oid),
                    Integer32(value)
                )
            ))
            error_indication, error_status, _, _ = result
            if error_indication or error_status:
                logger.warning(f"SNMP SET error [{full_oid}={value}]: {error_indication or error_status}")
                return False
            return True
        except Exception as e:
            logger.error(f"SNMP SET exception [{full_oid}={value}]: {e}")
            return False

    # -- Simulacion para testing sin hardware --

    _sim_state = {}

    def _simulated_get(self, oid: str) -> int:
        """Valores simulados para testing."""
        if "4.1.1.5" in oid:  # detector volume
            return np.random.randint(0, 25)
        elif "4.1.1.6" in oid:  # detector occupancy
            return np.random.randint(0, 80)
        elif "4.1.1.7" in oid:  # detector speed
            return np.random.randint(5, 60)
        elif "1.2.1.1.4" in oid:  # phase status
            return self._sim_state.get("phase_status", PhaseStatus.GREEN)
        elif "6.4.0" in oid:  # control mode
            return ControlMode.ADAPTIVE
        return 0

    def _simulated_set(self, oid: str, value: int) -> bool:
        """Escritura simulada."""
        self._sim_state[oid] = value
        logger.debug(f"SNMP SET simulado: {oid} = {value}")
        return True


# =============================================================================
# ADAPTADOR NTCIP PRINCIPAL
# =============================================================================

class NTCIPAdapter:
    """
    Adaptador NTCIP 1202 para ATLAS Pro.

    Traduce entre el mundo del agente RL (acciones 0-3, estados 26D)
    y el protocolo NTCIP del controlador de semaforos real.

    Uso tipico:
        config = NTCIPConfig(host="192.168.1.100", community_write="atlas")
        adapter = NTCIPAdapter(config)
        adapter.connect()

        # Loop de control
        while running:
            state = adapter.read_state()          # 26D numpy array
            action = agent.select_action(state)   # agente RL decide
            adapter.apply_action(action)           # ejecutar en controlador
            time.sleep(adapter.config.min_green_s) # esperar minimo verde

        adapter.disconnect()
    """

    def __init__(self, config: NTCIPConfig = None):
        self.config = config or NTCIPConfig()
        self.snmp = SNMPClient(self.config)

        # Estado interno
        self._current_phase = 0       # fase ATLAS actual (0-3)
        self._phase_start_time = 0    # cuando empezo la fase actual
        self._step = 0                # paso de simulacion
        self._total_steps = 0
        self._connected = False

        # Historial para calcular metricas
        self._volume_history = {d: deque(maxlen=60) for d in range(4)}  # N,S,E,W
        self._wait_estimates = {d: 0.0 for d in range(4)}
        self._speed_history = {d: deque(maxlen=30) for d in range(4)}

        # Preempcion activa
        self._preempt_active = False
        self._preempt_direction = None

        # Watchdog
        self._last_heartbeat = 0
        self._missed_heartbeats = 0
        self._watchdog_thread = None
        self._watchdog_running = False

        # Log de acciones para auditoria
        self._action_log = deque(maxlen=10000)

    def connect(self) -> bool:
        """Conecta con el controlador NTCIP."""
        if not self.snmp.connect():
            return False

        # Verificar que el controlador responde
        mode = self.snmp.get(NTCIPObjects.UNIT_CONTROL_MODE)
        if mode is None:
            logger.error("Controlador no responde al query de modo")
            return False

        logger.info(f"Controlador en modo: {ControlMode(mode).name}")

        # Poner en modo adaptativo para que ATLAS pueda controlar
        if mode != ControlMode.ADAPTIVE:
            logger.info("Cambiando a modo ADAPTIVE para control por ATLAS...")
            if not self.snmp.set(NTCIPObjects.UNIT_CONTROL_MODE, ControlMode.ADAPTIVE):
                logger.warning("No se pudo cambiar a modo ADAPTIVE. Puede requerir autorizacion.")

        # Iniciar watchdog
        self._start_watchdog()

        self._connected = True
        self._phase_start_time = time.time()
        logger.info("ATLAS NTCIP: Conectado y listo para control")
        return True

    def disconnect(self):
        """Desconecta del controlador de forma segura."""
        # Parar watchdog
        self._stop_watchdog()

        # Restaurar modo libre (el controlador vuelve a funcionar solo)
        if self._connected:
            logger.info("Restaurando controlador a modo FREE...")
            self.snmp.set(NTCIPObjects.UNIT_CONTROL_MODE, ControlMode.FREE)

        self.snmp.disconnect()
        self._connected = False
        logger.info("ATLAS NTCIP: Desconectado limpiamente")

    # =========================================================================
    # LECTURA DE ESTADO (Controlador -> 26D)
    # =========================================================================

    def read_state(self) -> np.ndarray:
        """
        Lee el estado actual del controlador y lo convierte a vector 26D.

        Vector de estado (26D):
          Por cada direccion (N=0, S=1, E=2, W=3) -> 6 features = 24
            0: cola_estimada / 30
            1: velocidad_media / 15
            2: tiempo_espera_estimado / 300
            3: num_vehiculos / 30
            4: ocupacion / 100
            5: hay_emergencia (0 o 1)
          + fase_actual_normalizada (0-1)
          + paso_normalizado (0-1)
        """
        state = np.zeros(26, dtype=np.float32)
        self._step += 1
        self._total_steps += 1

        for direction in range(4):  # N, S, E, W
            offset = direction * 6

            # Leer detectores para esta direccion
            det_data = self._read_detectors(direction)

            # Feature 0: Cola estimada (vehiculos parados / 30)
            queue = det_data["queue_estimate"]
            state[offset + 0] = min(queue / 30.0, 1.0)

            # Feature 1: Velocidad media / 15
            speed = det_data["speed"]
            state[offset + 1] = min(speed / 15.0, 1.0)

            # Feature 2: Tiempo espera estimado / 300
            wait = self._estimate_wait_time(direction, det_data)
            state[offset + 2] = min(wait / 300.0, 1.0)

            # Feature 3: Num vehiculos / 30
            volume = det_data["volume"]
            state[offset + 3] = min(volume / 30.0, 1.0)

            # Feature 4: Ocupacion / 100
            occupancy = det_data["occupancy"]
            state[offset + 4] = min(occupancy / 100.0, 1.0)

            # Feature 5: Emergencia (0 o 1)
            state[offset + 5] = 1.0 if self._detect_emergency(direction) else 0.0

        # Feature 24: Fase actual normalizada
        state[24] = self._current_phase / 3.0

        # Feature 25: Paso normalizado
        max_steps = 2000
        state[25] = min(self._step / max_steps, 1.0)

        return state

    def _read_detectors(self, direction: int) -> Dict:
        """Lee datos de detectores para una direccion."""
        data = {"volume": 0, "occupancy": 0, "speed": 0.0, "queue_estimate": 0}

        for det_id, (det_dir, det_type) in self.config.detector_map.items():
            if det_dir != direction:
                continue

            vol = self.snmp.get(NTCIPObjects.DETECTOR_VOLUME, det_id)
            occ = self.snmp.get(NTCIPObjects.DETECTOR_OCCUPANCY, det_id)
            spd = self.snmp.get(NTCIPObjects.DETECTOR_SPEED, det_id)

            if vol is not None:
                data["volume"] += vol
                self._volume_history[direction].append(vol)

            if occ is not None:
                data["occupancy"] = max(data["occupancy"], occ)

            if spd is not None and spd > 0:
                data["speed"] = spd
                self._speed_history[direction].append(spd)

            # Estimar cola desde ocupacion alta + velocidad baja en stop_line
            if det_type == "stop_line" and occ is not None and occ > 50:
                avg_veh_length = 6.0  # metros
                det_length = 2.0      # metros
                data["queue_estimate"] = int(occ / 100.0 * 30)  # estimacion

        # Velocidad media de historial
        if self._speed_history[direction]:
            data["speed"] = float(np.mean(list(self._speed_history[direction])[-10:]))

        return data

    def _estimate_wait_time(self, direction: int, det_data: Dict) -> float:
        """Estima tiempo de espera basandose en la fase actual y ocupacion."""
        # Si esta direccion tiene verde, espera = 0
        if (direction in (0, 1) and self._current_phase == 1) or \
           (direction in (2, 3) and self._current_phase == 2):
            self._wait_estimates[direction] = max(0, self._wait_estimates[direction] - 1)
        else:
            # Espera aumenta proporcional a la ocupacion
            occ_factor = det_data["occupancy"] / 100.0
            self._wait_estimates[direction] += 1.0 + occ_factor * 2.0

        return self._wait_estimates[direction]

    def _detect_emergency(self, direction: int) -> bool:
        """
        Detecta vehiculos de emergencia.

        En un sistema real, esto vendria de:
        - Detectores de preempcion (sirena/GPS)
        - Preempt Control Group de NTCIP
        - Sistema AVL (Automatic Vehicle Location) externo
        """
        # Leer estado de preempcion del controlador
        preempt = self.snmp.get(NTCIPObjects.PREEMPT_STATE, direction + 1)
        if preempt is not None and preempt > 0:
            return True

        return False

    # =========================================================================
    # APLICAR ACCIONES (ATLAS -> Controlador)
    # =========================================================================

    def apply_action(self, action: int) -> bool:
        """
        Aplica una accion del agente RL al controlador NTCIP.

        Acciones ATLAS:
          0 = Mantener fase actual
          1 = Activar fase N-S (fases 1,5 del controlador)
          2 = Activar fase E-O (fases 3,7 del controlador)
          3 = Extender fase actual

        Returns:
            True si la accion se aplico correctamente
        """
        if not self._connected:
            logger.error("No conectado al controlador")
            return False

        # Preempcion tiene prioridad absoluta
        if self._preempt_active:
            logger.info("Preempcion activa - ignorando accion del agente")
            return True

        # Verificar tiempo minimo de verde
        time_in_phase = time.time() - self._phase_start_time
        if time_in_phase < self.config.min_green_s and action != 0 and action != 3:
            logger.debug(
                f"Accion {action} ignorada: minimo verde no alcanzado "
                f"({time_in_phase:.1f}s < {self.config.min_green_s}s)"
            )
            return True  # No es un error, solo se ignora

        # Verificar tiempo maximo de verde
        if time_in_phase > self.config.max_green_s and action == 0:
            logger.warning(f"Verde maximo excedido ({time_in_phase:.1f}s). Forzando cambio.")
            action = 2 if self._current_phase == 1 else 1

        success = False

        if action == 0:
            # Mantener: no hacer nada
            success = True

        elif action == 1:
            # Activar N-S
            success = self._set_phase(1)

        elif action == 2:
            # Activar E-O
            success = self._set_phase(2)

        elif action == 3:
            # Extender: mantener fase actual un poco mas
            # Implementado como HOLD en NTCIP
            success = self._extend_phase()

        else:
            logger.error(f"Accion invalida: {action}")
            return False

        # Log para auditoria
        self._action_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "previous_phase": self._current_phase,
            "time_in_phase": round(time_in_phase, 1),
            "success": success,
        })

        return success

    def _set_phase(self, atlas_phase: int) -> bool:
        """
        Cambia a una fase especifica.

        Proceso NTCIP:
        1. Force-off la fase actual (si es diferente)
        2. Place call para la nueva fase
        3. El controlador maneja la transicion (amarillo + todo-rojo)
        """
        if atlas_phase == self._current_phase:
            return True  # Ya estamos en esa fase

        controller_phases = self.config.phase_map.get(atlas_phase, [])
        if not controller_phases:
            logger.error(f"Fase ATLAS {atlas_phase} no mapeada a fases del controlador")
            return False

        # Force-off fases actuales (si las hay)
        current_ctrl_phases = self.config.phase_map.get(self._current_phase, [])
        for phase_num in current_ctrl_phases:
            self.snmp.set(NTCIPObjects.PHASE_CONTROL_GROUP_PHASE_FORCE_OFF, 1, phase_num)

        # Call nuevas fases
        for phase_num in controller_phases:
            if not self.snmp.set(NTCIPObjects.PHASE_CONTROL_GROUP_PHASE_CALL, 1, phase_num):
                logger.error(f"Fallo al solicitar fase {phase_num}")
                return False

        old_phase = self._current_phase
        self._current_phase = atlas_phase
        self._phase_start_time = time.time()
        self._step = 0

        logger.info(f"Fase cambiada: {old_phase} -> {atlas_phase} "
                     f"(controlador: {controller_phases})")
        return True

    def _extend_phase(self) -> bool:
        """Extiende la fase actual (HOLD en NTCIP)."""
        controller_phases = self.config.phase_map.get(self._current_phase, [])
        for phase_num in controller_phases:
            self.snmp.set(NTCIPObjects.PHASE_CONTROL_GROUP_PHASE_HOLD, 1, phase_num)

        logger.debug(f"Fase {self._current_phase} extendida (HOLD)")
        return True

    # =========================================================================
    # PREEMPCION (Emergencias)
    # =========================================================================

    def activate_preemption(self, direction: str) -> bool:
        """
        Activa preempcion de emergencia.

        Args:
            direction: "NS" o "EW" segun por donde viene la emergencia
        """
        if direction.upper() in ("NS", "N-S", "N", "S"):
            preempt_phase = self.config.preempt_phase_ns
        elif direction.upper() in ("EW", "E-O", "E", "W"):
            preempt_phase = self.config.preempt_phase_ew
        else:
            logger.error(f"Direccion de preempcion invalida: {direction}")
            return False

        success = self.snmp.set(NTCIPObjects.PREEMPT_CONTROL, 1, preempt_phase)
        if success:
            self._preempt_active = True
            self._preempt_direction = direction
            logger.warning(f"PREEMPCION ACTIVADA: direccion {direction}")
        return success

    def deactivate_preemption(self) -> bool:
        """Desactiva preempcion de emergencia."""
        success = True
        for phase in [self.config.preempt_phase_ns, self.config.preempt_phase_ew]:
            if not self.snmp.set(NTCIPObjects.PREEMPT_CONTROL, 0, phase):
                success = False

        self._preempt_active = False
        self._preempt_direction = None
        logger.info("Preempcion desactivada")
        return success

    # =========================================================================
    # WATCHDOG
    # =========================================================================

    def _start_watchdog(self):
        """Inicia el thread de heartbeat/watchdog."""
        self._watchdog_running = True
        self._last_heartbeat = time.time()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        logger.info("Watchdog iniciado")

    def _stop_watchdog(self):
        """Para el watchdog."""
        self._watchdog_running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=5)

    def _watchdog_loop(self):
        """Loop del watchdog: verifica conexion periodicamente."""
        while self._watchdog_running:
            try:
                # Leer algo simple para verificar conexion
                mode = self.snmp.get(NTCIPObjects.UNIT_CONTROL_MODE)

                if mode is not None:
                    self._last_heartbeat = time.time()
                    self._missed_heartbeats = 0

                    # Verificar alarmas
                    alarm = self.snmp.get(NTCIPObjects.UNIT_ALARM_STATUS)
                    if alarm and alarm > 0:
                        logger.warning(f"ALARMA del controlador: codigo {alarm}")

                    # Verificar flash
                    flash = self.snmp.get(NTCIPObjects.UNIT_FLASH_STATUS)
                    if flash and flash > 0:
                        logger.critical("CONTROLADOR EN MODO FLASH - posible fallo")
                        self._handle_controller_fault()
                else:
                    self._missed_heartbeats += 1
                    logger.warning(
                        f"Heartbeat perdido ({self._missed_heartbeats}/"
                        f"{self.config.max_missed_heartbeats})"
                    )

                    if self._missed_heartbeats >= self.config.max_missed_heartbeats:
                        logger.critical("Demasiados heartbeats perdidos - activando fallback")
                        self._handle_connection_loss()

            except Exception as e:
                logger.error(f"Error en watchdog: {e}")

            time.sleep(self.config.heartbeat_interval_s)

    def _handle_controller_fault(self):
        """Maneja fallo del controlador."""
        logger.critical("FALLO DEL CONTROLADOR DETECTADO")
        # El controlador ya esta en flash, no intentar controlar
        self._connected = False

    def _handle_connection_loss(self):
        """Maneja perdida de conexion."""
        logger.critical("PERDIDA DE CONEXION CON CONTROLADOR")
        # El controlador volvera a modo libre automaticamente (timeout NTCIP)
        self._connected = False

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def get_status(self) -> Dict:
        """Retorna estado completo del adaptador."""
        return {
            "connected": self._connected,
            "host": self.config.host,
            "current_phase": self._current_phase,
            "time_in_phase": round(time.time() - self._phase_start_time, 1) if self._connected else 0,
            "total_steps": self._total_steps,
            "preempt_active": self._preempt_active,
            "preempt_direction": self._preempt_direction,
            "missed_heartbeats": self._missed_heartbeats,
            "last_heartbeat_ago": round(time.time() - self._last_heartbeat, 1) if self._last_heartbeat else None,
            "action_log_size": len(self._action_log),
        }

    def get_action_log(self, last_n: int = 50) -> List[Dict]:
        """Retorna las ultimas N acciones ejecutadas."""
        return list(self._action_log)[-last_n:]

    def reset_step_counter(self):
        """Resetea el contador de pasos (para nuevo episodio)."""
        self._step = 0


# =============================================================================
# CONTROLADOR DE PRODUCCION
# =============================================================================

class ATLASProductionController:
    """
    Controlador de produccion completo que integra:
    - NTCIPAdapter para hardware real
    - Agente RL para decision
    - MUSE para metacognicion
    - Sistema de seguridad
    - Logging completo

    Este es el modulo que corre en edge (Raspberry Pi / Jetson)
    controlando una interseccion real.
    """

    def __init__(self, config_path: str = "train_config.yaml",
                 ntcip_config: NTCIPConfig = None,
                 model_path: str = None):
        self.config_path = config_path
        self.ntcip_config = ntcip_config or NTCIPConfig()
        self.model_path = model_path

        self._running = False
        self._agent = None
        self._muse = None
        self._adapter = None

    def setup(self) -> bool:
        """Inicializa todos los componentes."""
        import yaml

        # Cargar config
        if not os.path.exists(self.config_path):
            logger.error(f"Config no encontrada: {self.config_path}")
            return False

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Cargar agente RL
        try:
            from algoritmos_avanzados import AgenteDuelingDDQN
            agent_cfg = self.config.get("agent", {})
            self._agent = AgenteDuelingDDQN(config=agent_cfg)

            if self.model_path and os.path.exists(self.model_path):
                if self._agent.load(self.model_path):
                    logger.info(f"Modelo cargado: {self.model_path}")
                else:
                    logger.error("Fallo cargando modelo")
                    return False
            else:
                logger.error("Se requiere un modelo entrenado para produccion")
                return False
        except ImportError:
            logger.error("algoritmos_avanzados.py no encontrado")
            return False

        # Cargar MUSE (opcional pero recomendado)
        try:
            from muse_metacognicion import MUSEController
            self._muse = MUSEController(self._agent, self.config)
            muse_path = self.model_path.replace("agente_pro_", "muse_")
            if os.path.exists(muse_path):
                self._muse.load(muse_path)
                logger.info(f"MUSE cargado: {muse_path}")
        except ImportError:
            logger.warning("MUSE no disponible, continuando sin metacognicion")

        # Conectar con controlador
        self._adapter = NTCIPAdapter(self.ntcip_config)
        if not self._adapter.connect():
            logger.error("No se pudo conectar con el controlador")
            return False

        logger.info("ATLAS Production Controller: LISTO")
        return True

    def run(self, max_cycles: int = 0):
        """
        Loop principal de control.

        Args:
            max_cycles: 0 = infinito (produccion real)
        """
        self._running = True
        cycle = 0

        logger.info("=" * 60)
        logger.info("  ATLAS Production Controller: INICIANDO CONTROL")
        logger.info("=" * 60)

        try:
            while self._running:
                cycle += 1
                if max_cycles > 0 and cycle > max_cycles:
                    break

                # 1. Leer estado
                state = self._adapter.read_state()

                # 2. Decidir accion (con o sin MUSE)
                if self._muse:
                    action = self._muse.act(state)
                else:
                    action = self._agent.select_action(state)

                # 3. Aplicar accion
                self._adapter.apply_action(action)

                # 4. Esperar intervalo de decision
                delta_time = self.config.get("environment", {}).get("delta_time", 10)
                time.sleep(delta_time)

                # 5. Log periodico
                if cycle % 30 == 0:
                    status = self._adapter.get_status()
                    logger.info(
                        f"Ciclo {cycle} | Fase: {status['current_phase']} | "
                        f"Tiempo fase: {status['time_in_phase']}s"
                    )
                    if self._muse:
                        diag = self._muse.last_diagnosis
                        logger.info(
                            f"  MUSE: comp={diag['competence']:.2f} "
                            f"nov={diag['novelty']:.2f} "
                            f"strat={diag['strategy']}"
                        )

        except KeyboardInterrupt:
            logger.info("Control interrumpido por el usuario")
        except Exception as e:
            logger.critical(f"Error critico en loop de control: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Para el controlador de forma segura."""
        self._running = False
        if self._adapter:
            self._adapter.disconnect()
        logger.info("ATLAS Production Controller: DETENIDO")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATLAS NTCIP Adapter")
    parser.add_argument("--host", default="192.168.1.100", help="IP del controlador")
    parser.add_argument("--port", type=int, default=161, help="Puerto SNMP")
    parser.add_argument("--community", default="atlas", help="SNMP community")
    parser.add_argument("--test", action="store_true", help="Modo test (simulado)")
    parser.add_argument("--model", default="modelos/agente_pro_heavy.pt", help="Modelo a usar")
    parser.add_argument("--config", default="train_config.yaml", help="Config YAML")
    parser.add_argument("--cycles", type=int, default=100, help="Ciclos de control (0=infinito)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    config = NTCIPConfig(
        host=args.host,
        port=args.port,
        community_write=args.community,
    )

    if args.test:
        print("\n  ATLAS NTCIP - Modo Test (Simulado)")
        print("=" * 50)
        adapter = NTCIPAdapter(config)
        adapter.connect()

        for i in range(20):
            state = adapter.read_state()
            action = np.random.randint(0, 4)
            adapter.apply_action(action)
            print(f"  Step {i+1:3d} | State sum: {state.sum():.2f} | Action: {action}")
            time.sleep(0.1)

        print(f"\n  Status: {adapter.get_status()}")
        adapter.disconnect()
        print("  Test completado OK")
    else:
        controller = ATLASProductionController(
            config_path=args.config,
            ntcip_config=config,
            model_path=args.model,
        )
        if controller.setup():
            controller.run(max_cycles=args.cycles)
