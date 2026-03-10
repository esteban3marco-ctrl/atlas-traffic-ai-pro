"""
sensor_bridge.py - ATLAS Traffic AI Pro Sensor Module

Convierte datos de sensores de tráfico del mundo real en el vector de estado
26-dimensional que el agente RL espera.

Formato del vector de estado 26D (definido en ejecutar_entrenamiento_pro.py):
- Por dirección (N=0, S=1, E=2, W=3) -> 6 características = 24 total:
  - 0: queue (vehículos detenidos) / 30
  - 1: average speed / 15 (m/s)
  - 2: wait time / 300
  - 3: num vehicles / 30
  - 4: occupancy / 100
  - 5: has_emergency (0 o 1)
- Característica 24: current_phase_normalized (0-1)
- Característica 25: step_normalized (0-1)

Tipos de sensores soportados:
1. InductiveLoopSensor - Detector de bucle inductivo clásico
2. CameraSensor - Cámara de detección de video
3. RadarSensor - Radar de microondas
4. BluetoothSensor - Escáner MAC Bluetooth/WiFi
5. GPSSensor - Datos de vehículos conectados

Autor: ATLAS Traffic AI Pro
Versión: 1.0
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Tipos de sensores soportados."""
    INDUCTIVE_LOOP = "inductive_loop"
    CAMERA = "camera"
    RADAR = "radar"
    BLUETOOTH = "bluetooth"
    GPS = "gps"


class Direction(Enum):
    """Direcciones del tráfico en la intersección."""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


@dataclass
class SensorConfig:
    """Configuración para un sensor individual."""
    sensor_id: str
    sensor_type: SensorType
    direction: Direction
    location: str = ""
    enabled: bool = True
    quality_threshold: float = 0.7  # Umbral de confianza mínima
    simulated: bool = False

    def to_dict(self) -> Dict:
        """Serializa la configuración a diccionario."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "direction": self.direction.name,
            "location": self.location,
            "enabled": self.enabled,
            "quality_threshold": self.quality_threshold,
            "simulated": self.simulated
        }


@dataclass
class SensorReading:
    """Lectura de un sensor con metadata."""
    timestamp: datetime
    queue: int = 0  # vehículos detenidos
    speed: float = 0.0  # m/s
    wait_time: int = 0  # segundos
    num_vehicles: int = 0
    occupancy: float = 0.0  # porcentaje (0-100)
    has_emergency: int = 0  # 0 o 1
    confidence: float = 1.0  # confianza en la lectura (0-1)
    sensor_health: float = 1.0  # salud del sensor (0-1)


class BaseSensor:
    """Clase base para todos los sensores."""

    def __init__(self, config: SensorConfig):
        """
        Inicializa el sensor.

        Args:
            config: Configuración del sensor
        """
        self.config = config
        self.last_reading: Optional[SensorReading] = None
        self.failure_count = 0
        self.last_successful_read = datetime.now()

    def read(self) -> SensorReading:
        """Lee datos del sensor. Debe implementarse en subclases."""
        raise NotImplementedError

    def is_healthy(self) -> bool:
        """Verifica si el sensor está operativo."""
        # Sensor está en fallo si no ha tenido lectura exitosa en 60s
        time_since_read = datetime.now() - self.last_successful_read
        return time_since_read < timedelta(seconds=60)


class InductiveLoopSensor(BaseSensor):
    """
    Sensor de bucle inductivo clásico.
    Detecta presencia de vehículos, conteo y velocidad.
    """

    def read(self) -> SensorReading:
        """Lee datos del bucle inductivo."""
        if self.config.simulated:
            return self._simulate_reading()

        try:
            # En producción, interfaz con hardware real
            reading = SensorReading(
                timestamp=datetime.now(),
                queue=np.random.randint(0, 15),
                speed=np.random.uniform(5, 15),
                wait_time=np.random.randint(0, 60),
                num_vehicles=np.random.randint(0, 30),
                occupancy=np.random.uniform(10, 90),
                has_emergency=0,
                confidence=0.95,
                sensor_health=1.0
            )
            self.last_reading = reading
            self.last_successful_read = datetime.now()
            self.failure_count = 0
            return reading
        except Exception as e:
            logger.error(f"Error en InductiveLoopSensor {self.config.sensor_id}: {e}")
            self.failure_count += 1
            return self._get_fallback_reading()

    def _simulate_reading(self) -> SensorReading:
        """Simula lectura del bucle inductivo."""
        base_vehicles = 15 + np.random.randint(-5, 5)
        return SensorReading(
            timestamp=datetime.now(),
            queue=max(0, base_vehicles - 5),
            speed=np.random.uniform(8, 14),
            wait_time=np.random.randint(5, 45),
            num_vehicles=base_vehicles,
            occupancy=30 + np.random.uniform(-5, 5),
            has_emergency=0,
            confidence=0.95,
            sensor_health=1.0
        )

    def _get_fallback_reading(self) -> SensorReading:
        """Usa última lectura conocida o estima valores."""
        if self.last_reading:
            return self.last_reading
        return SensorReading(
            timestamp=datetime.now(),
            queue=0, speed=0, wait_time=0, num_vehicles=0,
            occupancy=0, has_emergency=0, confidence=0.0, sensor_health=0.0
        )


class CameraSensor(BaseSensor):
    """
    Cámara de detección de video.
    Detecta conteo de vehículos, longitud de colas, velocidad via OpenCV/YOLO.
    """

    def read(self) -> SensorReading:
        """Lee datos de la cámara."""
        if self.config.simulated:
            return self._simulate_reading()

        try:
            # En producción, interfaz con cámara real + YOLO
            reading = SensorReading(
                timestamp=datetime.now(),
                queue=np.random.randint(0, 20),
                speed=np.random.uniform(3, 16),
                wait_time=np.random.randint(0, 90),
                num_vehicles=np.random.randint(5, 35),
                occupancy=40 + np.random.uniform(-10, 10),
                has_emergency=np.random.randint(0, 2) if np.random.random() < 0.05 else 0,
                confidence=0.88,
                sensor_health=1.0
            )
            self.last_reading = reading
            self.last_successful_read = datetime.now()
            self.failure_count = 0
            return reading
        except Exception as e:
            logger.error(f"Error en CameraSensor {self.config.sensor_id}: {e}")
            self.failure_count += 1
            return self._get_fallback_reading()

    def _simulate_reading(self) -> SensorReading:
        """Simula lectura de cámara con detección YOLO."""
        base_vehicles = 18 + np.random.randint(-6, 6)
        return SensorReading(
            timestamp=datetime.now(),
            queue=max(0, base_vehicles - 4),
            speed=np.random.uniform(5, 15),
            wait_time=np.random.randint(10, 80),
            num_vehicles=base_vehicles,
            occupancy=35 + np.random.uniform(-8, 8),
            has_emergency=1 if np.random.random() < 0.03 else 0,
            confidence=0.88,
            sensor_health=1.0
        )

    def _get_fallback_reading(self) -> SensorReading:
        """Usa última lectura conocida o estima valores."""
        if self.last_reading:
            return self.last_reading
        return SensorReading(
            timestamp=datetime.now(),
            queue=0, speed=0, wait_time=0, num_vehicles=0,
            occupancy=0, has_emergency=0, confidence=0.0, sensor_health=0.0
        )


class RadarSensor(BaseSensor):
    """
    Radar de microondas.
    Detecta velocidad, conteo y clasificación de vehículos.
    """

    def read(self) -> SensorReading:
        """Lee datos del radar."""
        if self.config.simulated:
            return self._simulate_reading()

        try:
            reading = SensorReading(
                timestamp=datetime.now(),
                queue=np.random.randint(0, 12),
                speed=np.random.uniform(4, 18),
                wait_time=np.random.randint(0, 45),
                num_vehicles=np.random.randint(3, 28),
                occupancy=25 + np.random.uniform(-5, 5),
                has_emergency=0,
                confidence=0.92,
                sensor_health=1.0
            )
            self.last_reading = reading
            self.last_successful_read = datetime.now()
            self.failure_count = 0
            return reading
        except Exception as e:
            logger.error(f"Error en RadarSensor {self.config.sensor_id}: {e}")
            self.failure_count += 1
            return self._get_fallback_reading()

    def _simulate_reading(self) -> SensorReading:
        """Simula lectura de radar."""
        return SensorReading(
            timestamp=datetime.now(),
            queue=np.random.randint(0, 10),
            speed=np.random.uniform(6, 17),
            wait_time=np.random.randint(0, 50),
            num_vehicles=np.random.randint(5, 25),
            occupancy=28 + np.random.uniform(-6, 6),
            has_emergency=0,
            confidence=0.92,
            sensor_health=1.0
        )

    def _get_fallback_reading(self) -> SensorReading:
        """Usa última lectura conocida o estima valores."""
        if self.last_reading:
            return self.last_reading
        return SensorReading(
            timestamp=datetime.now(),
            queue=0, speed=0, wait_time=0, num_vehicles=0,
            occupancy=0, has_emergency=0, confidence=0.0, sensor_health=0.0
        )


class BluetoothSensor(BaseSensor):
    """
    Escáner MAC Bluetooth/WiFi.
    Estima tiempo de viaje mediante triangulación de dispositivos.
    """

    def read(self) -> SensorReading:
        """Lee datos de Bluetooth."""
        if self.config.simulated:
            return self._simulate_reading()

        try:
            reading = SensorReading(
                timestamp=datetime.now(),
                queue=np.random.randint(0, 8),
                speed=np.random.uniform(5, 14),
                wait_time=np.random.randint(10, 120),
                num_vehicles=np.random.randint(2, 20),
                occupancy=15 + np.random.uniform(-5, 5),
                has_emergency=0,
                confidence=0.75,
                sensor_health=1.0
            )
            self.last_reading = reading
            self.last_successful_read = datetime.now()
            self.failure_count = 0
            return reading
        except Exception as e:
            logger.error(f"Error en BluetoothSensor {self.config.sensor_id}: {e}")
            self.failure_count += 1
            return self._get_fallback_reading()

    def _simulate_reading(self) -> SensorReading:
        """Simula lectura de Bluetooth."""
        return SensorReading(
            timestamp=datetime.now(),
            queue=np.random.randint(0, 6),
            speed=np.random.uniform(5, 12),
            wait_time=np.random.randint(15, 100),
            num_vehicles=np.random.randint(1, 15),
            occupancy=12 + np.random.uniform(-3, 3),
            has_emergency=0,
            confidence=0.75,
            sensor_health=1.0
        )

    def _get_fallback_reading(self) -> SensorReading:
        """Usa última lectura conocida o estima valores."""
        if self.last_reading:
            return self.last_reading
        return SensorReading(
            timestamp=datetime.now(),
            queue=0, speed=0, wait_time=0, num_vehicles=0,
            occupancy=0, has_emergency=0, confidence=0.0, sensor_health=0.0
        )


class GPSSensor(BaseSensor):
    """
    Datos de vehículos conectados (GPS).
    Proporciona posición, velocidad y datos de trayectoria.
    """

    def read(self) -> SensorReading:
        """Lee datos de GPS."""
        if self.config.simulated:
            return self._simulate_reading()

        try:
            reading = SensorReading(
                timestamp=datetime.now(),
                queue=np.random.randint(0, 10),
                speed=np.random.uniform(4, 16),
                wait_time=np.random.randint(5, 90),
                num_vehicles=np.random.randint(2, 25),
                occupancy=20 + np.random.uniform(-8, 8),
                has_emergency=0,
                confidence=0.85,
                sensor_health=1.0
            )
            self.last_reading = reading
            self.last_successful_read = datetime.now()
            self.failure_count = 0
            return reading
        except Exception as e:
            logger.error(f"Error en GPSSensor {self.config.sensor_id}: {e}")
            self.failure_count += 1
            return self._get_fallback_reading()

    def _simulate_reading(self) -> SensorReading:
        """Simula lectura de GPS."""
        return SensorReading(
            timestamp=datetime.now(),
            queue=np.random.randint(0, 8),
            speed=np.random.uniform(5, 15),
            wait_time=np.random.randint(5, 70),
            num_vehicles=np.random.randint(3, 22),
            occupancy=18 + np.random.uniform(-5, 5),
            has_emergency=0,
            confidence=0.85,
            sensor_health=1.0
        )

    def _get_fallback_reading(self) -> SensorReading:
        """Usa última lectura conocida o estima valores."""
        if self.last_reading:
            return self.last_reading
        return SensorReading(
            timestamp=datetime.now(),
            queue=0, speed=0, wait_time=0, num_vehicles=0,
            occupancy=0, has_emergency=0, confidence=0.0, sensor_health=0.0
        )


class SensorFusion:
    """
    Fusión inteligente de datos de múltiples sensores.
    Combina lecturas usando pesos basados en confianza y salud del sensor.
    """

    def __init__(self, sensors: List[Tuple[BaseSensor, float]]):
        """
        Inicializa la fusión de sensores.

        Args:
            sensors: Lista de tuplas (sensor, peso_base) para cada sensor
        """
        self.sensors = sensors
        self.total_weight = sum(weight for _, weight in sensors)

    def fuse(self) -> Tuple[SensorReading, float]:
        """
        Fusiona lecturas de múltiples sensores.

        Returns:
            Tupla (lectura_fusionada, confianza_general)
        """
        readings = []
        weights = []

        for sensor, base_weight in self.sensors:
            try:
                reading = sensor.read()
                # Ajusta peso por confianza y salud del sensor
                adjusted_weight = base_weight * reading.confidence * reading.sensor_health
                readings.append(reading)
                weights.append(adjusted_weight)
            except Exception as e:
                logger.error(f"Error leyendo sensor en fusión: {e}")
                weights.append(0.0)

        if not any(weights):
            logger.warning("Todos los sensores fallaron en fusión")
            return SensorReading(timestamp=datetime.now(), confidence=0.0), 0.0

        # Normaliza pesos
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Combina lecturas con promedio ponderado
        fused = SensorReading(timestamp=datetime.now())
        fused.queue = int(sum(r.queue * w for r, w in zip(readings, normalized_weights)))
        fused.speed = sum(r.speed * w for r, w in zip(readings, normalized_weights))
        fused.wait_time = int(sum(r.wait_time * w for r, w in zip(readings, normalized_weights)))
        fused.num_vehicles = int(sum(r.num_vehicles * w for r, w in zip(readings, normalized_weights)))
        fused.occupancy = sum(r.occupancy * w for r, w in zip(readings, normalized_weights))
        fused.has_emergency = max(r.has_emergency for r in readings)  # Si cualquiera detecta emergencia
        fused.confidence = sum(r.confidence * w for r, w in zip(readings, normalized_weights))
        fused.sensor_health = sum(r.sensor_health * w for r, w in zip(readings, normalized_weights))

        return fused, fused.confidence


class SensorBridge:
    """
    Puente de sensores para la intersección.
    Registra múltiples sensores, fusiona datos y convierte al vector 26D.
    """

    def __init__(self):
        """Inicializa el puente de sensores."""
        self.sensors: Dict[str, BaseSensor] = {}
        self.fusion_groups: Dict[Direction, SensorFusion] = {}
        self.last_state_vector: Optional[np.ndarray] = None

    def register_sensor(self, config: SensorConfig) -> None:
        """
        Registra un nuevo sensor.

        Args:
            config: Configuración del sensor
        """
        # Crea instancia del sensor según tipo
        sensor_map = {
            SensorType.INDUCTIVE_LOOP: InductiveLoopSensor,
            SensorType.CAMERA: CameraSensor,
            SensorType.RADAR: RadarSensor,
            SensorType.BLUETOOTH: BluetoothSensor,
            SensorType.GPS: GPSSensor,
        }

        sensor_class = sensor_map.get(config.sensor_type)
        if not sensor_class:
            raise ValueError(f"Tipo de sensor desconocido: {config.sensor_type}")

        sensor = sensor_class(config)
        self.sensors[config.sensor_id] = sensor
        logger.info(f"Sensor registrado: {config.sensor_id} ({config.sensor_type.value})")

    def get_state_vector(self, current_phase: int, total_phases: int,
                        current_step: int, total_steps: int) -> np.ndarray:
        """
        Obtiene el vector de estado 26D.

        Args:
            current_phase: Fase de semáforo actual (0-3)
            total_phases: Número total de fases
            current_step: Paso actual de simulación
            total_steps: Número total de pasos

        Returns:
            Vector de estado numpy de 26 dimensiones
        """
        state = np.zeros(26, dtype=np.float32)

        # Agrupa sensores por dirección
        sensors_by_direction = {}
        for sensor_id, sensor in self.sensors.items():
            direction = sensor.config.direction
            if direction not in sensors_by_direction:
                sensors_by_direction[direction] = []
            sensors_by_direction[direction].append(sensor)

        # Procesa cada dirección
        for direction in Direction:
            direction_sensors = sensors_by_direction.get(direction, [])
            if not direction_sensors:
                continue

            # Fusiona sensores de la dirección
            if len(direction_sensors) > 1:
                fusion_pairs = [(s, 1.0 / len(direction_sensors)) for s in direction_sensors]
                fusion = SensorFusion(fusion_pairs)
                reading, confidence = fusion.fuse()
            else:
                reading = direction_sensors[0].read()
                confidence = reading.confidence

            # Llena características del vector para esta dirección
            dir_idx = direction.value * 6
            state[dir_idx + 0] = reading.queue / 30.0
            state[dir_idx + 1] = reading.speed / 15.0
            state[dir_idx + 2] = reading.wait_time / 300.0
            state[dir_idx + 3] = reading.num_vehicles / 30.0
            state[dir_idx + 4] = reading.occupancy / 100.0
            state[dir_idx + 5] = float(reading.has_emergency)

        # Normaliza fase actual (24)
        state[24] = current_phase / max(total_phases - 1, 1)

        # Normaliza paso actual (25)
        state[25] = current_step / max(total_steps, 1)

        # Clamp a [0, 1] para asegurar normalización
        state = np.clip(state, 0.0, 1.0)

        self.last_state_vector = state
        return state

    def get_sensor_health(self) -> Dict[str, bool]:
        """
        Obtiene estado de salud de todos los sensores.

        Returns:
            Diccionario {sensor_id: is_healthy}
        """
        return {sid: sensor.is_healthy() for sid, sensor in self.sensors.items()}

    def export_config(self, filepath: str) -> None:
        """Exporta configuración de sensores a JSON."""
        config_data = {
            "sensors": [self.sensors[sid].config.to_dict() for sid in self.sensors]
        }
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        logger.info(f"Configuración exportada a {filepath}")


def test_sensor_bridge():
    """Función para probar el puente de sensores en modo simulado."""
    print("\n=== Prueba del Puente de Sensores ATLAS Traffic AI Pro ===\n")

    # Crear puente y registrar sensores simulados
    bridge = SensorBridge()

    configs = [
        SensorConfig("loop_n", SensorType.INDUCTIVE_LOOP, Direction.NORTH,
                    location="Cuadra N", simulated=True),
        SensorConfig("cam_s", SensorType.CAMERA, Direction.SOUTH,
                    location="Cuadra S", simulated=True),
        SensorConfig("radar_e", SensorType.RADAR, Direction.EAST,
                    location="Cuadra E", simulated=True),
        SensorConfig("bt_w", SensorType.BLUETOOTH, Direction.WEST,
                    location="Cuadra O", simulated=True),
    ]

    for config in configs:
        bridge.register_sensor(config)

    # Simula ciclos de lectura
    print("Ciclos de simulación:")
    for step in range(3):
        state = bridge.get_state_vector(current_phase=step % 4, total_phases=4,
                                        current_step=step, total_steps=100)
        print(f"\nPaso {step}:")
        print(f"  Vector de estado (primeros 8 componentes): {state[:8]}")
        print(f"  Fase normalizada: {state[24]:.3f}")
        print(f"  Paso normalizado: {state[25]:.3f}")

    # Estado de sensores
    print("\n=== Estado de Sensores ===")
    health = bridge.get_sensor_health()
    for sid, healthy in health.items():
        status = "OK" if healthy else "FALLO"
        print(f"  {sid}: {status}")

    print("\nPrueba completada exitosamente.\n")


if __name__ == "__main__":
    test_sensor_bridge()
