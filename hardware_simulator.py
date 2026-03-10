"""
Simulador de Controlador de Tráfico Virtual para ATLAS
======================================================
Módulo que simula un controlador de tráfico real para pruebas sin hardware físico.
Proporciona la misma interfaz que los adaptadores NTCIP y UTMC.

Características:
- Simulación de fase de semáforo con tiempos realistas
- Generador virtual de vehículos con patrones configurables
- Simulación de detectores (bucles inductivos)
- Inyección de vehículos de emergencia
- Ruido de sensores y fallos simulados
"""

import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficPattern(Enum):
    """Patrones de tráfico predefinidos para diferentes escenarios."""
    MORNING_RUSH = "morning_rush"      # 7-9am: tráfico N-S pesado
    EVENING_RUSH = "evening_rush"      # 5-7pm: tráfico E-W pesado
    NIGHT = "night"                     # Bajo volumen todas direcciones
    EVENT = "event"                     # Volumen masivo en una dirección
    BALANCED = "balanced"               # Igual en todas direcciones
    CUSTOM = "custom"                   # Definido por usuario


@dataclass
class ArrivalRates:
    """Tasas de llegada de vehículos por aproximación (vehículos/minuto)."""
    north: float
    south: float
    east: float
    west: float


@dataclass
class PhaseConfig:
    """Configuración de fase de semáforo."""
    green_time: float          # Tiempo en verde (segundos)
    yellow_time: float = 3.0   # Tiempo en amarillo (segundos)
    all_red_time: float = 1.0  # Tiempo en rojo total (segundos)


class VirtualDetector:
    """Simula un detector de bucle inductivo con ruido realista."""

    def __init__(self, detection_range: float = 30.0, noise_level: float = 0.05):
        """
        Inicializa detector virtual.

        Args:
            detection_range: Rango de detección en metros
            noise_level: Nivel de ruido (0.0-1.0)
        """
        self.detection_range = detection_range
        self.noise_level = noise_level
        self.vehicles_count = 0
        self.occupancy = 0.0  # 0.0-1.0
        self.speed = 0.0      # km/h
        self.failures = False
        self.failure_rate = 0.01

    def update(self, vehicles_in_range: int, avg_speed: float, max_occupancy: float = 1.0):
        """
        Actualiza las mediciones del detector.

        Args:
            vehicles_in_range: Número de vehículos detectados
            avg_speed: Velocidad promedio (km/h)
            max_occupancy: Ocupancia máxima del detector
        """
        # Simular fallo aleatorio
        if np.random.random() < self.failure_rate:
            self.failures = True
            return

        self.failures = False

        # Añadir ruido gaussiano
        noise = np.random.normal(0, self.noise_level)
        self.vehicles_count = max(0, int(vehicles_in_range + noise))
        self.occupancy = np.clip(max_occupancy + noise, 0.0, 1.0)
        self.speed = max(0, avg_speed + np.random.normal(0, 2.0))

    def get_reading(self) -> Tuple[int, float, float]:
        """Retorna (conteo, ocupancia, velocidad) con posibles fallos."""
        if self.failures:
            return 0, 0.0, 0.0
        return self.vehicles_count, self.occupancy, self.speed


class VirtualQueue:
    """Simula una cola de vehículos en una aproximación."""

    def __init__(self, approach_name: str, arrival_rate: float):
        """
        Inicializa cola virtual.

        Args:
            approach_name: Nombre de aproximación (N, S, E, W)
            arrival_rate: Tasa de llegada (vehículos/minuto)
        """
        self.approach_name = approach_name
        self.arrival_rate = arrival_rate  # veh/min
        self.queue_length = 0
        self.departed_count = 0
        self.arrival_count = 0
        self.avg_speed = 30.0  # km/h inicial

    def update(self, dt: float, is_green: bool, saturation_flow: float = 1800.0):
        """
        Actualiza la cola basado en el estado del semáforo.

        Args:
            dt: Delta de tiempo (segundos)
            is_green: Verdadero si la fase está en verde
            saturation_flow: Flujo de saturación (veh/hora)
        """
        # Generar llegadas con distribución de Poisson
        arrivals = np.random.poisson(self.arrival_rate * dt / 60.0)
        self.arrival_count += arrivals
        self.queue_length += arrivals

        # Procesamiento de salidas
        if is_green and self.queue_length > 0:
            departures_per_second = saturation_flow / 3600.0
            departures = min(
                self.queue_length,
                max(1, int(departures_per_second * dt))
            )
            self.queue_length -= departures
            self.departed_count += departures
            self.avg_speed = min(60.0, 30.0 + departures * 0.5)  # Velocidad aumenta con flujo
        else:
            # Red: velocidad baja o detenida
            self.avg_speed = max(0.0, self.avg_speed - 2.0)

    def get_occupancy(self, max_queue: float = 50.0) -> float:
        """Retorna ocupancia normalizada (0.0-1.0)."""
        return min(1.0, self.queue_length / max_queue)


class VirtualController:
    """Controlador de tráfico virtual que simula hardware real."""

    def __init__(self, pattern: TrafficPattern = TrafficPattern.BALANCED,
                 seed: int = 42):
        """
        Inicializa controlador virtual.

        Args:
            pattern: Patrón de tráfico a simular
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)

        self.pattern = pattern
        self.current_phase = 0  # 0=NS, 1=EW
        self.phase_time = 0.0
        self.total_time = 0.0
        self.connected = False

        # Configuración de fases
        self.phases = [
            PhaseConfig(green_time=30.0),  # NS fase
            PhaseConfig(green_time=30.0)   # EW fase
        ]

        # Colas virtuales
        self.queues = {
            'N': VirtualQueue('North', 15.0),
            'S': VirtualQueue('South', 15.0),
            'E': VirtualQueue('East', 15.0),
            'W': VirtualQueue('West', 15.0)
        }

        # Detectores
        self.detectors = {
            f'{approach}_{side}': VirtualDetector()
            for approach in ['N', 'S', 'E', 'W']
            for side in ['Advance', 'Stop']
        }

        # Estadísticas
        self.stats = {
            'total_vehicles': 0,
            'emergency_vehicles': 0,
            'avg_delay': 0.0,
            'intersection_occupancy': 0.0
        }

        self._set_pattern_rates()

    def _set_pattern_rates(self):
        """Establece tasas de llegada según el patrón."""
        rates_map = {
            TrafficPattern.MORNING_RUSH: ArrivalRates(40, 35, 10, 12),
            TrafficPattern.EVENING_RUSH: ArrivalRates(10, 12, 40, 35),
            TrafficPattern.NIGHT: ArrivalRates(5, 5, 5, 5),
            TrafficPattern.EVENT: ArrivalRates(60, 60, 10, 10),
            TrafficPattern.BALANCED: ArrivalRates(20, 20, 20, 20),
        }

        rates = rates_map.get(self.pattern, rates_map[TrafficPattern.BALANCED])
        for approach, rate in [('N', rates.north), ('S', rates.south),
                              ('E', rates.east), ('W', rates.west)]:
            self.queues[approach].arrival_rate = rate

    def connect(self) -> bool:
        """Conecta el controlador virtual."""
        logger.info("Controlador virtual conectado")
        self.connected = True
        return True

    def disconnect(self) -> bool:
        """Desconecta el controlador virtual."""
        logger.info("Controlador virtual desconectado")
        self.connected = False
        return True

    def _get_phase_state(self, phase: int) -> str:
        """Retorna estado de fase (green, yellow, red)."""
        if phase != self.current_phase:
            return "red"

        total_phase_time = sum([p.green_time + p.yellow_time + p.all_red_time
                               for p in self.phases])

        green_end = self.phases[phase].green_time
        yellow_end = green_end + self.phases[phase].yellow_time

        if self.phase_time < green_end:
            return "green"
        elif self.phase_time < yellow_end:
            return "yellow"
        else:
            return "red"

    def _is_green(self, phase: int) -> bool:
        """Verifica si una fase está en verde."""
        return phase == self.current_phase and self._get_phase_state(phase) == "green"

    def update(self, dt: float = 1.0):
        """
        Actualiza el estado del controlador.

        Args:
            dt: Delta de tiempo en segundos (típicamente 1.0)
        """
        if not self.connected:
            return

        self.phase_time += dt
        self.total_time += dt

        # Verificar transición de fase
        current_phase_config = self.phases[self.current_phase]
        phase_duration = (current_phase_config.green_time +
                         current_phase_config.yellow_time +
                         current_phase_config.all_red_time)

        if self.phase_time >= phase_duration:
            self.current_phase = 1 - self.current_phase
            self.phase_time = 0.0
            logger.debug(f"Transición a fase {self.current_phase}")

        # Actualizar colas de vehículos
        ns_green = self._is_green(0)
        ew_green = self._is_green(1)

        self.queues['N'].update(dt, ns_green)
        self.queues['S'].update(dt, ns_green)
        self.queues['E'].update(dt, ew_green)
        self.queues['W'].update(dt, ew_green)

        # Actualizar detectores
        for approach in ['N', 'S', 'E', 'W']:
            queue = self.queues[approach]
            self.detectors[f'{approach}_Advance'].update(
                int(queue.queue_length * 0.3),
                queue.avg_speed,
                queue.get_occupancy()
            )
            self.detectors[f'{approach}_Stop'].update(
                int(queue.queue_length * 0.7),
                queue.avg_speed * 0.5,
                queue.get_occupancy()
            )

        # Inyectar vehículos de emergencia aleatoriamente
        if np.random.random() < 0.001:  # ~0.1% por segundo
            self.stats['emergency_vehicles'] += 1
            logger.warning("Vehículo de emergencia inyectado")

        # Actualizar estadísticas
        self.stats['total_vehicles'] = sum(q.arrival_count for q in self.queues.values())
        self.stats['intersection_occupancy'] = np.mean([q.get_occupancy()
                                                        for q in self.queues.values()])

    def read_state(self) -> np.ndarray:
        """
        Retorna vector de estado de 26D compatible con interfaz NTCIP.

        Estado: [fase, tiempo_fase, ocupancia_N, ocupancia_S, ocupancia_E, ocupancia_W,
                 velocidad_N, velocidad_S, velocidad_E, velocidad_W,
                 cola_N, cola_S, cola_E, cola_W,
                 detector_ns_count, detector_ew_count, detector_occupancy,
                 phase_0_state, phase_1_state, emergency_flag,
                 total_vehicles, avg_delay, intersection_occ, time_elapsed,
                 pattern_id, controller_status]
        """
        state = np.zeros(26, dtype=np.float32)

        # Índices 0-4: información de fase
        state[0] = float(self.current_phase)
        state[1] = self.phase_time
        state[2] = float(self._get_phase_state(0) == "green")
        state[3] = float(self._get_phase_state(1) == "green")
        state[4] = float(self.connected)

        # Índices 5-8: ocupancia por aproximación
        state[5] = self.queues['N'].get_occupancy()
        state[6] = self.queues['S'].get_occupancy()
        state[7] = self.queues['E'].get_occupancy()
        state[8] = self.queues['W'].get_occupancy()

        # Índices 9-12: velocidades por aproximación
        state[9] = self.queues['N'].avg_speed
        state[10] = self.queues['S'].avg_speed
        state[11] = self.queues['E'].avg_speed
        state[12] = self.queues['W'].avg_speed

        # Índices 13-16: longitud de colas
        state[13] = float(self.queues['N'].queue_length)
        state[14] = float(self.queues['S'].queue_length)
        state[15] = float(self.queues['E'].queue_length)
        state[16] = float(self.queues['W'].queue_length)

        # Índices 17-20: lecturas de detectores (conteos)
        ns_count = (self.detectors['N_Advance'].vehicles_count +
                   self.detectors['N_Stop'].vehicles_count +
                   self.detectors['S_Advance'].vehicles_count +
                   self.detectors['S_Stop'].vehicles_count)
        ew_count = (self.detectors['E_Advance'].vehicles_count +
                   self.detectors['E_Stop'].vehicles_count +
                   self.detectors['W_Advance'].vehicles_count +
                   self.detectors['W_Stop'].vehicles_count)

        state[17] = float(ns_count)
        state[18] = float(ew_count)
        state[19] = self.stats['intersection_occupancy']
        state[20] = float(self.stats['emergency_vehicles'])

        # Índices 21-25: estadísticas y metadatos
        state[21] = float(self.stats['total_vehicles'])
        state[22] = self.stats['avg_delay']
        state[23] = self.total_time
        state[24] = float(self.pattern.value == "custom")
        state[25] = float(self.connected)

        return state

    def apply_action(self, green_time_ns: float, green_time_ew: float) -> bool:
        """
        Aplica acción de control (tiempos verdes).

        Args:
            green_time_ns: Tiempo verde para fase N-S (segundos)
            green_time_ew: Tiempo verde para fase E-W (segundos)

        Returns:
            Verdadero si la acción fue aplicada exitosamente
        """
        if not self.connected:
            return False

        # Validar y aplicar tiempos
        self.phases[0].green_time = np.clip(green_time_ns, 10.0, 120.0)
        self.phases[1].green_time = np.clip(green_time_ew, 10.0, 120.0)

        logger.debug(f"Acción aplicada: NS={self.phases[0].green_time}s, EW={self.phases[1].green_time}s")
        return True


class HardwareSimulator:
    """Envoltorio que proporciona interfaz unificada para simulación de hardware."""

    def __init__(self, pattern: TrafficPattern = TrafficPattern.BALANCED,
                 simulation_speed: float = 1.0, seed: int = 42):
        """
        Inicializa simulador de hardware.

        Args:
            pattern: Patrón de tráfico
            simulation_speed: Velocidad de simulación (1.0 = tiempo real, 10.0 = 10x)
            seed: Semilla aleatoria
        """
        self.controller = VirtualController(pattern, seed)
        self.simulation_speed = simulation_speed
        self.start_time = time.time()
        self.simulation_time = 0.0
        self.is_running = False

    def connect(self) -> bool:
        """Conecta el simulador."""
        return self.controller.connect()

    def disconnect(self) -> bool:
        """Desconecta el simulador."""
        return self.controller.disconnect()

    def step(self, dt: float = 1.0):
        """
        Avanza la simulación un paso.

        Args:
            dt: Delta de tiempo simulado en segundos
        """
        self.controller.update(dt)
        self.simulation_time += dt

    def read_state(self) -> np.ndarray:
        """Retorna vector de estado actual."""
        return self.controller.read_state()

    def apply_action(self, green_time_ns: float, green_time_ew: float) -> bool:
        """Aplica acción de control."""
        return self.controller.apply_action(green_time_ns, green_time_ew)

    def run_scenario(self, duration: int = 3600):
        """
        Ejecuta escenario de simulación.

        Args:
            duration: Duración de simulación en segundos
        """
        self.is_running = True
        logger.info(f"Iniciando simulación de {duration}s a velocidad {self.simulation_speed}x")

        try:
            while self.simulation_time < duration and self.is_running:
                # Calcular dt basado en velocidad de simulación
                dt = 1.0 * self.simulation_speed
                self.step(dt)

                # Aplicar acción de control ejemplo (round-robin)
                ns_time = 30 + 10 * np.sin(self.simulation_time / 100)
                ew_time = 30 + 10 * np.cos(self.simulation_time / 100)
                self.apply_action(ns_time, ew_time)

                if int(self.simulation_time) % 60 == 0:
                    state = self.read_state()
                    logger.info(f"t={self.simulation_time:.0f}s - "
                              f"Occ={state[19]:.2f} Veh={int(state[21])} "
                              f"EVeh={int(state[20])}")

        except KeyboardInterrupt:
            logger.info("Simulación interrumpida por usuario")
        finally:
            self.is_running = False
            logger.info(f"Simulación completada. Tiempo total: {self.simulation_time}s")


def main():
    """Punto de entrada CLI del simulador de hardware."""
    parser = argparse.ArgumentParser(
        description='Simulador de Controlador de Tráfico para ATLAS'
    )
    parser.add_argument('--pattern',
                       choices=[p.value for p in TrafficPattern],
                       default='balanced',
                       help='Patrón de tráfico a simular')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Velocidad de simulación (ej: 10.0 para 10x)')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Duración de simulación en segundos')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria para reproducibilidad')

    args = parser.parse_args()

    # Convertir string a TrafficPattern
    pattern = TrafficPattern(args.pattern)

    # Crear y ejecutar simulador
    simulator = HardwareSimulator(
        pattern=pattern,
        simulation_speed=args.speed,
        seed=args.seed
    )

    simulator.connect()
    simulator.run_scenario(duration=args.duration)
    simulator.disconnect()


if __name__ == '__main__':
    main()
