"""
ATLAS Pro - Onda Verde Adaptativa (Adaptive Green Wave)
=======================================================
Sistema avanzado de coordinación de onda verde que se adapta
en tiempo real a las condiciones del tráfico.

Características:
- Onda verde bidireccional (MAXBAND algorithm)
- Adaptación en tiempo real con RL feedback
- Soporte multi-corredor y topología grid
- Integración con QMIX multi-intersección
- Optimización de bandwidth con programación lineal
- Predicción de demanda para ajuste proactivo
- Métricas de eficiencia y reporting

Refs:
- Little (1966): MAXBAND - maximizing bandwidth
- Gartner et al. (1991): MULTIBAND - variable bandwidth
- Cesme & Furth (2014): Self-organizing green wave
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum

logger = logging.getLogger("ATLAS.OndaVerde")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class WaveDirection(Enum):
    """Dirección de la onda verde"""
    INBOUND = "inbound"       # Hacia centro ciudad
    OUTBOUND = "outbound"     # Desde centro ciudad
    BIDIRECTIONAL = "bidi"    # Ambas direcciones


@dataclass
class CorridorConfig:
    """Configuración de un corredor de onda verde"""
    corridor_id: str
    intersection_ids: List[str]
    distances_m: List[float]           # Distancias entre intersecciones consecutivas (m)
    speed_limit_kmh: float = 50.0      # Velocidad límite (km/h)
    direction: WaveDirection = WaveDirection.BIDIRECTIONAL
    cycle_time_s: float = 90.0         # Duración del ciclo (s)
    min_green_s: float = 15.0          # Verde mínimo (s)
    max_green_s: float = 60.0          # Verde máximo (s)
    priority: int = 1                  # Prioridad del corredor (1=máxima)

    @property
    def speed_ms(self) -> float:
        return self.speed_limit_kmh / 3.6

    @property
    def n_intersections(self) -> int:
        return len(self.intersection_ids)

    @property
    def total_length_m(self) -> float:
        return sum(self.distances_m)


@dataclass
class WaveState:
    """Estado actual de una onda verde"""
    corridor_id: str
    offsets: List[float]               # Offset de cada intersección (s)
    green_times: List[float]           # Tiempo de verde coordinado por intersección (s)
    bandwidth_inbound: float = 0.0     # Bandwidth dirección entrante (ratio)
    bandwidth_outbound: float = 0.0    # Bandwidth dirección saliente (ratio)
    efficiency: float = 0.0            # Eficiencia global (0-1)
    vehicles_in_wave: int = 0          # Vehículos en la onda
    stops_avoided: int = 0             # Paradas evitadas
    timestamp: float = 0.0
    active: bool = True


@dataclass
class TrafficSnapshot:
    """Snapshot del tráfico en un corredor"""
    flows: List[float]                 # Flujo en cada intersección (veh/h)
    queues: List[int]                  # Cola en cada intersección (veh)
    speeds: List[float]                # Velocidad media en cada tramo (km/h)
    saturation: List[float]            # Grado de saturación (0-1)
    timestamp: float = 0.0


# =============================================================================
# MAXBAND OPTIMIZER
# =============================================================================

class MAXBANDOptimizer:
    """
    Implementación del algoritmo MAXBAND para onda verde bidireccional.

    Maximiza el bandwidth (ancho de banda) en ambas direcciones simultáneamente
    usando optimización. Adaptado para uso con RL.

    MAXBAND maximiza: b_in + b_out
    sujeto a restricciones de offsets y tiempos de verde.
    """

    def __init__(self, config: CorridorConfig):
        self.config = config
        self.travel_times = self._compute_travel_times()

    def _compute_travel_times(self) -> List[float]:
        """Calcula tiempos de viaje entre intersecciones consecutivas"""
        return [d / self.config.speed_ms for d in self.config.distances_m]

    def optimize(self, green_ratios: List[float] = None,
                 weight_inbound: float = 0.5,
                 n_iterations: int = 2000) -> Tuple[List[float], float, float]:
        """
        Optimiza offsets para maximizar bandwidth bidireccional.

        Args:
            green_ratios: Ratio de verde para cada intersección (0-1).
                         Si None, usa 0.5 para todas.
            weight_inbound: Peso para dirección inbound (0-1).
                           outbound = 1 - weight_inbound
            n_iterations: Iteraciones de optimización

        Returns:
            (offsets, bandwidth_inbound, bandwidth_outbound)
        """
        n = self.config.n_intersections
        C = self.config.cycle_time_s

        if green_ratios is None:
            green_ratios = [0.5] * n

        green_times = [r * C for r in green_ratios]

        # Inicializar con offsets basados en travel time
        best_offsets = self._initial_offsets(C)
        best_bw_in = self._bandwidth_direction(best_offsets, green_times, C, "inbound")
        best_bw_out = self._bandwidth_direction(best_offsets, green_times, C, "outbound")
        best_score = weight_inbound * best_bw_in + (1 - weight_inbound) * best_bw_out

        # Optimización por simulated annealing
        temperature = 10.0
        cooling = 0.995
        offsets = best_offsets.copy()

        for iteration in range(n_iterations):
            # Perturbar un offset aleatorio (excepto el primero)
            idx = np.random.randint(1, n)
            perturbation = np.random.normal(0, temperature)
            new_offsets = offsets.copy()
            new_offsets[idx] = (offsets[idx] + perturbation) % C

            bw_in = self._bandwidth_direction(new_offsets, green_times, C, "inbound")
            bw_out = self._bandwidth_direction(new_offsets, green_times, C, "outbound")
            score = weight_inbound * bw_in + (1 - weight_inbound) * bw_out

            # Criterio de aceptación (simulated annealing)
            delta = score - best_score
            if delta > 0 or np.random.random() < np.exp(delta / max(temperature, 0.01)):
                offsets = new_offsets
                if score > best_score:
                    best_offsets = new_offsets.copy()
                    best_bw_in = bw_in
                    best_bw_out = bw_out
                    best_score = score

            temperature *= cooling

        return best_offsets, best_bw_in, best_bw_out

    def _initial_offsets(self, cycle_time: float) -> List[float]:
        """Offsets iniciales basados en tiempos de viaje"""
        offsets = [0.0]
        cumulative = 0.0
        for tt in self.travel_times:
            cumulative += tt
            offsets.append(cumulative % cycle_time)
        return offsets

    def _bandwidth_direction(self, offsets: List[float], green_times: List[float],
                            cycle_time: float, direction: str) -> float:
        """
        Calcula bandwidth en una dirección.

        El bandwidth es la ventana de tiempo máxima continua en la que un vehículo
        puede recorrer todo el corredor sin detenerse.
        """
        n = len(offsets)
        if n < 2:
            return 1.0

        travel_times = self.travel_times
        if direction == "outbound":
            travel_times = list(reversed(travel_times))
            offsets = list(reversed(offsets))
            green_times = list(reversed(green_times))

        # Para cada par consecutivo, calcular la ventana de paso
        min_bandwidth = float('inf')

        for i in range(n - 1):
            tt = travel_times[i]
            green_start_next = offsets[i + 1]
            green_end_next = (offsets[i + 1] + green_times[i + 1]) % cycle_time

            # Tiempo de llegada al siguiente semáforo
            arrival = (offsets[i] + tt) % cycle_time

            # Calcular overlap entre ventana de llegada y verde
            overlap = self._green_overlap(
                arrival, arrival + green_times[i],
                green_start_next, green_end_next,
                cycle_time
            )
            min_bandwidth = min(min_bandwidth, overlap)

        return max(0, min_bandwidth / cycle_time)

    def _green_overlap(self, arr_start: float, arr_end: float,
                       green_start: float, green_end: float,
                       cycle: float) -> float:
        """Calcula overlap entre ventana de llegada y fase verde (modular)"""
        # Normalizar al ciclo
        arr_start = arr_start % cycle
        arr_end = arr_end % cycle
        green_start = green_start % cycle
        green_end = green_end % cycle

        # Caso simple (sin wrap-around)
        if arr_start <= arr_end and green_start <= green_end:
            overlap_start = max(arr_start, green_start)
            overlap_end = min(arr_end, green_end)
            return max(0, overlap_end - overlap_start)

        # Caso con wrap-around: probar múltiples alineaciones
        best = 0
        for shift in [0, cycle, -cycle]:
            os = max(arr_start, green_start + shift)
            oe = min(arr_end, green_end + shift)
            best = max(best, oe - os)
        return max(0, best)


# =============================================================================
# ADAPTIVE GREEN WAVE CONTROLLER
# =============================================================================

class AdaptiveGreenWave:
    """
    Controlador de onda verde adaptativa.

    Ajusta los offsets y tiempos de verde en tiempo real basándose en:
    - Flujo de tráfico actual (sensores/simulación)
    - Predicción de demanda (histórico + tendencia)
    - Feedback del sistema QMIX
    - Condiciones especiales (emergencias, eventos, clima)

    Modes:
    - FIXED: Offsets fijos optimizados offline
    - REACTIVE: Ajusta offsets cuando detecta cambio significativo
    - PROACTIVE: Predice demanda y pre-ajusta offsets
    - RL_INTEGRATED: Los offsets son sugerencias del agente QMIX
    """

    class Mode(Enum):
        FIXED = "fixed"
        REACTIVE = "reactive"
        PROACTIVE = "proactive"
        RL_INTEGRATED = "rl_integrated"

    def __init__(self, corridors: List[CorridorConfig],
                 mode: str = "reactive",
                 update_interval_s: float = 300.0,
                 history_window: int = 24):
        """
        Args:
            corridors: Lista de configuraciones de corredores
            mode: Modo de operación
            update_interval_s: Intervalo de actualización (segundos)
            history_window: Ventanas de histórico para predicción (horas)
        """
        self.corridors = {c.corridor_id: c for c in corridors}
        self.mode = self.Mode(mode)
        self.update_interval_s = update_interval_s
        self.history_window = history_window

        # Estado por corredor
        self.wave_states: Dict[str, WaveState] = {}
        self.optimizers: Dict[str, MAXBANDOptimizer] = {}
        self.traffic_history: Dict[str, deque] = {}

        # Métricas acumuladas
        self.metrics = {
            'total_updates': 0,
            'total_vehicles_in_wave': 0,
            'total_stops_avoided': 0,
            'avg_bandwidth': 0.0,
            'corridors_active': 0
        }

        # Inicializar
        for cid, config in self.corridors.items():
            self.optimizers[cid] = MAXBANDOptimizer(config)
            self.traffic_history[cid] = deque(maxlen=history_window * 12)  # cada 5 min
            self._initialize_corridor(cid)

        logger.info(f"AdaptiveGreenWave inicializado: {len(corridors)} corredores, modo={mode}")

    def _initialize_corridor(self, corridor_id: str):
        """Inicializa onda verde para un corredor con optimización offline"""
        config = self.corridors[corridor_id]
        optimizer = self.optimizers[corridor_id]

        # Optimización bidireccional inicial
        offsets, bw_in, bw_out = optimizer.optimize(
            weight_inbound=0.5,
            n_iterations=3000
        )

        green_times = [config.cycle_time_s * 0.5] * config.n_intersections

        self.wave_states[corridor_id] = WaveState(
            corridor_id=corridor_id,
            offsets=offsets,
            green_times=green_times,
            bandwidth_inbound=bw_in,
            bandwidth_outbound=bw_out,
            efficiency=0.5 * (bw_in + bw_out),
            timestamp=time.time(),
            active=True
        )

        logger.info(
            f"Corredor {corridor_id}: BW_in={bw_in:.3f}, BW_out={bw_out:.3f}, "
            f"offsets={[f'{o:.1f}' for o in offsets]}"
        )

    # =========================================================================
    # CORE UPDATE LOOP
    # =========================================================================

    def update(self, corridor_id: str, traffic: TrafficSnapshot,
               qmix_suggestion: Dict = None) -> WaveState:
        """
        Actualiza la onda verde basándose en tráfico actual.

        Args:
            corridor_id: ID del corredor
            traffic: Snapshot actual del tráfico
            qmix_suggestion: Sugerencia del sistema QMIX (opcional)
                             {'offsets': [...], 'green_ratios': [...]}

        Returns:
            WaveState actualizado
        """
        if corridor_id not in self.corridors:
            raise ValueError(f"Corredor {corridor_id} no registrado")

        # Almacenar histórico
        self.traffic_history[corridor_id].append(traffic)

        config = self.corridors[corridor_id]
        current_state = self.wave_states[corridor_id]

        if self.mode == self.Mode.FIXED:
            # No actualizar, mantener offsets fijos
            current_state.timestamp = time.time()
            return current_state

        elif self.mode == self.Mode.RL_INTEGRATED and qmix_suggestion:
            # Usar sugerencias del QMIX directamente
            return self._apply_rl_suggestion(corridor_id, qmix_suggestion, traffic)

        elif self.mode == self.Mode.PROACTIVE:
            # Predecir y pre-ajustar
            return self._proactive_update(corridor_id, traffic)

        else:
            # REACTIVE: ajustar si cambio significativo
            return self._reactive_update(corridor_id, traffic)

    def _reactive_update(self, corridor_id: str, traffic: TrafficSnapshot) -> WaveState:
        """Actualización reactiva: reoptimiza si el tráfico cambió significativamente"""
        config = self.corridors[corridor_id]
        current = self.wave_states[corridor_id]

        # Detectar cambio significativo
        if not self._significant_change(corridor_id, traffic):
            current.timestamp = time.time()
            self._update_wave_metrics(corridor_id, traffic)
            return current

        logger.info(f"Corredor {corridor_id}: cambio significativo detectado, reoptimizando")

        # Calcular peso direccional basado en flujo
        weight_in = self._compute_directional_weight(traffic)

        # Calcular green ratios adaptativos
        green_ratios = self._compute_adaptive_green_ratios(corridor_id, traffic)

        # Reoptimizar
        optimizer = self.optimizers[corridor_id]
        offsets, bw_in, bw_out = optimizer.optimize(
            green_ratios=green_ratios,
            weight_inbound=weight_in,
            n_iterations=1500
        )

        green_times = [r * config.cycle_time_s for r in green_ratios]

        new_state = WaveState(
            corridor_id=corridor_id,
            offsets=offsets,
            green_times=green_times,
            bandwidth_inbound=bw_in,
            bandwidth_outbound=bw_out,
            efficiency=weight_in * bw_in + (1 - weight_in) * bw_out,
            timestamp=time.time(),
            active=True
        )

        self._update_wave_metrics(corridor_id, traffic)
        self.wave_states[corridor_id] = new_state
        self.metrics['total_updates'] += 1

        return new_state

    def _proactive_update(self, corridor_id: str, traffic: TrafficSnapshot) -> WaveState:
        """Actualización proactiva: predice demanda futura y pre-ajusta"""
        config = self.corridors[corridor_id]
        history = list(self.traffic_history[corridor_id])

        # Predecir flujo futuro (media móvil ponderada + tendencia)
        predicted_flows = self._predict_flows(history, traffic)
        predicted_traffic = TrafficSnapshot(
            flows=predicted_flows,
            queues=traffic.queues,
            speeds=traffic.speeds,
            saturation=[min(1.0, f / 1800) for f in predicted_flows],
            timestamp=time.time()
        )

        # Usar tráfico predicho para optimizar
        weight_in = self._compute_directional_weight(predicted_traffic)
        green_ratios = self._compute_adaptive_green_ratios(corridor_id, predicted_traffic)

        optimizer = self.optimizers[corridor_id]
        offsets, bw_in, bw_out = optimizer.optimize(
            green_ratios=green_ratios,
            weight_inbound=weight_in,
            n_iterations=2000
        )

        green_times = [r * config.cycle_time_s for r in green_ratios]

        new_state = WaveState(
            corridor_id=corridor_id,
            offsets=offsets,
            green_times=green_times,
            bandwidth_inbound=bw_in,
            bandwidth_outbound=bw_out,
            efficiency=weight_in * bw_in + (1 - weight_in) * bw_out,
            timestamp=time.time(),
            active=True
        )

        self._update_wave_metrics(corridor_id, traffic)
        self.wave_states[corridor_id] = new_state
        self.metrics['total_updates'] += 1

        return new_state

    def _apply_rl_suggestion(self, corridor_id: str,
                             suggestion: Dict,
                             traffic: TrafficSnapshot) -> WaveState:
        """Aplica sugerencia del sistema QMIX con validación de seguridad"""
        config = self.corridors[corridor_id]

        offsets = suggestion.get('offsets', self.wave_states[corridor_id].offsets)
        green_ratios = suggestion.get('green_ratios', [0.5] * config.n_intersections)

        # Validación de seguridad: verde mínimo/máximo
        validated_greens = []
        for ratio in green_ratios:
            green_s = ratio * config.cycle_time_s
            green_s = max(config.min_green_s, min(config.max_green_s, green_s))
            validated_greens.append(green_s)

        # Validar offsets dentro del ciclo
        validated_offsets = [0.0] + [
            o % config.cycle_time_s for o in offsets[1:]
        ]

        # Calcular bandwidth resultante
        optimizer = self.optimizers[corridor_id]
        green_ratios_validated = [g / config.cycle_time_s for g in validated_greens]
        bw_in = optimizer._bandwidth_direction(
            validated_offsets, validated_greens, config.cycle_time_s, "inbound"
        )
        bw_out = optimizer._bandwidth_direction(
            validated_offsets, validated_greens, config.cycle_time_s, "outbound"
        )

        new_state = WaveState(
            corridor_id=corridor_id,
            offsets=validated_offsets,
            green_times=validated_greens,
            bandwidth_inbound=bw_in,
            bandwidth_outbound=bw_out,
            efficiency=0.5 * (bw_in + bw_out),
            timestamp=time.time(),
            active=True
        )

        self._update_wave_metrics(corridor_id, traffic)
        self.wave_states[corridor_id] = new_state
        self.metrics['total_updates'] += 1

        return new_state

    # =========================================================================
    # ADAPTIVE COMPUTATIONS
    # =========================================================================

    def _significant_change(self, corridor_id: str, traffic: TrafficSnapshot,
                           threshold: float = 0.2) -> bool:
        """Detecta si el tráfico cambió significativamente"""
        history = self.traffic_history[corridor_id]
        if len(history) < 3:
            return True

        # Comparar flujo actual con media de últimos 3 snapshots
        recent_flows = [h.flows for h in list(history)[-3:]]
        avg_flows = np.mean(recent_flows, axis=0)

        if np.mean(avg_flows) == 0:
            return True

        change = np.mean(np.abs(np.array(traffic.flows) - avg_flows) / (avg_flows + 1e-6))
        return change > threshold

    def _compute_directional_weight(self, traffic: TrafficSnapshot) -> float:
        """
        Calcula peso para dirección inbound vs outbound.

        Si el flujo es mayor en la primera mitad del corredor (entrando),
        prioriza inbound. Si es mayor en la segunda mitad, prioriza outbound.
        """
        n = len(traffic.flows)
        if n < 2:
            return 0.5

        first_half = np.mean(traffic.flows[:n // 2])
        second_half = np.mean(traffic.flows[n // 2:])
        total = first_half + second_half

        if total == 0:
            return 0.5

        # Más flujo entrando -> más peso inbound
        weight = first_half / total
        # Suavizar para evitar extremos
        return 0.3 + 0.4 * weight  # Rango [0.3, 0.7]

    def _compute_adaptive_green_ratios(self, corridor_id: str,
                                        traffic: TrafficSnapshot) -> List[float]:
        """
        Calcula ratios de verde adaptativos basados en demanda.

        Intersecciones con más tráfico reciben más verde.
        """
        config = self.corridors[corridor_id]

        if not traffic.flows or all(f == 0 for f in traffic.flows):
            return [0.5] * config.n_intersections

        total_flow = sum(traffic.flows)
        n = config.n_intersections

        ratios = []
        for i in range(n):
            if i < len(traffic.flows):
                demand_ratio = traffic.flows[i] / (total_flow / n) if total_flow > 0 else 1.0
                # Escalar entre 0.35 y 0.65
                ratio = 0.35 + 0.30 * min(2.0, demand_ratio) / 2.0
            else:
                ratio = 0.5

            # Aplicar límites
            min_ratio = config.min_green_s / config.cycle_time_s
            max_ratio = config.max_green_s / config.cycle_time_s
            ratios.append(max(min_ratio, min(max_ratio, ratio)))

        return ratios

    def _predict_flows(self, history: List[TrafficSnapshot],
                      current: TrafficSnapshot) -> List[float]:
        """
        Predice flujos futuros usando media móvil exponencial + tendencia.

        Usa los últimos snapshots para detectar tendencia y extrapolar.
        """
        if len(history) < 5:
            return current.flows

        n_points = min(12, len(history))
        recent = [h.flows for h in list(history)[-n_points:]]

        # EMA (Exponential Moving Average)
        alpha = 0.3
        ema = np.array(recent[0], dtype=float)
        for flows in recent[1:]:
            ema = alpha * np.array(flows) + (1 - alpha) * ema

        # Tendencia lineal
        if n_points >= 6:
            first_half = np.mean(recent[:n_points // 2], axis=0)
            second_half = np.mean(recent[n_points // 2:], axis=0)
            trend = second_half - first_half
            predicted = ema + 0.5 * trend  # Extrapolar media tendencia
        else:
            predicted = ema

        # Clamp a valores razonables
        predicted = np.maximum(0, predicted)
        return predicted.tolist()

    def _update_wave_metrics(self, corridor_id: str, traffic: TrafficSnapshot):
        """Actualiza métricas de la onda verde"""
        state = self.wave_states[corridor_id]

        # Estimar vehículos en la onda (flujo * bandwidth * ciclo / 3600)
        config = self.corridors[corridor_id]
        avg_flow = np.mean(traffic.flows) if traffic.flows else 0
        bw_avg = (state.bandwidth_inbound + state.bandwidth_outbound) / 2
        vehicles = int(avg_flow * bw_avg * config.cycle_time_s / 3600)
        state.vehicles_in_wave = vehicles

        # Estimar paradas evitadas
        stops = int(vehicles * bw_avg * 0.7)  # 70% de vehículos en onda no paran
        state.stops_avoided = stops

        self.metrics['total_vehicles_in_wave'] += vehicles
        self.metrics['total_stops_avoided'] += stops

    # =========================================================================
    # MULTI-CORRIDOR COORDINATION
    # =========================================================================

    def coordinate_corridors(self, traffic_data: Dict[str, TrafficSnapshot]) -> Dict[str, WaveState]:
        """
        Coordina múltiples corredores simultáneamente.

        Cuando dos corredores comparten intersecciones, resuelve conflictos
        priorizando el corredor con mayor flujo/prioridad.

        Args:
            traffic_data: {corridor_id: TrafficSnapshot}

        Returns:
            {corridor_id: WaveState}
        """
        results = {}

        # Ordenar por prioridad
        sorted_corridors = sorted(
            traffic_data.keys(),
            key=lambda cid: self.corridors[cid].priority
        )

        # Intersecciones ya asignadas (para resolver conflictos)
        locked_offsets: Dict[str, float] = {}

        for cid in sorted_corridors:
            if cid not in self.corridors:
                continue

            traffic = traffic_data[cid]
            config = self.corridors[cid]

            # Verificar conflictos con intersecciones ya bloqueadas
            constraints = {}
            for i, iid in enumerate(config.intersection_ids):
                if iid in locked_offsets:
                    constraints[i] = locked_offsets[iid]

            if constraints:
                # Optimizar con restricciones
                state = self._constrained_update(cid, traffic, constraints)
            else:
                # Optimización libre
                state = self.update(cid, traffic)

            results[cid] = state

            # Bloquear offsets de este corredor
            for i, iid in enumerate(config.intersection_ids):
                if iid not in locked_offsets:
                    locked_offsets[iid] = state.offsets[i]

        # Actualizar conteo de corredores activos
        self.metrics['corridors_active'] = sum(1 for s in results.values() if s.active)

        return results

    def _constrained_update(self, corridor_id: str,
                           traffic: TrafficSnapshot,
                           constraints: Dict[int, float]) -> WaveState:
        """
        Optimiza corredor con offsets fijos en ciertas intersecciones.

        Args:
            constraints: {intersection_index: fixed_offset}
        """
        config = self.corridors[corridor_id]
        optimizer = self.optimizers[corridor_id]

        weight_in = self._compute_directional_weight(traffic)
        green_ratios = self._compute_adaptive_green_ratios(corridor_id, traffic)
        C = config.cycle_time_s

        # Optimizar solo offsets libres
        best_offsets = self.wave_states[corridor_id].offsets.copy()
        for idx, offset in constraints.items():
            best_offsets[idx] = offset

        best_bw_in = optimizer._bandwidth_direction(
            best_offsets, [r * C for r in green_ratios], C, "inbound"
        )
        best_bw_out = optimizer._bandwidth_direction(
            best_offsets, [r * C for r in green_ratios], C, "outbound"
        )
        best_score = weight_in * best_bw_in + (1 - weight_in) * best_bw_out

        for _ in range(1000):
            new_offsets = best_offsets.copy()
            # Solo perturbar offsets libres
            free_indices = [i for i in range(1, config.n_intersections) if i not in constraints]
            if not free_indices:
                break
            idx = np.random.choice(free_indices)
            new_offsets[idx] = (new_offsets[idx] + np.random.normal(0, 3)) % C

            bw_in = optimizer._bandwidth_direction(
                new_offsets, [r * C for r in green_ratios], C, "inbound"
            )
            bw_out = optimizer._bandwidth_direction(
                new_offsets, [r * C for r in green_ratios], C, "outbound"
            )
            score = weight_in * bw_in + (1 - weight_in) * bw_out

            if score > best_score:
                best_offsets = new_offsets
                best_bw_in = bw_in
                best_bw_out = bw_out
                best_score = score

        green_times = [r * C for r in green_ratios]

        new_state = WaveState(
            corridor_id=corridor_id,
            offsets=best_offsets,
            green_times=green_times,
            bandwidth_inbound=best_bw_in,
            bandwidth_outbound=best_bw_out,
            efficiency=best_score,
            timestamp=time.time(),
            active=True
        )

        self.wave_states[corridor_id] = new_state
        return new_state

    # =========================================================================
    # QMIX INTEGRATION
    # =========================================================================

    def get_rl_observation(self, corridor_id: str) -> np.ndarray:
        """
        Genera observación para el agente QMIX incluyendo estado de onda verde.

        Returns:
            Vector con [offsets_norm, green_ratios, bandwidth_in, bandwidth_out,
                        efficiency, vehicles_in_wave_norm]
        """
        state = self.wave_states.get(corridor_id)
        config = self.corridors.get(corridor_id)

        if not state or not config:
            return np.zeros(20)

        C = config.cycle_time_s
        obs = []

        # Offsets normalizados
        obs.extend([o / C for o in state.offsets])

        # Green ratios
        obs.extend([g / C for g in state.green_times])

        # Métricas globales
        obs.append(state.bandwidth_inbound)
        obs.append(state.bandwidth_outbound)
        obs.append(state.efficiency)
        obs.append(min(1.0, state.vehicles_in_wave / 100.0))

        # Pad/truncate a tamaño fijo
        target_size = 20
        if len(obs) > target_size:
            obs = obs[:target_size]
        else:
            obs.extend([0.0] * (target_size - len(obs)))

        return np.array(obs, dtype=np.float32)

    def apply_rl_action(self, corridor_id: str, action: np.ndarray,
                       traffic: TrafficSnapshot) -> WaveState:
        """
        Aplica acción del agente QMIX como ajuste a la onda verde.

        El action vector se interpreta como delta_offsets + delta_green_ratios.
        """
        config = self.corridors[corridor_id]
        n = config.n_intersections
        C = config.cycle_time_s

        current = self.wave_states[corridor_id]

        # Interpretar acción: primera mitad = delta offsets, segunda = delta greens
        delta_offsets = action[:n] * 10.0  # Escala: +/- 10 segundos
        delta_greens = action[n:2*n] * 0.1 if len(action) > n else np.zeros(n)

        # Aplicar deltas
        new_offsets = [0.0]  # Primer offset siempre 0
        for i in range(1, n):
            if i < len(delta_offsets):
                new_offset = (current.offsets[i] + delta_offsets[i]) % C
            else:
                new_offset = current.offsets[i]
            new_offsets.append(new_offset)

        new_green_ratios = []
        for i in range(n):
            current_ratio = current.green_times[i] / C
            if i < len(delta_greens):
                new_ratio = current_ratio + delta_greens[i]
            else:
                new_ratio = current_ratio
            min_ratio = config.min_green_s / C
            max_ratio = config.max_green_s / C
            new_green_ratios.append(max(min_ratio, min(max_ratio, new_ratio)))

        suggestion = {
            'offsets': new_offsets,
            'green_ratios': new_green_ratios
        }

        return self._apply_rl_suggestion(corridor_id, suggestion, traffic)

    # =========================================================================
    # EMERGENCY & SPECIAL MODES
    # =========================================================================

    def emergency_preemption(self, corridor_id: str,
                            intersection_idx: int,
                            direction: str = "inbound") -> WaveState:
        """
        Preempción de emergencia: fuerza verde en un punto del corredor.

        Usado cuando un vehículo de emergencia se acerca. Temporalmente
        rompe la onda verde para dar paso prioritario.
        """
        config = self.corridors[corridor_id]
        state = self.wave_states[corridor_id]

        logger.warning(
            f"EMERGENCIA: Preempción en corredor {corridor_id}, "
            f"intersección {intersection_idx}"
        )

        # Forzar verde máximo en la intersección de emergencia
        new_greens = state.green_times.copy()
        new_greens[intersection_idx] = config.max_green_s

        # Reducir verde en adyacentes para compensar
        for i in range(config.n_intersections):
            if i != intersection_idx and abs(i - intersection_idx) <= 1:
                new_greens[i] = max(config.min_green_s, new_greens[i] * 0.7)

        state.green_times = new_greens
        state.timestamp = time.time()

        return state

    def transit_signal_priority(self, corridor_id: str,
                               bus_position_idx: int,
                               bus_speed_kmh: float = 30.0) -> WaveState:
        """
        Prioridad de transporte público (TSP).

        Extiende o adelanta el verde para que el autobús/tranvía
        no tenga que detenerse. Menos agresivo que preempción de emergencia.
        """
        config = self.corridors[corridor_id]
        state = self.wave_states[corridor_id]

        # Calcular tiempo de llegada del bus a próximas intersecciones
        bus_speed_ms = bus_speed_kmh / 3.6

        for i in range(bus_position_idx, min(bus_position_idx + 3, config.n_intersections)):
            if i > bus_position_idx and i - 1 < len(config.distances_m):
                travel = config.distances_m[i - 1] / bus_speed_ms

                # Ajustar offset para que el bus llegue durante el verde
                desired_offset = (state.offsets[bus_position_idx] + travel) % config.cycle_time_s
                current_offset = state.offsets[i]

                # Solo ajustar si la diferencia es pequeña (< 15s)
                diff = (desired_offset - current_offset) % config.cycle_time_s
                if diff < 15:
                    state.offsets[i] = desired_offset
                    # Extender verde ligeramente
                    state.green_times[i] = min(
                        config.max_green_s,
                        state.green_times[i] + 5.0
                    )

        state.timestamp = time.time()
        logger.info(f"TSP aplicado en corredor {corridor_id}, bus en intersección {bus_position_idx}")

        return state

    # =========================================================================
    # REPORTING & EXPORT
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Estado completo del sistema de onda verde"""
        return {
            'mode': self.mode.value,
            'corridors': {
                cid: {
                    'config': {
                        'n_intersections': c.n_intersections,
                        'total_length_m': c.total_length_m,
                        'speed_limit_kmh': c.speed_limit_kmh,
                        'cycle_time_s': c.cycle_time_s,
                        'direction': c.direction.value
                    },
                    'state': asdict(self.wave_states[cid]) if cid in self.wave_states else None
                }
                for cid, c in self.corridors.items()
            },
            'metrics': self.metrics
        }

    def get_corridor_report(self, corridor_id: str) -> Dict:
        """Reporte detallado de un corredor"""
        state = self.wave_states.get(corridor_id)
        config = self.corridors.get(corridor_id)

        if not state or not config:
            return {'error': f'Corredor {corridor_id} no encontrado'}

        return {
            'corridor_id': corridor_id,
            'intersections': config.intersection_ids,
            'total_length_km': config.total_length_m / 1000,
            'speed_limit_kmh': config.speed_limit_kmh,
            'cycle_time_s': config.cycle_time_s,
            'offsets_s': [round(o, 1) for o in state.offsets],
            'green_times_s': [round(g, 1) for g in state.green_times],
            'bandwidth_inbound': round(state.bandwidth_inbound, 4),
            'bandwidth_outbound': round(state.bandwidth_outbound, 4),
            'efficiency': round(state.efficiency, 4),
            'vehicles_in_wave': state.vehicles_in_wave,
            'stops_avoided': state.stops_avoided,
            'active': state.active,
            'last_update': state.timestamp
        }

    def export_timing_plan(self, corridor_id: str, filepath: str = None) -> Dict:
        """
        Exporta plan de tiempos en formato compatible con controladores.

        Formato compatible con NTCIP 1202 y UTMC.
        """
        state = self.wave_states.get(corridor_id)
        config = self.corridors.get(corridor_id)

        if not state or not config:
            return {}

        plan = {
            'version': '1.0',
            'generator': 'ATLAS Pro - Adaptive Green Wave',
            'corridor': corridor_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'cycle_time_s': config.cycle_time_s,
            'intersections': []
        }

        for i, iid in enumerate(config.intersection_ids):
            plan['intersections'].append({
                'id': iid,
                'offset_s': round(state.offsets[i], 1),
                'phases': [
                    {
                        'phase_id': 1,
                        'description': 'Coordinated Green (Main Street)',
                        'green_s': round(state.green_times[i], 1),
                        'yellow_s': 3.0,
                        'all_red_s': 2.0
                    },
                    {
                        'phase_id': 2,
                        'description': 'Cross Street',
                        'green_s': round(config.cycle_time_s - state.green_times[i] - 5.0, 1),
                        'yellow_s': 3.0,
                        'all_red_s': 2.0
                    }
                ]
            })

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(plan, f, indent=2)
            logger.info(f"Plan exportado a {filepath}")

        return plan


# =============================================================================
# GRID GREEN WAVE (2D)
# =============================================================================

class GridGreenWave:
    """
    Onda verde para topología de rejilla (grid).

    Coordina ondas verdes en dos ejes (horizontal y vertical) simultáneamente.
    Más complejo que corredor lineal: cada intersección pertenece a dos ondas.
    """

    def __init__(self, rows: int, cols: int,
                 block_size_m: float = 200.0,
                 speed_kmh: float = 50.0,
                 cycle_time_s: float = 90.0):
        self.rows = rows
        self.cols = cols
        self.block_size = block_size_m
        self.speed = speed_kmh
        self.cycle_time = cycle_time_s

        # Crear corredores horizontales y verticales
        self.h_corridors: Dict[str, CorridorConfig] = {}
        self.v_corridors: Dict[str, CorridorConfig] = {}

        self._create_corridors()

        # Controlador adaptativo con todos los corredores
        all_corridors = list(self.h_corridors.values()) + list(self.v_corridors.values())
        self.controller = AdaptiveGreenWave(
            corridors=all_corridors,
            mode="reactive",
            update_interval_s=300
        )

    def _create_corridors(self):
        """Crea corredores horizontales y verticales"""
        distances = [self.block_size] * (self.cols - 1)

        # Horizontales (prioridad más alta)
        for r in range(self.rows):
            cid = f"h_{r}"
            ids = [f"int_{r}_{c}" for c in range(self.cols)]
            self.h_corridors[cid] = CorridorConfig(
                corridor_id=cid,
                intersection_ids=ids,
                distances_m=distances.copy(),
                speed_limit_kmh=self.speed,
                direction=WaveDirection.BIDIRECTIONAL,
                cycle_time_s=self.cycle_time,
                priority=1
            )

        # Verticales (prioridad secundaria)
        v_distances = [self.block_size] * (self.rows - 1)
        for c in range(self.cols):
            cid = f"v_{c}"
            ids = [f"int_{r}_{c}" for r in range(self.rows)]
            self.v_corridors[cid] = CorridorConfig(
                corridor_id=cid,
                intersection_ids=ids,
                distances_m=v_distances.copy(),
                speed_limit_kmh=self.speed,
                direction=WaveDirection.BIDIRECTIONAL,
                cycle_time_s=self.cycle_time,
                priority=2
            )

    def update(self, traffic_matrix: np.ndarray) -> Dict[str, WaveState]:
        """
        Actualiza todas las ondas verdes de la grid.

        Args:
            traffic_matrix: [rows, cols] con flujo en cada intersección

        Returns:
            Dict con estado de cada corredor
        """
        traffic_data = {}

        # Crear snapshots para corredores horizontales
        for r in range(self.rows):
            cid = f"h_{r}"
            flows = traffic_matrix[r, :].tolist() if r < traffic_matrix.shape[0] else [500] * self.cols
            traffic_data[cid] = TrafficSnapshot(
                flows=flows,
                queues=[5] * self.cols,
                speeds=[self.speed] * self.cols,
                saturation=[f / 1800 for f in flows],
                timestamp=time.time()
            )

        # Crear snapshots para corredores verticales
        for c in range(self.cols):
            cid = f"v_{c}"
            flows = traffic_matrix[:, c].tolist() if c < traffic_matrix.shape[1] else [500] * self.rows
            traffic_data[cid] = TrafficSnapshot(
                flows=flows,
                queues=[5] * self.rows,
                speeds=[self.speed] * self.rows,
                saturation=[f / 1800 for f in flows],
                timestamp=time.time()
            )

        return self.controller.coordinate_corridors(traffic_data)

    def get_intersection_timing(self, row: int, col: int) -> Dict:
        """Obtiene timing combinado para una intersección específica"""
        h_state = self.controller.wave_states.get(f"h_{row}")
        v_state = self.controller.wave_states.get(f"v_{col}")

        result = {'intersection': f"int_{row}_{col}"}

        if h_state and col < len(h_state.offsets):
            result['h_offset'] = round(h_state.offsets[col], 1)
            result['h_green'] = round(h_state.green_times[col], 1)

        if v_state and row < len(v_state.offsets):
            result['v_offset'] = round(v_state.offsets[row], 1)
            result['v_green'] = round(v_state.green_times[row], 1)

        return result


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_onda_verde():
    """Demo del sistema de onda verde adaptativa"""

    print("=" * 70)
    print("  ATLAS Pro - Demo Onda Verde Adaptativa")
    print("=" * 70)

    # 1. CORREDOR LINEAL (5 intersecciones, 200m entre cada una)
    print("\n--- Corredor Lineal (5 intersecciones) ---")

    corridor = CorridorConfig(
        corridor_id="avenida_principal",
        intersection_ids=["INT_01", "INT_02", "INT_03", "INT_04", "INT_05"],
        distances_m=[200.0, 250.0, 180.0, 220.0],
        speed_limit_kmh=50.0,
        cycle_time_s=90.0,
        priority=1
    )

    controller = AdaptiveGreenWave(
        corridors=[corridor],
        mode="reactive"
    )

    # Simular tráfico
    traffic = TrafficSnapshot(
        flows=[800, 1200, 1500, 900, 600],
        queues=[3, 8, 12, 5, 2],
        speeds=[45, 38, 32, 42, 48],
        saturation=[0.44, 0.67, 0.83, 0.50, 0.33],
        timestamp=time.time()
    )

    state = controller.update("avenida_principal", traffic)

    print(f"  Offsets (s):       {[f'{o:.1f}' for o in state.offsets]}")
    print(f"  Green times (s):   {[f'{g:.1f}' for g in state.green_times]}")
    print(f"  Bandwidth IN:      {state.bandwidth_inbound:.4f}")
    print(f"  Bandwidth OUT:     {state.bandwidth_outbound:.4f}")
    print(f"  Efficiency:        {state.efficiency:.4f}")
    print(f"  Vehicles in wave:  {state.vehicles_in_wave}")

    # Simular cambio de tráfico (hora punta)
    print("\n--- Hora Punta (tráfico aumenta) ---")

    traffic_peak = TrafficSnapshot(
        flows=[1400, 1800, 2000, 1600, 1200],
        queues=[8, 15, 22, 12, 6],
        speeds=[35, 28, 22, 30, 40],
        saturation=[0.78, 1.0, 1.11, 0.89, 0.67],
        timestamp=time.time()
    )

    state2 = controller.update("avenida_principal", traffic_peak)

    print(f"  Offsets (s):       {[f'{o:.1f}' for o in state2.offsets]}")
    print(f"  Green times (s):   {[f'{g:.1f}' for g in state2.green_times]}")
    print(f"  Bandwidth IN:      {state2.bandwidth_inbound:.4f}")
    print(f"  Bandwidth OUT:     {state2.bandwidth_outbound:.4f}")
    print(f"  Efficiency:        {state2.efficiency:.4f}")

    # Exportar plan
    plan = controller.export_timing_plan("avenida_principal")
    print(f"\n  Timing plan exportado: {len(plan['intersections'])} intersecciones")

    # 2. GRID 3x3
    print("\n\n--- Grid 3x3 (onda verde 2D) ---")

    grid = GridGreenWave(rows=3, cols=3, block_size_m=200, speed_kmh=50)

    traffic_matrix = np.array([
        [600, 900, 700],
        [1200, 1500, 1100],
        [800, 1000, 600]
    ], dtype=float)

    results = grid.update(traffic_matrix)

    for cid, ws in results.items():
        direction = "Horizontal" if cid.startswith("h_") else "Vertical"
        print(f"  {direction} {cid}: BW_in={ws.bandwidth_inbound:.3f}, "
              f"BW_out={ws.bandwidth_outbound:.3f}, Eff={ws.efficiency:.3f}")

    # Timing de intersección central
    center = grid.get_intersection_timing(1, 1)
    print(f"\n  Intersección central (1,1):")
    print(f"    H offset={center.get('h_offset')}s, green={center.get('h_green')}s")
    print(f"    V offset={center.get('v_offset')}s, green={center.get('v_green')}s")

    # 3. TSP (Transit Signal Priority)
    print("\n\n--- Transit Signal Priority (Bus) ---")

    state_tsp = controller.transit_signal_priority(
        "avenida_principal",
        bus_position_idx=1,
        bus_speed_kmh=30
    )
    print(f"  TSP aplicado: offsets ajustados para bus en INT_02")
    print(f"  Nuevos offsets: {[f'{o:.1f}' for o in state_tsp.offsets]}")

    # Status general
    print("\n\n--- Status General ---")
    status = controller.get_status()
    print(f"  Modo:              {status['mode']}")
    print(f"  Actualizaciones:   {status['metrics']['total_updates']}")
    print(f"  Veh. en onda:      {status['metrics']['total_vehicles_in_wave']}")
    print(f"  Paradas evitadas:  {status['metrics']['total_stops_avoided']}")

    print("\n" + "=" * 70)
    print("  Demo completada OK")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_onda_verde()
