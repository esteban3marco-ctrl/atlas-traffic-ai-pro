"""
ATLAS Pro - Framework de A/B Testing para Modelos
===================================================
Comparación rigurosa de versiones de modelos:
- Split testing con asignación aleatoria
- Métricas estadísticas (t-test, Mann-Whitney)
- Análisis de significancia
- Multi-Armed Bandit para selección adaptativa
- Reportes de comparación
"""

import os
import json
import time
import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger("ATLAS.ABTesting")


@dataclass
class ExperimentResult:
    """Resultado de un paso del experimento"""
    timestamp: str
    variant: str
    action: int
    reward: float
    wait_time: float
    throughput: float
    queue_length: float
    latency_ms: float


class Variant:
    """Representa una variante (modelo) en el A/B test"""

    def __init__(self, name: str, model=None, select_action_fn: Callable = None):
        self.name = name
        self.model = model
        self.select_action_fn = select_action_fn
        self.results: List[ExperimentResult] = []
        self.total_reward = 0.0
        self.count = 0

    def record_result(self, result: ExperimentResult):
        self.results.append(result)
        self.total_reward += result.reward
        self.count += 1

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.count if self.count > 0 else 0

    @property
    def avg_wait_time(self) -> float:
        if not self.results:
            return 0
        return np.mean([r.wait_time for r in self.results])

    @property
    def avg_throughput(self) -> float:
        if not self.results:
            return 0
        return np.mean([r.throughput for r in self.results])

    @property
    def avg_queue(self) -> float:
        if not self.results:
            return 0
        return np.mean([r.queue_length for r in self.results])

    def get_metrics(self) -> Dict:
        return {
            'name': self.name,
            'count': self.count,
            'avg_reward': self.avg_reward,
            'avg_wait_time': self.avg_wait_time,
            'avg_throughput': self.avg_throughput,
            'avg_queue': self.avg_queue,
            'reward_std': float(np.std([r.reward for r in self.results])) if self.results else 0
        }


class ABTestExperiment:
    """
    Framework de A/B testing para comparar modelos ATLAS.

    Soporta:
    - Split testing simple (50/50)
    - Multi-variante (A/B/C/...)
    - Epsilon-greedy exploration
    - Thompson Sampling
    """

    def __init__(self, name: str, variants: List[Variant],
                 split_method: str = "random",
                 min_samples: int = 100,
                 confidence_level: float = 0.95):
        self.name = name
        self.variants = {v.name: v for v in variants}
        self.split_method = split_method
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        self.start_time = datetime.now().isoformat()
        self.is_active = True
        self.total_steps = 0

        # Thompson Sampling priors
        self.ts_alpha = {v.name: 1.0 for v in variants}
        self.ts_beta = {v.name: 1.0 for v in variants}

        logger.info(f"Experimento '{name}' creado con {len(variants)} variantes")

    def select_variant(self) -> str:
        """Selecciona variante según el método de split"""
        if self.split_method == "random":
            return random.choice(list(self.variants.keys()))

        elif self.split_method == "epsilon_greedy":
            epsilon = max(0.1, 1.0 - self.total_steps / 1000)
            if random.random() < epsilon:
                return random.choice(list(self.variants.keys()))
            else:
                return max(self.variants.values(), key=lambda v: v.avg_reward).name

        elif self.split_method == "thompson":
            samples = {}
            for name in self.variants:
                samples[name] = np.random.beta(self.ts_alpha[name], self.ts_beta[name])
            return max(samples, key=samples.get)

        elif self.split_method == "ucb":
            # Upper Confidence Bound
            if self.total_steps < len(self.variants):
                return list(self.variants.keys())[self.total_steps]
            ucb_values = {}
            for name, variant in self.variants.items():
                if variant.count == 0:
                    ucb_values[name] = float('inf')
                else:
                    exploitation = variant.avg_reward
                    exploration = np.sqrt(2 * np.log(self.total_steps) / variant.count)
                    ucb_values[name] = exploitation + exploration
            return max(ucb_values, key=ucb_values.get)

        return random.choice(list(self.variants.keys()))

    def record_result(self, variant_name: str, action: int, reward: float,
                     wait_time: float = 0, throughput: float = 0,
                     queue_length: float = 0, latency_ms: float = 0):
        """Registra resultado de un paso"""
        if variant_name not in self.variants:
            return

        result = ExperimentResult(
            timestamp=datetime.now().isoformat(),
            variant=variant_name,
            action=action,
            reward=reward,
            wait_time=wait_time,
            throughput=throughput,
            queue_length=queue_length,
            latency_ms=latency_ms
        )

        self.variants[variant_name].record_result(result)
        self.total_steps += 1

        # Update Thompson Sampling
        normalized_reward = (reward + 10) / 20  # Normalizar a [0, 1]
        normalized_reward = max(0, min(1, normalized_reward))
        self.ts_alpha[variant_name] += normalized_reward
        self.ts_beta[variant_name] += (1 - normalized_reward)

    def statistical_test(self, metric: str = 'reward') -> Dict:
        """
        Realiza test estadístico entre variantes.
        Usa t-test para muestras independientes y Mann-Whitney U.
        """
        from scipy import stats

        variant_list = list(self.variants.values())
        if len(variant_list) < 2:
            return {"error": "Se necesitan al menos 2 variantes"}

        results = {}

        for i in range(len(variant_list)):
            for j in range(i + 1, len(variant_list)):
                v1 = variant_list[i]
                v2 = variant_list[j]

                if not v1.results or not v2.results:
                    continue

                if metric == 'reward':
                    data1 = [r.reward for r in v1.results]
                    data2 = [r.reward for r in v2.results]
                elif metric == 'wait_time':
                    data1 = [r.wait_time for r in v1.results]
                    data2 = [r.wait_time for r in v2.results]
                elif metric == 'throughput':
                    data1 = [r.throughput for r in v1.results]
                    data2 = [r.throughput for r in v2.results]
                else:
                    continue

                # T-test
                t_stat, t_pvalue = stats.ttest_ind(data1, data2)

                # Mann-Whitney U
                u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

                pair_key = f"{v1.name}_vs_{v2.name}"
                results[pair_key] = {
                    'variant_a': v1.name,
                    'variant_b': v2.name,
                    'metric': metric,
                    'mean_a': float(np.mean(data1)),
                    'mean_b': float(np.mean(data2)),
                    'std_a': float(np.std(data1)),
                    'std_b': float(np.std(data2)),
                    't_statistic': float(t_stat),
                    't_pvalue': float(t_pvalue),
                    'u_statistic': float(u_stat),
                    'u_pvalue': float(u_pvalue),
                    'cohens_d': float(cohens_d),
                    'significant': t_pvalue < (1 - self.confidence_level),
                    'winner': v1.name if np.mean(data1) > np.mean(data2) else v2.name,
                    'samples_a': len(data1),
                    'samples_b': len(data2)
                }

        return results

    def get_summary(self) -> Dict:
        """Resumen del experimento"""
        variant_metrics = {name: v.get_metrics() for name, v in self.variants.items()}

        best_reward = max(self.variants.values(), key=lambda v: v.avg_reward)
        best_throughput = max(self.variants.values(), key=lambda v: v.avg_throughput)

        return {
            'experiment_name': self.name,
            'start_time': self.start_time,
            'total_steps': self.total_steps,
            'split_method': self.split_method,
            'is_active': self.is_active,
            'variants': variant_metrics,
            'best_reward': best_reward.name,
            'best_throughput': best_throughput.name,
            'sufficient_data': all(v.count >= self.min_samples for v in self.variants.values())
        }

    def generate_report(self) -> str:
        """Genera reporte textual del experimento"""
        lines = [
            "=" * 60,
            f"📊 A/B Test Report: {self.name}",
            "=" * 60,
            f"Inicio: {self.start_time}",
            f"Total pasos: {self.total_steps}",
            f"Método: {self.split_method}",
            ""
        ]

        for name, variant in self.variants.items():
            metrics = variant.get_metrics()
            lines.append(f"--- {name} ({metrics['count']} muestras) ---")
            lines.append(f"  Reward:     {metrics['avg_reward']:+.3f} ± {metrics['reward_std']:.3f}")
            lines.append(f"  Wait Time:  {metrics['avg_wait_time']:.1f}s")
            lines.append(f"  Throughput: {metrics['avg_throughput']:.1f} veh/min")
            lines.append(f"  Queue:      {metrics['avg_queue']:.1f} veh")
            lines.append("")

        # Test estadístico
        try:
            test_results = self.statistical_test('reward')
            if test_results:
                lines.append("--- Análisis Estadístico ---")
                for pair, result in test_results.items():
                    sig = "✅ SIGNIFICATIVO" if result['significant'] else "❌ No significativo"
                    lines.append(f"  {pair}: p={result['t_pvalue']:.4f} (d={result['cohens_d']:.3f}) {sig}")
                    if result['significant']:
                        lines.append(f"    Ganador: {result['winner']}")
        except ImportError:
            lines.append("  (scipy no disponible para test estadístico)")

        return "\n".join(lines)

    def save(self, path: str):
        """Guarda estado del experimento"""
        data = {
            'name': self.name,
            'start_time': self.start_time,
            'split_method': self.split_method,
            'total_steps': self.total_steps,
            'variants': {}
        }
        for name, variant in self.variants.items():
            data['variants'][name] = {
                'metrics': variant.get_metrics(),
                'results': [asdict(r) for r in variant.results[-1000:]]
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Experimento guardado: {path}")


# =============================================================================
# EJEMPLO
# =============================================================================

def ejemplo_ab_testing():
    """Demo del framework de A/B testing"""
    print("\n" + "=" * 70)
    print("🧪 ATLAS Pro - Framework de A/B Testing")
    print("=" * 70)

    # Crear variantes simuladas
    variants = [
        Variant("DuelingDDQN_v1"),
        Variant("DuelingDDQN_v2"),
        Variant("C51_v1")
    ]

    experiment = ABTestExperiment(
        name="Comparación Algoritmos Q1-2026",
        variants=variants,
        split_method="thompson",
        min_samples=50
    )

    # Simular tráfico
    print("\n🔄 Simulando 500 pasos de tráfico...")
    for step in range(500):
        selected = experiment.select_variant()

        # Simular rendimiento (v2 es ligeramente mejor)
        if selected == "DuelingDDQN_v2":
            reward = np.random.normal(2.0, 3.0)
            wait = np.random.uniform(20, 60)
        elif selected == "C51_v1":
            reward = np.random.normal(1.5, 2.5)
            wait = np.random.uniform(25, 65)
        else:
            reward = np.random.normal(1.0, 3.5)
            wait = np.random.uniform(30, 70)

        experiment.record_result(
            selected,
            action=random.randint(0, 3),
            reward=reward,
            wait_time=wait,
            throughput=np.random.uniform(30, 60),
            queue_length=np.random.uniform(5, 30)
        )

    # Reporte
    print(experiment.generate_report())

    # Summary
    summary = experiment.get_summary()
    print(f"\n🏆 Mejor reward: {summary['best_reward']}")
    print(f"   Datos suficientes: {summary['sufficient_data']}")

    print("\n✅ Demo completada")


if __name__ == "__main__":
    ejemplo_ab_testing()
