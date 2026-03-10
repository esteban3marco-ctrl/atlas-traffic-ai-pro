"""
ATLAS Pro - Motor de Explicabilidad (XAI)
==========================================
Herramientas para explicar las decisiones de la IA:
- Mapas de saliencia (Gradient-based Saliency)
- Atribución de features (Integrated Gradients)
- Análisis SHAP-like para importancia de variables
- Generación de explicaciones en lenguaje natural
- Visualización de la toma de decisiones
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("ATLAS.XAI")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_DISPONIBLE = True
except ImportError:
    MPL_DISPONIBLE = False


# =============================================================================
# NOMBRES DE FEATURES DEL ESTADO
# =============================================================================

FEATURE_NAMES_12D = [
    'Cola Norte', 'Cola Sur', 'Cola Este', 'Cola Oeste',
    'Coches', 'Motos', 'Buses', 'Camiones',
    'Reservado 1', 'Reservado 2', 'Reservado 3', 'Reservado 4'
]

FEATURE_NAMES_26D = [
    'Cola Norte', 'Cola Sur', 'Cola Este', 'Cola Oeste',
    'Espera Norte', 'Espera Sur', 'Espera Este', 'Espera Oeste',
    'Velocidad Norte', 'Velocidad Sur', 'Velocidad Este', 'Velocidad Oeste',
    'Vehículos Norte', 'Vehículos Sur', 'Vehículos Este', 'Vehículos Oeste',
    'Fase Actual', 'Tiempo en Fase', 'Hora del Día',
    'Es Hora Punta', 'Es Fin de Semana',
    'Emergencia Norte', 'Emergencia Sur', 'Emergencia Este', 'Emergencia Oeste',
    'Densidad Global'
]

ACTION_NAMES = ['Mantener Fase', 'Cambiar a N-S', 'Cambiar a E-O', 'Extender Fase']


class GradientSaliency:
    """
    Mapas de saliencia basados en gradientes.
    Muestra qué features del input son más importantes para la decisión.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def compute_saliency(self, state: np.ndarray, action: int = None) -> np.ndarray:
        """
        Calcula mapa de saliencia para un estado.

        Args:
            state: Vector de estado
            action: Acción específica (None = mejor acción)

        Returns:
            Array de saliencia por feature
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_tensor.requires_grad_(True)

        q_values = self.model(state_tensor)

        if action is None:
            action = q_values.argmax(dim=1).item()

        # Gradiente respecto al Q-value de la acción seleccionada
        self.model.zero_grad()
        q_values[0, action].backward()

        saliency = state_tensor.grad.data.abs().squeeze().numpy()
        return saliency

    def compute_smooth_saliency(self, state: np.ndarray, action: int = None,
                                n_samples: int = 50, noise_std: float = 0.1) -> np.ndarray:
        """
        SmoothGrad: Promedia gradientes sobre inputs perturbados.
        Más estable que saliencia vanilla.
        """
        saliency_maps = []

        for _ in range(n_samples):
            noisy_state = state + np.random.normal(0, noise_std, state.shape).astype(np.float32)
            saliency = self.compute_saliency(noisy_state, action)
            saliency_maps.append(saliency)

        return np.mean(saliency_maps, axis=0)


class IntegratedGradients:
    """
    Integrated Gradients para atribución de features.
    Más riguroso que saliencia simple, satisface axiomas de atribución.

    Ref: "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def compute_attribution(self, state: np.ndarray, action: int = None,
                           baseline: np.ndarray = None,
                           n_steps: int = 100) -> np.ndarray:
        """
        Calcula atribución por Integrated Gradients.

        Args:
            state: Estado actual
            action: Acción objetivo
            baseline: Estado de referencia (default: zeros)
            n_steps: Pasos de integración

        Returns:
            Array de atribución por feature
        """
        if baseline is None:
            baseline = np.zeros_like(state)

        # Interpolación lineal entre baseline y state
        alphas = np.linspace(0, 1, n_steps + 1)
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (state - baseline)
            interp_tensor = torch.FloatTensor(interpolated).unsqueeze(0)
            interp_tensor.requires_grad_(True)

            q_values = self.model(interp_tensor)
            if action is None:
                action = q_values.argmax(dim=1).item()

            self.model.zero_grad()
            q_values[0, action].backward()

            gradients.append(interp_tensor.grad.data.squeeze().numpy())

        # Integración trapezoidal
        avg_gradients = np.mean(gradients, axis=0)
        attributions = (state - baseline) * avg_gradients

        return attributions


class FeatureImportanceAnalyzer:
    """
    Análisis de importancia de features estilo SHAP.
    Usa perturbaciones para estimar contribución marginal.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def permutation_importance(self, states: np.ndarray,
                              n_repeats: int = 10) -> Dict[str, float]:
        """
        Importancia por permutación: mide cuánto empeora el Q-value
        al permutar cada feature.
        """
        states_tensor = torch.FloatTensor(states)
        baseline_q = self.model(states_tensor).max(dim=1)[0].mean().item()

        importance = {}
        n_features = states.shape[1]

        for feat_idx in range(n_features):
            scores = []
            for _ in range(n_repeats):
                permuted = states.copy()
                np.random.shuffle(permuted[:, feat_idx])

                permuted_tensor = torch.FloatTensor(permuted)
                permuted_q = self.model(permuted_tensor).max(dim=1)[0].mean().item()
                scores.append(baseline_q - permuted_q)

            feature_name = self._get_feature_name(feat_idx, n_features)
            importance[feature_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'raw_scores': [float(s) for s in scores]
            }

        # Ordenar por importancia
        importance = dict(sorted(importance.items(),
                                key=lambda x: abs(x[1]['mean']), reverse=True))
        return importance

    def _get_feature_name(self, idx: int, total: int) -> str:
        if total <= 12:
            names = FEATURE_NAMES_12D
        else:
            names = FEATURE_NAMES_26D
        return names[idx] if idx < len(names) else f"Feature_{idx}"


class NaturalLanguageExplainer:
    """
    Genera explicaciones en lenguaje natural para las decisiones.
    """

    def __init__(self, model: nn.Module = None):
        self.model = model
        if model and TORCH_DISPONIBLE:
            self.saliency = GradientSaliency(model)
            self.ig = IntegratedGradients(model)
        else:
            self.saliency = None
            self.ig = None

    def explain_decision(self, state: np.ndarray, action: int,
                        q_values: np.ndarray = None,
                        traffic_state: Dict = None) -> Dict:
        """
        Genera explicación completa de una decisión.

        Returns:
            Dict con explicación, razones, confianza, etc.
        """
        state_dim = len(state)
        feature_names = FEATURE_NAMES_26D if state_dim > 12 else FEATURE_NAMES_12D

        # Saliencia
        if self.saliency:
            saliency = self.saliency.compute_saliency(state, action)
            top_features_idx = np.argsort(saliency)[::-1][:5]
            top_features = [(feature_names[i] if i < len(feature_names) else f"Feature_{i}",
                            float(saliency[i])) for i in top_features_idx]
        else:
            top_features = []

        # Q-values
        if q_values is not None:
            confidence = self._compute_confidence(q_values, action)
            action_ranking = np.argsort(q_values)[::-1]
        else:
            confidence = 0.5
            action_ranking = [action]

        # Generar explicación
        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"Acción {action}"
        explanation = self._generate_text(
            action_name, top_features, confidence, state, traffic_state
        )

        return {
            'action': action,
            'action_name': action_name,
            'explanation': explanation,
            'confidence': confidence,
            'top_features': top_features,
            'q_values': q_values.tolist() if q_values is not None else None,
            'action_ranking': [ACTION_NAMES[i] for i in action_ranking if i < len(ACTION_NAMES)],
            'timestamp': datetime.now().isoformat()
        }

    def _compute_confidence(self, q_values: np.ndarray, action: int) -> float:
        """Calcula confianza como diferencia normalizada con segunda mejor"""
        sorted_q = np.sort(q_values)[::-1]
        if len(sorted_q) < 2:
            return 1.0
        diff = sorted_q[0] - sorted_q[1]
        # Normalizar a [0, 1] usando sigmoid
        confidence = 1 / (1 + np.exp(-diff * 2))
        return float(confidence)

    def _generate_text(self, action_name: str, top_features: List,
                      confidence: float, state: np.ndarray,
                      traffic_state: Dict = None) -> str:
        """Genera texto de explicación"""
        lines = []

        # Decisión principal
        conf_text = "alta" if confidence > 0.7 else "media" if confidence > 0.4 else "baja"
        lines.append(f"Decisión: {action_name} (confianza {conf_text}: {confidence:.1%})")

        # Razones basadas en features
        if top_features:
            lines.append("\nFactores principales:")
            for name, importance in top_features[:3]:
                lines.append(f"  • {name} (importancia: {importance:.3f})")

        # Contexto del tráfico
        if traffic_state:
            queues = {
                'Norte': traffic_state.get('cola_norte', 0),
                'Sur': traffic_state.get('cola_sur', 0),
                'Este': traffic_state.get('cola_este', 0),
                'Oeste': traffic_state.get('cola_oeste', 0)
            }
            max_dir = max(queues, key=queues.get)
            max_queue = queues[max_dir]

            if max_queue > 30:
                lines.append(f"\nAlerta: Cola significativa en dirección {max_dir} ({max_queue} vehículos)")
            elif max_queue > 15:
                lines.append(f"\nCola moderada en {max_dir} ({max_queue} vehículos)")

        # Justificación de la acción
        if "Mantener" in action_name:
            lines.append("\nRazón: El tráfico está equilibrado o la fase actual aún es efectiva.")
        elif "N-S" in action_name:
            lines.append("\nRazón: Mayor demanda detectada en el eje Norte-Sur.")
        elif "E-O" in action_name:
            lines.append("\nRazón: Mayor demanda detectada en el eje Este-Oeste.")
        elif "Extender" in action_name:
            lines.append("\nRazón: La fase actual requiere más tiempo para despejar la cola.")

        return "\n".join(lines)


class XAIVisualizer:
    """
    Visualización de explicaciones XAI.
    """

    def __init__(self, output_dir: str = "xai_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_saliency_bar(self, saliency: np.ndarray, feature_names: List[str] = None,
                         title: str = "Saliencia por Feature",
                         filename: str = None) -> Optional[str]:
        """Gráfico de barras de saliencia"""
        if not MPL_DISPONIBLE:
            return None

        n = len(saliency)
        if feature_names is None:
            feature_names = FEATURE_NAMES_26D[:n] if n > 12 else FEATURE_NAMES_12D[:n]

        # Ordenar por importancia
        indices = np.argsort(saliency)[::-1]
        sorted_saliency = saliency[indices]
        sorted_names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in indices]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn_r(sorted_saliency / max(sorted_saliency) if max(sorted_saliency) > 0 else sorted_saliency)
        bars = ax.barh(range(n), sorted_saliency, color=colors)

        ax.set_yticks(range(n))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Importancia (Saliencia)')
        ax.set_title(title)
        ax.invert_yaxis()

        plt.tight_layout()

        if filename is None:
            filename = f"saliency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_decision_summary(self, explanation: Dict,
                            filename: str = None) -> Optional[str]:
        """Gráfico resumen de decisión"""
        if not MPL_DISPONIBLE:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Q-Values
        ax1 = axes[0]
        if explanation.get('q_values'):
            q_vals = explanation['q_values']
            colors = ['#2ecc71' if i == explanation['action'] else '#95a5a6'
                     for i in range(len(q_vals))]
            bars = ax1.bar(ACTION_NAMES[:len(q_vals)], q_vals, color=colors)
            ax1.set_ylabel('Q-Value')
            ax1.set_title('Valores Q por Acción')
            ax1.tick_params(axis='x', rotation=20)

        # Top Features
        ax2 = axes[1]
        if explanation.get('top_features'):
            names = [f[0] for f in explanation['top_features']]
            values = [f[1] for f in explanation['top_features']]
            ax2.barh(names, values, color='#3498db')
            ax2.set_xlabel('Importancia')
            ax2.set_title('Features más Relevantes')
            ax2.invert_yaxis()

        fig.suptitle(f"Decisión: {explanation.get('action_name', '?')} "
                    f"(Confianza: {explanation.get('confidence', 0):.1%})",
                    fontsize=13, fontweight='bold')

        plt.tight_layout()

        if filename is None:
            filename = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_feature_importance_history(self, history: List[Dict],
                                       filename: str = None) -> Optional[str]:
        """Evolución temporal de importancia de features"""
        if not MPL_DISPONIBLE or not history:
            return None

        fig, ax = plt.subplots(figsize=(14, 6))

        all_features = set()
        for h in history:
            for feat, _ in h.get('top_features', []):
                all_features.add(feat)

        top_features = list(all_features)[:8]

        for feat in top_features:
            values = []
            for h in history:
                val = 0
                for f, v in h.get('top_features', []):
                    if f == feat:
                        val = v
                        break
                values.append(val)
            ax.plot(values, label=feat, linewidth=2)

        ax.set_xlabel('Paso de Decisión')
        ax.set_ylabel('Importancia')
        ax.set_title('Evolución de Importancia de Features')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if filename is None:
            filename = f"importance_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath


# =============================================================================
# MOTOR XAI INTEGRADO
# =============================================================================

class MotorXAI:
    """
    Motor XAI integrado que combina todas las herramientas de explicabilidad.
    """

    def __init__(self, model: nn.Module = None, output_dir: str = "xai_reports"):
        self.model = model
        self.explainer = NaturalLanguageExplainer(model)
        self.visualizer = XAIVisualizer(output_dir)
        self.explanation_history = []

        if model and TORCH_DISPONIBLE:
            self.saliency = GradientSaliency(model)
            self.ig = IntegratedGradients(model)
            self.feature_analyzer = FeatureImportanceAnalyzer(model)
        else:
            self.saliency = None
            self.ig = None
            self.feature_analyzer = None

        logger.info("MotorXAI inicializado")

    @torch.no_grad() if TORCH_DISPONIBLE else lambda f: f
    def explain(self, state: np.ndarray, action: int = None,
               traffic_state: Dict = None,
               generate_plot: bool = False) -> Dict:
        """
        Genera explicación completa de una decisión.
        """
        if self.model is None:
            return {"error": "Modelo no configurado"}

        # Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.enable_grad():
            q_values = self.model(state_tensor).detach().numpy()[0]

        if action is None:
            action = int(np.argmax(q_values))

        # Explicación textual
        explanation = self.explainer.explain_decision(
            state, action, q_values, traffic_state
        )

        # Saliencia
        if self.saliency:
            with torch.enable_grad():
                saliency = self.saliency.compute_smooth_saliency(state, action)
            explanation['saliency'] = saliency.tolist()

        # Integrated Gradients
        if self.ig:
            with torch.enable_grad():
                attribution = self.ig.compute_attribution(state, action)
            explanation['attribution'] = attribution.tolist()

        # Guardar en historial
        self.explanation_history.append(explanation)

        # Generar visualización
        if generate_plot and self.saliency:
            plot_path = self.visualizer.plot_decision_summary(explanation)
            explanation['plot_path'] = plot_path

        return explanation

    def analyze_feature_importance(self, states: np.ndarray) -> Dict:
        """Análisis global de importancia de features"""
        if self.feature_analyzer is None:
            return {"error": "Modelo no configurado"}
        return self.feature_analyzer.permutation_importance(states)

    def get_history(self) -> List[Dict]:
        """Historial de explicaciones"""
        return self.explanation_history

    def generate_report(self) -> str:
        """Genera reporte de explicabilidad"""
        if not self.explanation_history:
            return "No hay explicaciones registradas."

        lines = [
            "=" * 60,
            "📊 ATLAS XAI - Reporte de Explicabilidad",
            "=" * 60,
            f"Total de decisiones explicadas: {len(self.explanation_history)}",
            ""
        ]

        # Estadísticas de acciones
        actions = [e['action_name'] for e in self.explanation_history]
        for action_name in ACTION_NAMES:
            count = actions.count(action_name)
            pct = count / len(actions) * 100 if actions else 0
            lines.append(f"  {action_name}: {count} ({pct:.1f}%)")

        # Confianza promedio
        confidences = [e.get('confidence', 0) for e in self.explanation_history]
        lines.append(f"\nConfianza promedio: {np.mean(confidences):.1%}")
        lines.append(f"Confianza mínima:  {np.min(confidences):.1%}")

        # Features más importantes
        if self.explanation_history[0].get('top_features'):
            feature_counts = {}
            for exp in self.explanation_history:
                for feat, _ in exp.get('top_features', []):
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1

            lines.append("\nFeatures más frecuentemente importantes:")
            for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  • {feat}: {count} veces")

        return "\n".join(lines)


# =============================================================================
# EJEMPLO
# =============================================================================

def ejemplo_xai():
    """Demo del motor XAI"""
    print("\n" + "=" * 70)
    print("🔍 ATLAS Pro - Motor de Explicabilidad (XAI)")
    print("=" * 70)

    if not TORCH_DISPONIBLE:
        print("❌ PyTorch necesario")
        return

    # Crear modelo simple
    model = nn.Sequential(
        nn.Linear(26, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 4)
    )

    motor = MotorXAI(model)

    # Simular decisiones
    print("\n🧪 Generando explicaciones...")
    for i in range(10):
        state = np.random.randn(26).astype(np.float32)
        explanation = motor.explain(state, generate_plot=(i == 0))

        if i < 3:
            print(f"\n--- Decisión {i+1} ---")
            print(explanation['explanation'])

    # Reporte
    print("\n" + motor.generate_report())
    print("\n✅ Demo completada")


if __name__ == "__main__":
    ejemplo_xai()
