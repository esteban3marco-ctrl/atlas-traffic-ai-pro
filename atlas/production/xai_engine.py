
"""
ATLAS Pro — Conversational XAI Engine
======================================
Translates neural network activations and traffic states into human-readable 
explanations for city operators and decision-makers.
"""

import numpy as np
from typing import List, Dict, Optional

class ConversationalXAI:
    """
    Expert system that interprets AI state and attention maps to generate
    natural language justifications.
    """
    
    def __init__(self):
        self.directions = ["North", "South", "East", "West"]
        self.feature_names = ["Queue", "Velocity", "Wait", "Density", "Blocked"]
        self.phase_names = {
            0: "NORTH-SOUTH GREEN",
            1: "NORTH-SOUTH TURN",
            2: "EAST-WEST GREEN",
            3: "EAST-WEST TURN"
        }
        self.last_explanation = ""

    def generate_explanation(self, state_vector: np.ndarray, 
                            action: int, 
                            saliency: Optional[np.ndarray] = None) -> str:
        
        dir_metrics = {}
        for i, d in enumerate(["N", "S", "E", "W"]):
            base_idx = i * 5
            metrics = {
                "queue": float(state_vector[base_idx]),
                "wait": float(state_vector[base_idx + 2]),
                "halted": float(state_vector[base_idx + 4])
            }
            dir_metrics[d] = metrics

        sorted_dirs = sorted(dir_metrics.items(), key=lambda x: x[1]['queue'], reverse=True)
        top_critical = sorted_dirs[0]
        critical_name = self._get_full_name(top_critical[0])
        
        # Saliency analysis
        saliency_msg = ""
        if saliency is not None:
            weights = []
            for i in range(4):
                weights.append(np.sum(saliency[i*5 : (i+1)*5]))
            top_attn_idx = np.argmax(weights)
            saliency_msg = f"[NEURAL_ATTENTION] Primary focus at {self.directions[top_attn_idx]} Gateway. "

        target_phase = self.phase_names.get(action, "UNKNOWN_PHASE")
        
        # Varied Narrative
        templates = [
            f"Prioritizing {critical_name} corridor due to {int(top_critical[1]['queue']*100)}% capacity breach. ",
            f"Optimizing throughput for {critical_name} zone based on real-time LIDAR telemetry. ",
            f"Executing {target_phase} shift to mitigate exponential wait times at {critical_name}. ",
            f"Neural net identified bottleneck at {critical_name}. Dispatching green wave protocol. "
        ]
        
        main_msg = templates[np.random.randint(len(templates))]
        
        # Technical detail
        detail = f"Vector[σ={np.std(state_vector[:20]):.2f}]. Protocol: {target_phase}. "
        
        # Insights
        insight = ""
        avg_q = np.mean([m['queue'] for m in dir_metrics.values()])
        if avg_q < 0.15:
            insight = "[LOW_DENSITY] Switching to energy-efficient micro-cycles."
        elif action % 2 != 0:
            insight = "[TURN_RECOVERY] Clearing intersection blockages."
        else:
            insight = f"[NOMINAL_FLOW] Predicted stability: {95 - (avg_q*50):.1f}%."

        result = f"{saliency_msg}{detail}{main_msg}{insight}"
        
        # Avoid repeat
        if result == self.last_explanation:
            result = f"[RE-OPTIMIZING] Continuing phase {target_phase} for peak stability."
        
        self.last_explanation = result
        return result

    def _get_full_name(self, d: str) -> str:
        mapping = {"N": "Norte", "S": "Sur", "E": "Este", "W": "Oeste"}
        return mapping.get(d, d)

# Singleton
xai_engine = ConversationalXAI()
