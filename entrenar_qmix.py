"""
ATLAS Pro - Entrenamiento QMIX Multi-Interseccion Completo
============================================================
Entrena QMIX en múltiples topologías de red:
  - Grid 3x3 (9 intersecciones)
  - Corredor 5 (5 intersecciones en línea)
  - Grid 2x2 (4 intersecciones, rápido)

Cada topología se entrena con optimización de onda verde.

Uso:
    python entrenar_qmix.py                    # Entrenamiento completo
    python entrenar_qmix.py --topologia grid3  # Solo grid 3x3
    python entrenar_qmix.py --topologia corr5  # Solo corredor
    python entrenar_qmix.py --rapido           # Entrenamiento rápido (menos episodios)
    python entrenar_qmix.py --ver-plan         # Solo ver el plan
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

# Verificar dependencias
try:
    import torch
    print(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("ERROR: PyTorch no disponible. pip install torch")
    sys.exit(1)

from multi_interseccion import (
    MultiIntersectionTrainer, NetworkTopology,
    QMIXSystem, MultiIntersectionEnv
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# CONFIGURACIONES DE ENTRENAMIENTO
# =============================================================================

TOPOLOGIAS = {
    "grid2": {
        "nombre": "Grid 2x2 (Calentamiento)",
        "tipo": "grid",
        "params": {"rows": 2, "cols": 2},
        "episodios": 300,
        "episodios_rapido": 100,
        "max_steps": 400,
        "descripcion": "Grid 2x2 para validar que QMIX aprende correctamente",
    },
    "grid3": {
        "nombre": "Grid 3x3 (Produccion)",
        "tipo": "grid",
        "params": {"rows": 3, "cols": 3},
        "episodios": 800,
        "episodios_rapido": 200,
        "max_steps": 600,
        "descripcion": "Grid 3x3 = 9 intersecciones. Escenario principal de produccion",
    },
    "corr5": {
        "nombre": "Corredor 5 Intersecciones",
        "tipo": "corridor",
        "params": {"n_intersections": 5},
        "episodios": 500,
        "episodios_rapido": 150,
        "max_steps": 500,
        "descripcion": "Corredor lineal de 5 semaforos. Ideal para onda verde",
    },
}


def mostrar_plan():
    """Muestra el plan de entrenamiento."""
    print(f"\n{'='*65}")
    print(f"  ATLAS Pro — Plan de Entrenamiento QMIX Multi-Interseccion")
    print(f"{'='*65}\n")

    total_ep = 0
    for key, cfg in TOPOLOGIAS.items():
        print(f"  [{key}] {cfg['nombre']}")
        print(f"         {cfg['descripcion']}")
        print(f"         Episodios: {cfg['episodios']} | Steps/ep: {cfg['max_steps']}")
        if cfg['tipo'] == 'grid':
            n = cfg['params']['rows'] * cfg['params']['cols']
        else:
            n = cfg['params']['n_intersections']
        print(f"         Agentes: {n} intersecciones")
        print(f"         Modelo: modelos/qmix_{key}.pt")
        print()
        total_ep += cfg['episodios']

    print(f"  Total: {total_ep} episodios en {len(TOPOLOGIAS)} topologias")
    print(f"  Tiempo estimado: ~{total_ep * 0.8 / 60:.0f}-{total_ep * 1.5 / 60:.0f} minutos")
    print(f"{'='*65}\n")


def entrenar_topologia(key: str, cfg: dict, rapido: bool = False) -> dict:
    """Entrena una topología específica."""
    episodios = cfg['episodios_rapido'] if rapido else cfg['episodios']
    save_path = os.path.join(MODEL_DIR, f"qmix_{key}.pt")

    print(f"\n{'='*65}", flush=True)
    print(f"  Entrenando: {cfg['nombre']}", flush=True)
    print(f"  Episodios: {episodios} | Steps: {cfg['max_steps']}", flush=True)
    print(f"  Guardando en: {save_path}", flush=True)
    print(f"{'='*65}\n", flush=True)

    t0 = time.time()

    trainer = MultiIntersectionTrainer(
        topology_type=cfg['tipo'],
        n_episodes=episodios,
        max_steps=cfg['max_steps'],
        **cfg['params']
    )

    # Change log_interval to 1 so we see progress every episode
    results = trainer.train(save_path=save_path, log_interval=1)
    elapsed = time.time() - t0

    results['topologia'] = key
    results['nombre'] = cfg['nombre']
    results['tiempo_segundos'] = elapsed
    results['save_path'] = save_path

    print(f"\n  Completado en {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"  Reward final: {results['final_avg_reward']:.2f}", flush=True)
    print(f"  Mejor reward: {results['best_avg_reward']:.2f}", flush=True)

    if results.get('green_wave'):
        gw = results['green_wave']
        print(f"  Onda verde — Bandwidth: {gw['bandwidth']:.1%}", flush=True)
        print(f"               Offsets: {[f'{o:.1f}s' for o in gw['offsets']]}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Pro - Entrenamiento QMIX Multi-Interseccion"
    )
    parser.add_argument("--topologia", type=str, default=None,
                       choices=list(TOPOLOGIAS.keys()),
                       help="Entrenar solo una topologia especifica")
    parser.add_argument("--rapido", action="store_true",
                       help="Entrenamiento rapido (menos episodios)")
    parser.add_argument("--ver-plan", action="store_true",
                       help="Solo mostrar plan de entrenamiento")
    args = parser.parse_args()

    if args.ver_plan:
        mostrar_plan()
        return

    print(f"\n{'#'*65}")
    print(f"  ATLAS Pro — Entrenamiento QMIX Multi-Interseccion")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Modo: {'Rapido' if args.rapido else 'Completo'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'#'*65}")

    # Seleccionar topologías
    if args.topologia:
        topos = {args.topologia: TOPOLOGIAS[args.topologia]}
    else:
        topos = TOPOLOGIAS

    # Entrenar
    all_results = {}
    t_total = time.time()

    for key, cfg in topos.items():
        results = entrenar_topologia(key, cfg, rapido=args.rapido)
        all_results[key] = {
            'nombre': results['nombre'],
            'final_avg_reward': results['final_avg_reward'],
            'best_avg_reward': results['best_avg_reward'],
            'episodes': results['episodes'],
            'tiempo_min': results['tiempo_segundos'] / 60,
            'green_wave': results.get('green_wave'),
            'save_path': results['save_path'],
        }

    total_time = time.time() - t_total

    # Resumen final
    print(f"\n{'#'*65}")
    print(f"  RESUMEN FINAL — Entrenamiento QMIX")
    print(f"{'#'*65}\n")

    for key, r in all_results.items():
        print(f"  {r['nombre']:40s} | Reward: {r['best_avg_reward']:>7.2f} | {r['tiempo_min']:.1f} min")

    print(f"\n  Tiempo total: {total_time/60:.1f} minutos")
    print(f"  Modelos guardados en: {MODEL_DIR}/")

    # Guardar resultados
    results_path = os.path.join(RESULTS_DIR, f"qmix_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Resultados: {results_path}")
    print(f"\n{'#'*65}\n")


if __name__ == "__main__":
    main()
