"""
ATLAS Pro - Ronda 5: Optimizacion Intensiva de Escenarios Debiles
==================================================================
Foco en romper los techos de:
  - evento (5.8 en R4, el mas dificil)
  - heavy (118.6 estancado desde R2)

Estrategia:
  1. Reward shaping especifico por escenario
  2. Mas episodios concentrados en los debiles
  3. Fine-tuning con LR reducido para no perder lo aprendido
  4. MUSE v2 activo para metacognicion
  5. Entrenamiento cruzado: alternar heavy <-> evento para generalizacion

Uso:
    python entrenar_ronda5.py              # Ronda completa
    python entrenar_ronda5.py --fase 3     # Empezar desde fase 3
    python entrenar_ronda5.py --ver-plan
"""

import os
import sys
import time
import copy
import argparse
import numpy as np

try:
    import yaml
except ImportError:
    print("ERROR: pip install pyyaml")
    sys.exit(1)

from ejecutar_entrenamiento_pro import entrenar_pro, ESCENARIOS, MODEL_DIR


# =============================================================================
# CONFIGURACIONES ESPECIALIZADAS POR ESCENARIO
# =============================================================================

def config_evento_intensivo(base_config):
    """
    Config optimizada para evento masivo:
    - throughput mucho mas alto (lo que importa es mover coches)
    - penalizaciones MUY suaves (en eventos hay congestion inevitable)
    - LR bajo para fine-tuning estable
    """
    cfg = copy.deepcopy(base_config)
    cfg["agent"]["lr"] = 0.0001          # LR bajo para no destruir
    cfg["agent"]["noisy_sigma"] = 0.3    # menos exploracion, mas explotacion

    cfg["reward"]["throughput_weight"] = 2.0    # mover coches es TODO
    cfg["reward"]["speed_weight"] = 0.5         # premiar flujo
    cfg["reward"]["queue_length_weight"] = -0.05  # casi no penalizar colas
    cfg["reward"]["wait_time_weight"] = -0.04     # casi no penalizar espera
    cfg["reward"]["phase_change_penalty"] = -0.08  # cambiar fase muy barato
    cfg["reward"]["clip_min"] = -25.0
    cfg["reward"]["clip_max"] = 25.0

    cfg["environment"]["max_steps"] = 2500   # episodios mas largos para eventos
    return cfg


def config_heavy_intensivo(base_config):
    """
    Config optimizada para heavy/hora punta:
    - Equilibrio entre throughput y reducir colas
    - LR bajo para fine-tuning
    """
    cfg = copy.deepcopy(base_config)
    cfg["agent"]["lr"] = 0.00015         # LR medio-bajo
    cfg["agent"]["noisy_sigma"] = 0.35

    cfg["reward"]["throughput_weight"] = 1.8
    cfg["reward"]["speed_weight"] = 0.5
    cfg["reward"]["queue_length_weight"] = -0.08
    cfg["reward"]["wait_time_weight"] = -0.06
    cfg["reward"]["phase_change_penalty"] = -0.10
    cfg["reward"]["clip_min"] = -22.0
    cfg["reward"]["clip_max"] = 22.0

    cfg["environment"]["max_steps"] = 2200
    return cfg


def config_consolidacion(base_config):
    """Config para fases de consolidacion (LR muy bajo, no perder lo aprendido)."""
    cfg = copy.deepcopy(base_config)
    cfg["agent"]["lr"] = 0.00008
    cfg["agent"]["noisy_sigma"] = 0.25   # poca exploracion
    return cfg


# =============================================================================
# PLAN RONDA 5
# =============================================================================

PLAN_RONDA5 = [
    # Fase 1: Evento intensivo (el mas debil: 5.8)
    {
        "fase": 1,
        "nombre": "EVENTO INTENSIVO",
        "descripcion": "Config especial evento: throughput=2.0, penalizaciones minimas",
        "escenario": "evento",
        "episodios": 800,
        "config_fn": "config_evento_intensivo",
    },
    # Fase 2: Heavy intensivo (estancado en 118.6)
    {
        "fase": 2,
        "nombre": "HEAVY INTENSIVO",
        "descripcion": "Config especial heavy: throughput=1.8, LR reducido",
        "escenario": "heavy",
        "episodios": 700,
        "config_fn": "config_heavy_intensivo",
    },
    # Fase 3: Cruzado evento -> heavy (generalizacion)
    {
        "fase": 3,
        "nombre": "CRUZADO: EVENTO->HEAVY",
        "descripcion": "Entrenar heavy con modelo de evento para compartir estrategias",
        "escenario": "heavy",
        "episodios": 300,
        "config_fn": "config_heavy_intensivo",
    },
    # Fase 4: Cruzado heavy -> evento
    {
        "fase": 4,
        "nombre": "CRUZADO: HEAVY->EVENTO",
        "descripcion": "Entrenar evento con lo aprendido en heavy",
        "escenario": "evento",
        "episodios": 500,
        "config_fn": "config_evento_intensivo",
    },
    # Fase 5: Consolidar emergencias (ya bueno, mantener)
    {
        "fase": 5,
        "nombre": "CONSOLIDAR EMERGENCIAS",
        "descripcion": "Mantener emergencias alto con LR muy bajo",
        "escenario": "emergencias",
        "episodios": 200,
        "config_fn": "config_consolidacion",
    },
    # Fase 6: Consolidar avenida
    {
        "fase": 6,
        "nombre": "CONSOLIDAR AVENIDA",
        "descripcion": "Mantener avenida alto con LR muy bajo",
        "escenario": "avenida",
        "episodios": 200,
        "config_fn": "config_consolidacion",
    },
    # Fase 7: Test final evento
    {
        "fase": 7,
        "nombre": "EVENTO TEST FINAL",
        "descripcion": "Ultimo push en evento con config agresiva",
        "escenario": "evento",
        "episodios": 400,
        "config_fn": "config_evento_intensivo",
    },
]


CONFIG_FUNCTIONS = {
    "config_evento_intensivo": config_evento_intensivo,
    "config_heavy_intensivo": config_heavy_intensivo,
    "config_consolidacion": config_consolidacion,
}


def mostrar_plan(plan):
    total_eps = sum(f["episodios"] for f in plan)
    print()
    print("=" * 70)
    print("  ATLAS Pro - RONDA 5: Optimizacion Intensiva")
    print("  Foco: evento (5.8) y heavy (118.6)")
    print("  Configs especializadas por escenario + MUSE v2")
    print("=" * 70)
    print()
    for f in plan:
        print(f"  Fase {f['fase']}: {f['nombre']:<25s} | "
              f"{f['escenario']:<15s} | {f['episodios']:>4d} eps")
        print(f"          {f['descripcion']}")
    print()
    print(f"  TOTAL: {total_eps} episodios en {len(plan)} fases")
    print("=" * 70)
    print()


def ejecutar_plan(plan, base_config, desde_fase=1):
    resultados = {}
    inicio_total = time.time()

    for etapa in plan:
        fase = etapa["fase"]
        if fase < desde_fase:
            print(f"\n  [Saltando Fase {fase}: {etapa['nombre']}]")
            continue

        # Usar config especializada si existe
        config_fn_name = etapa.get("config_fn")
        if config_fn_name and config_fn_name in CONFIG_FUNCTIONS:
            config = CONFIG_FUNCTIONS[config_fn_name](base_config)
            print(f"\n  Config especializada: {config_fn_name}")
        else:
            config = base_config

        print()
        print("#" * 70)
        print(f"#  FASE {fase}/{len(plan)}: {etapa['nombre']}")
        print(f"#  Escenario: {etapa['escenario']} | Episodios: {etapa['episodios']}")
        print(f"#  {etapa['descripcion']}")
        print(f"#  LR: {config['agent']['lr']} | Throughput: {config['reward']['throughput_weight']}")
        print("#" * 70)

        inicio_fase = time.time()

        rewards = entrenar_pro(
            config=config,
            scenario=etapa["escenario"],
            num_episodes=etapa["episodios"],
            gui=False,
            use_muse=True,
        )

        duracion = time.time() - inicio_fase

        if rewards:
            mejor = max(rewards)
            media_final = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
            media_inicio = np.mean(rewards[:10]) if len(rewards) >= 10 else np.mean(rewards[:3])
            mejora = media_final - media_inicio
        else:
            mejor = media_final = mejora = 0

        resultados[fase] = {
            "nombre": etapa["nombre"],
            "escenario": etapa["escenario"],
            "episodios": etapa["episodios"],
            "config": config_fn_name or "base",
            "mejor_reward": mejor,
            "media_final": media_final,
            "mejora": mejora,
            "duracion_min": duracion / 60,
        }

        print()
        print(f"  >>> Fase {fase} completada en {duracion/60:.1f} min")
        print(f"  >>> Mejor: {mejor:.1f} | Media final: {media_final:.1f} | Mejora: {mejora:+.1f}")
        print()

    # Resumen final
    duracion_total = time.time() - inicio_total
    print()
    print("=" * 70)
    print("  ATLAS Pro - RESUMEN RONDA 5 (Optimizacion Intensiva)")
    print("=" * 70)
    print()
    print(f"  {'Fase':<5s} {'Nombre':<25s} {'Escenario':<12s} "
          f"{'Config':<22s} {'Mejor':>8s} {'Media':>8s} {'Mejora':>8s} {'Tiempo':>8s}")
    print("  " + "-" * 95)

    for fase, r in sorted(resultados.items()):
        print(f"  {fase:<5d} {r['nombre']:<25s} {r['escenario']:<12s} "
              f"{r['config']:<22s} {r['mejor_reward']:>8.1f} {r['media_final']:>8.1f} "
              f"{r['mejora']:>+8.1f} {r['duracion_min']:>7.1f}m")

    print()
    print(f"  Tiempo total: {duracion_total/60:.1f} minutos ({duracion_total/3600:.1f} horas)")
    print()

    # Comparativa historica
    historico = {
        "simple":       {"R1": 76.9, "R2": 359.1, "R3": 293.7, "R4": 293.7},
        "noche":        {"R1": 291.8, "R2": 291.8, "R3": 717.2, "R4": 717.2},
        "heavy":        {"R1": -35.1, "R2": 118.6, "R3": 53.7,  "R4": 118.1},
        "emergencias":  {"R1": 118.9, "R2": 429.5, "R3": 311.9, "R4": 424.4},
        "evento":       {"R1": -73.1, "R2": -7.1,  "R3": -57.8, "R4": 5.8},
        "avenida":      {"R1": -1.4,  "R2": 238.9, "R3": 143.7, "R4": 257.4},
    }

    print("  EVOLUCION HISTORICA COMPLETA:")
    print(f"  {'Escenario':<15s} {'R1':>8s} {'R2':>8s} {'R3':>8s} {'R4':>8s} {'R5':>8s} {'Progreso':>10s}")
    print("  " + "-" * 75)

    for esc, hist in historico.items():
        # Buscar el ultimo resultado de R5 para este escenario
        r5_val = None
        for fase, r in sorted(resultados.items(), reverse=True):
            if r["escenario"] == esc:
                r5_val = r["media_final"]
                break
        r5_str = f"{r5_val:>8.1f}" if r5_val is not None else "     ---"
        r1 = hist.get("R1", 0)
        progreso = f"{(r5_val or hist.get('R4', 0)) - r1:+.1f}" if r1 else ""
        print(f"  {esc:<15s} {hist.get('R1',0):>8.1f} {hist.get('R2',0):>8.1f} "
              f"{hist.get('R3',0):>8.1f} {hist.get('R4',0):>8.1f} {r5_str} {progreso:>10s}")

    print()
    print("=" * 70)
    print("  RONDA 5 FINALIZADA")
    print("=" * 70)
    print()

    return resultados


def main():
    parser = argparse.ArgumentParser(description="ATLAS Pro - Ronda 5 Optimizacion")
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--fase", type=int, default=1,
                        help="Empezar desde esta fase (default: 1)")
    parser.add_argument("--ver-plan", action="store_true",
                        help="Solo mostrar el plan")
    args = parser.parse_args()

    if args.ver_plan:
        mostrar_plan(PLAN_RONDA5)
        return

    if not os.path.exists(args.config):
        print(f"ERROR: No se encontro {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Verificar MUSE
    try:
        from muse_metacognicion import MUSEController
        print("\n  MUSE v2: OK")
    except ImportError:
        print("\n  ADVERTENCIA: muse_metacognicion.py no encontrado")
        print("  Continuando sin MUSE...")

    mostrar_plan(PLAN_RONDA5)
    print("  Iniciando en 3 segundos...")
    time.sleep(3)

    ejecutar_plan(PLAN_RONDA5, base_config, desde_fase=args.fase)


if __name__ == "__main__":
    main()
