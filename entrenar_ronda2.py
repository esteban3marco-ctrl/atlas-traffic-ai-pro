"""
ATLAS Pro - Ronda 2: Romper el techo en escenarios dificiles
=============================================================
Red neuronal mas grande [512, 256, 256, 128] + episodios mas largos (2000 pasos).
IMPORTANTE: Al cambiar la arquitectura, los modelos anteriores NO son compatibles.
Se entrena desde cero con la nueva red, pero converge mas rapido y mas alto.

Uso:
    python entrenar_ronda2.py           # Ronda completa
    python entrenar_ronda2.py --fase 3  # Empezar desde fase 3
"""

import os
import sys
import time
import argparse
import numpy as np

try:
    import yaml
except ImportError:
    print("ERROR: pip install pyyaml")
    sys.exit(1)

from ejecutar_entrenamiento_pro import entrenar_pro, ESCENARIOS, MODEL_DIR


# =============================================================================
# PLAN RONDA 2: enfocado en los escenarios dificiles
# =============================================================================

PLAN_RONDA2 = [
    # Fase 1: Base solida con la nueva red
    {
        "fase": 1,
        "nombre": "BASE NUEVA RED",
        "descripcion": "Entrenar fundamentos con red [512,256,256,128]",
        "escenario": "simple",
        "episodios": 400,
    },
    # Fase 2: Hora punta - objetivo: cruzar a positivo
    {
        "fase": 2,
        "nombre": "HORA PUNTA v2",
        "descripcion": "Objetivo: romper el techo de -35 y llegar a positivo",
        "escenario": "heavy",
        "episodios": 600,
    },
    # Fase 3: Evento - objetivo: cruzar a positivo
    {
        "fase": 3,
        "nombre": "EVENTO v2",
        "descripcion": "Objetivo: romper el techo de -73 y llegar a positivo",
        "escenario": "evento",
        "episodios": 600,
    },
    # Fase 4: Emergencias - consolidar
    {
        "fase": 4,
        "nombre": "EMERGENCIAS v2",
        "descripcion": "Consolidar gestion de emergencias con red grande",
        "escenario": "emergencias",
        "episodios": 300,
    },
    # Fase 5: Avenida - consolidar
    {
        "fase": 5,
        "nombre": "AVENIDA v2",
        "descripcion": "Consolidar avenida con red grande",
        "escenario": "avenida",
        "episodios": 300,
    },
    # Fase 6: Repaso heavy final
    {
        "fase": 6,
        "nombre": "HEAVY FINAL",
        "descripcion": "Pulir rendimiento en hora punta con toda la experiencia",
        "escenario": "heavy",
        "episodios": 400,
    },
    # Fase 7: Repaso evento final
    {
        "fase": 7,
        "nombre": "EVENTO FINAL",
        "descripcion": "Pulir rendimiento en evento con toda la experiencia",
        "escenario": "evento",
        "episodios": 400,
    },
]


def mostrar_plan(plan):
    total_eps = sum(f["episodios"] for f in plan)
    print()
    print("=" * 70)
    print("  ATLAS Pro - RONDA 2: Romper el Techo")
    print("  Red: [512, 256, 256, 128] | Max steps: 2000")
    print("=" * 70)
    print()
    for f in plan:
        print(f"  Fase {f['fase']}: {f['nombre']:<20s} | "
              f"{f['escenario']:<15s} | {f['episodios']:>4d} episodios")
        print(f"          {f['descripcion']}")
    print()
    print(f"  TOTAL: {total_eps} episodios en {len(plan)} fases")
    print("=" * 70)
    print()


def ejecutar_plan(plan, config, desde_fase=1):
    resultados = {}
    inicio_total = time.time()

    for etapa in plan:
        fase = etapa["fase"]
        if fase < desde_fase:
            print(f"\n  [Saltando Fase {fase}: {etapa['nombre']}]")
            continue

        print()
        print("#" * 70)
        print(f"#  FASE {fase}/{len(plan)}: {etapa['nombre']}")
        print(f"#  Escenario: {etapa['escenario']} | Episodios: {etapa['episodios']}")
        print(f"#  {etapa['descripcion']}")
        print("#" * 70)

        inicio_fase = time.time()

        rewards = entrenar_pro(
            config=config,
            scenario=etapa["escenario"],
            num_episodes=etapa["episodios"],
            gui=False,
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
    print("  ATLAS Pro - RESUMEN RONDA 2")
    print("=" * 70)
    print()
    print(f"  {'Fase':<5s} {'Nombre':<20s} {'Escenario':<15s} "
          f"{'Mejor':>8s} {'Media':>8s} {'Mejora':>8s} {'Tiempo':>8s}")
    print("  " + "-" * 72)

    for fase, r in sorted(resultados.items()):
        print(f"  {fase:<5d} {r['nombre']:<20s} {r['escenario']:<15s} "
              f"{r['mejor_reward']:>8.1f} {r['media_final']:>8.1f} "
              f"{r['mejora']:>+8.1f} {r['duracion_min']:>7.1f}m")

    print()
    print(f"  Tiempo total: {duracion_total/60:.1f} minutos ({duracion_total/3600:.1f} horas)")
    print()

    # Comparativa con Ronda 1
    print("  COMPARATIVA CON RONDA 1:")
    ronda1 = {
        "simple": 76.9, "noche": 291.8, "heavy": -35.8,
        "emergencias": 118.9, "evento": -73.2, "avenida": 15.3
    }
    for fase, r in sorted(resultados.items()):
        esc = r["escenario"]
        if esc in ronda1:
            diff = r["media_final"] - ronda1[esc]
            signo = "+" if diff > 0 else ""
            print(f"    {esc:<15s}: Ronda1={ronda1[esc]:>8.1f} -> Ronda2={r['media_final']:>8.1f} ({signo}{diff:.1f})")
    print()

    # Modelos
    print("  Modelos guardados:")
    if os.path.exists(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.endswith(".pt"):
                size_mb = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024*1024)
                print(f"    {f} ({size_mb:.1f} MB)")

    print()
    print("=" * 70)
    print("  RONDA 2 FINALIZADA")
    print("=" * 70)
    print()

    return resultados


def main():
    parser = argparse.ArgumentParser(description="ATLAS Pro - Ronda 2")
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--fase", type=int, default=1,
                        help="Empezar desde esta fase (default: 1)")
    parser.add_argument("--ver-plan", action="store_true",
                        help="Solo mostrar el plan")
    args = parser.parse_args()

    if args.ver_plan:
        mostrar_plan(PLAN_RONDA2)
        return

    if not os.path.exists(args.config):
        print(f"ERROR: No se encontro {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mostrar_plan(PLAN_RONDA2)
    print("  Iniciando en 3 segundos...")
    time.sleep(3)

    ejecutar_plan(PLAN_RONDA2, config, desde_fase=args.fase)


if __name__ == "__main__":
    main()
