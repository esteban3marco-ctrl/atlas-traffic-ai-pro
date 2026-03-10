"""
ATLAS Pro - Entrenamiento Completo Automatizado
=================================================
Ejecuta el curriculum completo de entrenamiento progresivo.
Cada fase se construye sobre la anterior (fine-tuning acumulativo).

Uso:
    python entrenar_todo_pro.py              # Entrenamiento completo
    python entrenar_todo_pro.py --fase 3     # Empezar desde fase 3
    python entrenar_todo_pro.py --rapido     # Version rapida (menos episodios)
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
# PLAN DE ENTRENAMIENTO PROGRESIVO
# =============================================================================

PLAN_COMPLETO = [
    # Fase 1: Fundamentos - aprender lo basico con poco trafico
    {
        "fase": 1,
        "nombre": "FUNDAMENTOS",
        "descripcion": "Aprender lo basico: que es un semaforo, como fluye el trafico",
        "escenario": "simple",
        "episodios": 300,
    },
    # Fase 2: Nocturno - trafico ligero, perfeccionar timing
    {
        "fase": 2,
        "nombre": "NOCTURNO",
        "descripcion": "Poco trafico, aprender a no desperdiciar verde innecesario",
        "escenario": "noche",
        "episodios": 150,
    },
    # Fase 3: Hora punta - el doble de trafico
    {
        "fase": 3,
        "nombre": "HORA PUNTA",
        "descripcion": "Doble de trafico, aprender gestion de colas bajo presion",
        "escenario": "heavy",
        "episodios": 400,
    },
    # Fase 4: Emergencias - priorizar ambulancias y policia
    {
        "fase": 4,
        "nombre": "EMERGENCIAS",
        "descripcion": "Muchas ambulancias/policia, aprender prioridad de paso",
        "escenario": "emergencias",
        "episodios": 250,
    },
    # Fase 5: Evento masivo - trafico extremo (concierto/partido)
    {
        "fase": 5,
        "nombre": "EVENTO MASIVO",
        "descripcion": "Trafico extremo, aprender a evitar bloqueo total",
        "escenario": "evento",
        "episodios": 300,
    },
    # Fase 6: Avenida - topologia diferente (4 carriles principales)
    {
        "fase": 6,
        "nombre": "AVENIDA",
        "descripcion": "Avenida con 4 carriles, aprender asimetria de trafico",
        "escenario": "avenida",
        "episodios": 250,
    },
    # Fase 7: Repaso final - volver a heavy para consolidar
    {
        "fase": 7,
        "nombre": "REPASO FINAL",
        "descripcion": "Volver a hora punta para consolidar todo lo aprendido",
        "escenario": "heavy",
        "episodios": 200,
    },
]

PLAN_RAPIDO = [
    {"fase": 1, "nombre": "FUNDAMENTOS",  "escenario": "simple",      "episodios": 100, "descripcion": "Basicos"},
    {"fase": 2, "nombre": "HORA PUNTA",   "escenario": "heavy",       "episodios": 150, "descripcion": "Trafico pesado"},
    {"fase": 3, "nombre": "EMERGENCIAS",  "escenario": "emergencias", "episodios": 100, "descripcion": "Prioridad emergencias"},
    {"fase": 4, "nombre": "EVENTO",       "escenario": "evento",      "episodios": 100, "descripcion": "Trafico extremo"},
    {"fase": 5, "nombre": "REPASO FINAL", "escenario": "heavy",       "episodios": 100, "descripcion": "Consolidacion"},
]


def mostrar_plan(plan):
    """Muestra el plan de entrenamiento."""
    total_eps = sum(f["episodios"] for f in plan)
    print()
    print("=" * 70)
    print("  ATLAS Pro - Plan de Entrenamiento Progresivo")
    print("=" * 70)
    print()
    for f in plan:
        print(f"  Fase {f['fase']}: {f['nombre']:<20s} | "
              f"{f['escenario']:<15s} | {f['episodios']:>4d} episodios")
        print(f"          {f.get('descripcion', '')}")
    print()
    print(f"  TOTAL: {total_eps} episodios en {len(plan)} fases")
    print("=" * 70)
    print()


def ejecutar_plan(plan, config, desde_fase=1):
    """Ejecuta el plan completo de entrenamiento."""
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
        print(f"#  {etapa.get('descripcion', '')}")
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
            mejor = 0
            media_final = 0
            mejora = 0

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
    print("  ATLAS Pro - RESUMEN DE ENTRENAMIENTO COMPLETO")
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

    # Modelos guardados
    print("  Modelos guardados:")
    if os.path.exists(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.endswith(".pt"):
                size_mb = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024*1024)
                print(f"    {f} ({size_mb:.1f} MB)")

    print()
    print("=" * 70)
    print("  ENTRENAMIENTO COMPLETO FINALIZADO")
    print("=" * 70)
    print()

    return resultados


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Pro - Entrenamiento Completo Automatizado"
    )
    parser.add_argument("--config", default="train_config.yaml",
                        help="Archivo de configuracion YAML")
    parser.add_argument("--fase", type=int, default=1,
                        help="Empezar desde esta fase (default: 1)")
    parser.add_argument("--rapido", action="store_true",
                        help="Plan rapido (menos episodios, menos fases)")
    parser.add_argument("--ver-plan", action="store_true",
                        help="Solo mostrar el plan, no ejecutar")
    args = parser.parse_args()

    plan = PLAN_RAPIDO if args.rapido else PLAN_COMPLETO

    if args.ver_plan:
        mostrar_plan(plan)
        return

    # Cargar YAML
    if not os.path.exists(args.config):
        print(f"ERROR: No se encontro {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mostrar_plan(plan)

    print("  Iniciando en 3 segundos...")
    time.sleep(3)

    ejecutar_plan(plan, config, desde_fase=args.fase)


if __name__ == "__main__":
    main()
