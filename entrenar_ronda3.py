"""
ATLAS Pro - Ronda 3: Metacognicion con MUSE
=============================================
Entrenamiento con MUSE activado para que el agente desarrolle:
  - Auto-evaluacion de competencia en cada situacion
  - Deteccion de situaciones nuevas/anomalas
  - Fallback a estrategias seguras cuando no sabe que hacer
  - Modelo del mundo para anticipar consecuencias

Partimos de los modelos de Ronda 2 (red [512,256,256,128]).
MUSE se entrena en paralelo al agente RL.

Uso:
    python entrenar_ronda3.py           # Ronda completa con MUSE
    python entrenar_ronda3.py --fase 3  # Empezar desde fase 3
    python entrenar_ronda3.py --ver-plan
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
# PLAN RONDA 3: Aprendizaje con Metacognicion MUSE
# =============================================================================
# Estrategia: empezar por escenarios conocidos para calibrar competencia,
# luego ir a los dificiles donde MUSE realmente brilla con fallbacks.

PLAN_RONDA3 = [
    # Fase 1: Calibrar MUSE con escenario facil
    {
        "fase": 1,
        "nombre": "CALIBRACION MUSE",
        "descripcion": "MUSE aprende que es 'normal' - calibra competencia y modelo del mundo",
        "escenario": "simple",
        "episodios": 200,
    },
    # Fase 2: MUSE en noche (poco trafico, decision de no actuar)
    {
        "fase": 2,
        "nombre": "NOCHE + MUSE",
        "descripcion": "MUSE aprende a no intervenir innecesariamente",
        "escenario": "noche",
        "episodios": 150,
    },
    # Fase 3: Hora punta - aqui MUSE detecta presion y posibles anomalias
    {
        "fase": 3,
        "nombre": "HORA PUNTA + MUSE",
        "descripcion": "MUSE detecta alta presion y activa estrategias adaptativas",
        "escenario": "heavy",
        "episodios": 500,
    },
    # Fase 4: Emergencias - MUSE debe priorizar emergency_priority
    {
        "fase": 4,
        "nombre": "EMERGENCIAS + MUSE",
        "descripcion": "MUSE aprende cuando activar fallback de prioridad emergencias",
        "escenario": "emergencias",
        "episodios": 400,
    },
    # Fase 5: Evento masivo - el escenario mas dificil, MUSE es clave
    {
        "fase": 5,
        "nombre": "EVENTO + MUSE",
        "descripcion": "Trafico extremo: MUSE detecta novedad y usa fallbacks inteligentes",
        "escenario": "evento",
        "episodios": 600,
    },
    # Fase 6: Avenida - topologia diferente, MUSE detecta novedad
    {
        "fase": 6,
        "nombre": "AVENIDA + MUSE",
        "descripcion": "Topologia nueva: MUSE evalua competencia en entorno diferente",
        "escenario": "avenida",
        "episodios": 300,
    },
    # Fase 7: Repaso heavy - consolidar con MUSE calibrado
    {
        "fase": 7,
        "nombre": "HEAVY FINAL + MUSE",
        "descripcion": "Consolidar hora punta con MUSE ya experimentado",
        "escenario": "heavy",
        "episodios": 400,
    },
    # Fase 8: Repaso evento - el test definitivo
    {
        "fase": 8,
        "nombre": "EVENTO FINAL + MUSE",
        "descripcion": "Test definitivo: evento masivo con MUSE completamente entrenado",
        "escenario": "evento",
        "episodios": 400,
    },
]


def mostrar_plan(plan):
    total_eps = sum(f["episodios"] for f in plan)
    print()
    print("=" * 70)
    print("  ATLAS Pro - RONDA 3: Metacognicion con MUSE")
    print("  Red: [512, 256, 256, 128] | Max steps: 2000 | MUSE: ON")
    print("=" * 70)
    print()
    for f in plan:
        print(f"  Fase {f['fase']}: {f['nombre']:<25s} | "
              f"{f['escenario']:<15s} | {f['episodios']:>4d} episodios")
        print(f"          {f['descripcion']}")
    print()
    print(f"  TOTAL: {total_eps} episodios en {len(plan)} fases")
    print(f"  MUSE: Metacognicion activada en TODAS las fases")
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
        print(f"#  MUSE: ACTIVADO")
        print("#" * 70)

        inicio_fase = time.time()

        rewards = entrenar_pro(
            config=config,
            scenario=etapa["escenario"],
            num_episodes=etapa["episodios"],
            gui=False,
            use_muse=True,  # <-- MUSE siempre activado en Ronda 3
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
    print("  ATLAS Pro - RESUMEN RONDA 3 (con MUSE)")
    print("=" * 70)
    print()
    print(f"  {'Fase':<5s} {'Nombre':<25s} {'Escenario':<15s} "
          f"{'Mejor':>8s} {'Media':>8s} {'Mejora':>8s} {'Tiempo':>8s}")
    print("  " + "-" * 80)

    for fase, r in sorted(resultados.items()):
        print(f"  {fase:<5d} {r['nombre']:<25s} {r['escenario']:<15s} "
              f"{r['mejor_reward']:>8.1f} {r['media_final']:>8.1f} "
              f"{r['mejora']:>+8.1f} {r['duracion_min']:>7.1f}m")

    print()
    print(f"  Tiempo total: {duracion_total/60:.1f} minutos ({duracion_total/3600:.1f} horas)")
    print()

    # Comparativa con Ronda 2
    print("  COMPARATIVA CON RONDA 2:")
    ronda2 = {
        "simple": 359.1, "heavy": 118.6, "evento": -7.1,
        "emergencias": 429.5, "avenida": 238.9, "noche": 291.8
    }
    for fase, r in sorted(resultados.items()):
        esc = r["escenario"]
        if esc in ronda2:
            diff = r["media_final"] - ronda2[esc]
            signo = "+" if diff > 0 else ""
            print(f"    {esc:<15s}: Ronda2={ronda2[esc]:>8.1f} -> "
                  f"Ronda3={r['media_final']:>8.1f} ({signo}{diff:.1f})")
    print()

    # Modelos guardados
    print("  Modelos guardados:")
    if os.path.exists(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.endswith((".pt", ".muse")):
                size_mb = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024*1024)
                print(f"    {f} ({size_mb:.1f} MB)")

    print()
    print("=" * 70)
    print("  RONDA 3 CON MUSE FINALIZADA")
    print("=" * 70)
    print()

    return resultados


def main():
    parser = argparse.ArgumentParser(description="ATLAS Pro - Ronda 3 con MUSE")
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--fase", type=int, default=1,
                        help="Empezar desde esta fase (default: 1)")
    parser.add_argument("--ver-plan", action="store_true",
                        help="Solo mostrar el plan")
    args = parser.parse_args()

    if args.ver_plan:
        mostrar_plan(PLAN_RONDA3)
        return

    if not os.path.exists(args.config):
        print(f"ERROR: No se encontro {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Verificar que MUSE esta disponible
    try:
        from muse_metacognicion import MUSEController
        print("\n  MUSE: Modulo de metacognicion encontrado OK")
    except ImportError:
        print("\n  ERROR: No se encontro muse_metacognicion.py")
        print("  Ronda 3 REQUIERE MUSE. Verifica que el archivo existe.")
        sys.exit(1)

    mostrar_plan(PLAN_RONDA3)
    print("  Iniciando en 3 segundos...")
    time.sleep(3)

    ejecutar_plan(PLAN_RONDA3, config, desde_fase=args.fase)


if __name__ == "__main__":
    main()
