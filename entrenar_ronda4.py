"""
ATLAS Pro - Ronda 4: MUSE v2 (Corregido)
==========================================
Correcciones tras la regresion de Ronda 3:
  - MUSE intervenia demasiado (5736 fallbacks en evento)
  - Umbrales recalibrados: COMPETENCE_LOW 0.3->0.12, NOVELTY_HIGH 3.0->5.0
  - Warmup de 5000 pasos: MUSE observa antes de intervenir
  - Cooldown de 50 pasos entre fallbacks
  - balanced_rotation ahora mira colas reales
  - emergency_priority solo si competencia MUY baja
  - El agente RL de Ronda 2 ya es bueno -> MUSE debe respetar sus decisiones

Estrategia: primero recuperar scores de Ronda 2, luego superarlos.
Empezamos desde modelos de Ronda 2 (best_pro_*.pt) para no arrastrar
la regresion de Ronda 3.

Uso:
    python entrenar_ronda4.py              # Ronda completa
    python entrenar_ronda4.py --fase 3     # Empezar desde fase 3
    python entrenar_ronda4.py --ver-plan   # Solo ver plan
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np

try:
    import yaml
except ImportError:
    print("ERROR: pip install pyyaml")
    sys.exit(1)

from ejecutar_entrenamiento_pro import entrenar_pro, ESCENARIOS, MODEL_DIR


# =============================================================================
# PREPARACION: Restaurar modelos de Ronda 2 (best_pro_*)
# =============================================================================

def restaurar_modelos_ronda2():
    """
    Copia best_pro_*.pt a agente_pro_*.pt para partir de Ronda 2.
    Los modelos agente_pro_* de Ronda 3 estan degradados.
    Los best_pro_* preservan el mejor punto de Ronda 2.
    """
    print("\n  Restaurando modelos de Ronda 2 (best_pro_* -> agente_pro_*)...")
    escenarios = ["simple", "noche", "heavy", "emergencias", "evento", "avenida"]
    restaurados = 0

    for esc in escenarios:
        best = os.path.join(MODEL_DIR, f"best_pro_{esc}.pt")
        agente = os.path.join(MODEL_DIR, f"agente_pro_{esc}.pt")

        if os.path.exists(best):
            shutil.copy2(best, agente)
            size_mb = os.path.getsize(best) / (1024 * 1024)
            print(f"    {best} -> {agente} ({size_mb:.1f} MB)")
            restaurados += 1
        else:
            print(f"    {best} no encontrado (se usara modelo actual)")

    # Eliminar modelos MUSE de Ronda 3 (estaban mal calibrados)
    print("\n  Limpiando modelos MUSE de Ronda 3 (mal calibrados)...")
    for esc in escenarios:
        muse_path = os.path.join(MODEL_DIR, f"muse_{esc}.pt")
        if os.path.exists(muse_path):
            os.remove(muse_path)
            print(f"    Eliminado: {muse_path}")

    print(f"\n  {restaurados} modelos restaurados. MUSE empezara desde cero (v2).")
    return restaurados


# =============================================================================
# PLAN RONDA 4
# =============================================================================

PLAN_RONDA4 = [
    # Fase 1: Calibrar MUSE v2 en escenario facil
    {
        "fase": 1,
        "nombre": "CALIBRACION MUSE v2",
        "descripcion": "MUSE v2 observa y calibra (warmup=5000) sin intervenir",
        "escenario": "simple",
        "episodios": 200,
    },
    # Fase 2: Noche (confirmar que noche sigue bien)
    {
        "fase": 2,
        "nombre": "NOCHE + MUSE v2",
        "descripcion": "Confirmar la mejora de noche de R3 (+425) con MUSE v2",
        "escenario": "noche",
        "episodios": 200,
    },
    # Fase 3: Heavy - el escenario que bajo de 118.6 a 53.7
    {
        "fase": 3,
        "nombre": "HEAVY + MUSE v2",
        "descripcion": "Recuperar y superar heavy (objetivo: >120)",
        "escenario": "heavy",
        "episodios": 600,
    },
    # Fase 4: Emergencias
    {
        "fase": 4,
        "nombre": "EMERGENCIAS + MUSE v2",
        "descripcion": "Recuperar emergencias (objetivo: >430)",
        "escenario": "emergencias",
        "episodios": 400,
    },
    # Fase 5: Evento - el mas dificil
    {
        "fase": 5,
        "nombre": "EVENTO + MUSE v2",
        "descripcion": "Superar evento (objetivo: >0, R2 fue -7.1)",
        "escenario": "evento",
        "episodios": 700,
    },
    # Fase 6: Avenida
    {
        "fase": 6,
        "nombre": "AVENIDA + MUSE v2",
        "descripcion": "Recuperar avenida (objetivo: >240)",
        "escenario": "avenida",
        "episodios": 400,
    },
    # Fase 7: Repaso heavy final
    {
        "fase": 7,
        "nombre": "HEAVY FINAL",
        "descripcion": "Consolidar heavy con MUSE v2 calibrado",
        "escenario": "heavy",
        "episodios": 400,
    },
]


def mostrar_plan(plan):
    total_eps = sum(f["episodios"] for f in plan)
    print()
    print("=" * 70)
    print("  ATLAS Pro - RONDA 4: MUSE v2 (Corregido)")
    print("  Cambios: warmup=5000, cooldown=50, umbrales conservadores")
    print("  Partiendo de modelos BEST de Ronda 2")
    print("=" * 70)
    print()
    for f in plan:
        print(f"  Fase {f['fase']}: {f['nombre']:<25s} | "
              f"{f['escenario']:<15s} | {f['episodios']:>4d} episodios")
        print(f"          {f['descripcion']}")
    print()
    print(f"  TOTAL: {total_eps} episodios en {len(plan)} fases")
    print(f"  MUSE v2: Warmup 5000 pasos | Cooldown 50 pasos | Umbrales conservadores")
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
        print(f"#  MUSE v2: Warmup=5000 | Cooldown=50")
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
    print("  ATLAS Pro - RESUMEN RONDA 4 (MUSE v2 Corregido)")
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

    # Comparativa con Ronda 2 y Ronda 3
    ronda2 = {
        "simple": 359.1, "heavy": 118.6, "evento": -7.1,
        "emergencias": 429.5, "avenida": 238.9, "noche": 291.8
    }
    ronda3 = {
        "simple": 293.7, "heavy": 53.7, "evento": -57.8,
        "emergencias": 311.9, "avenida": 143.7, "noche": 717.2
    }

    print("  COMPARATIVA:")
    print(f"  {'Escenario':<15s} {'Ronda2':>8s} {'Ronda3':>8s} {'Ronda4':>8s} {'vs R2':>8s} {'vs R3':>8s}")
    print("  " + "-" * 60)

    for fase, r in sorted(resultados.items()):
        esc = r["escenario"]
        if esc in ronda2:
            r2 = ronda2[esc]
            r3 = ronda3.get(esc, 0)
            r4 = r["media_final"]
            diff_r2 = r4 - r2
            diff_r3 = r4 - r3
            print(f"  {esc:<15s} {r2:>8.1f} {r3:>8.1f} {r4:>8.1f} "
                  f"{diff_r2:>+8.1f} {diff_r3:>+8.1f}")

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
    print("  RONDA 4 (MUSE v2) FINALIZADA")
    print("=" * 70)
    print()

    return resultados


def main():
    parser = argparse.ArgumentParser(description="ATLAS Pro - Ronda 4 MUSE v2")
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--fase", type=int, default=1,
                        help="Empezar desde esta fase (default: 1)")
    parser.add_argument("--ver-plan", action="store_true",
                        help="Solo mostrar el plan")
    parser.add_argument("--no-restaurar", action="store_true",
                        help="No restaurar modelos de Ronda 2 (usar actuales)")
    args = parser.parse_args()

    if args.ver_plan:
        mostrar_plan(PLAN_RONDA4)
        return

    if not os.path.exists(args.config):
        print(f"ERROR: No se encontro {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Verificar MUSE
    try:
        from muse_metacognicion import MUSEController
        print("\n  MUSE v2: Modulo de metacognicion encontrado OK")
    except ImportError:
        print("\n  ERROR: No se encontro muse_metacognicion.py")
        sys.exit(1)

    # Restaurar modelos de Ronda 2 (los best_pro_*)
    if not args.no_restaurar and args.fase == 1:
        restaurar_modelos_ronda2()

    mostrar_plan(PLAN_RONDA4)
    print("  Iniciando en 3 segundos...")
    time.sleep(3)

    ejecutar_plan(PLAN_RONDA4, config, desde_fase=args.fase)


if __name__ == "__main__":
    main()
