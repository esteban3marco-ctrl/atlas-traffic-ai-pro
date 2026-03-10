"""
ATLAS Pro - Ejecutor de Entrenamiento con Configuracion YAML
=============================================================
Conecta el motor de IA avanzado (Dueling DDQN + PER + Noisy Networks)
con el simulador SUMO usando los parametros de train_config.yaml.

Uso:
    python ejecutar_entrenamiento_pro.py                    # Normal (simple)
    python ejecutar_entrenamiento_pro.py --episodes 50      # Episodios custom
    python ejecutar_entrenamiento_pro.py --scenario heavy    # Trafico pesado
    python ejecutar_entrenamiento_pro.py --gui               # Con interfaz
"""

import os
import sys
import time
import argparse
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ATLAS_PRO")

# --- Dependencias ---
try:
    import yaml
except ImportError:
    print("ERROR: Instala pyyaml con: pip install pyyaml")
    sys.exit(1)

try:
    import traci
except ImportError:
    print("ERROR: Instala traci con: pip install traci")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: Instala PyTorch. Ver https://pytorch.org/get-started/locally/")
    sys.exit(1)

from algoritmos_avanzados import AgenteDuelingDDQN

# MUSE (opcional)
try:
    from muse_metacognicion import MUSEController
    MUSE_DISPONIBLE = True
except ImportError:
    MUSE_DISPONIBLE = False


# =============================================================================
# ESCENARIOS DISPONIBLES
# =============================================================================

ESCENARIOS = {
    "simple":       "simulations/simple/simulation.sumocfg",
    "heavy":        "simulations/simple_hora_punta/simulation.sumocfg",
    "hora_punta":   "simulations/simple_hora_punta/simulation.sumocfg",
    "noche":        "simulations/simple_noche/simulation.sumocfg",
    "emergencias":  "simulations/simple_emergencias/simulation.sumocfg",
    "avenida":      "simulations/avenida/simulation.sumocfg",
    "cruce_t":      "simulations/cruce_t/simulation.sumocfg",
    "doble":        "simulations/doble/simulation.sumocfg",
    "complejo":     "simulations/complejo/simulation.sumocfg",
    "evento":       "simulations/evento/simulation.sumocfg",
}

# Ruta absoluta para evitar [WinError 433] si el CWD cambia o el disco tiene hiccups
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")


# =============================================================================
# ENTORNO SUMO PRO (26 dimensiones de estado)
# =============================================================================

class EntornoSUMO_Pro:
    """
    Wrapper de SUMO que extrae estados de 26 dimensiones para ATLAS Pro.

    Vector de estado (26D):
      Por cada direccion (N, S, E, W) -> 6 features = 24
        0: cola (vehiculos parados) / 30
        1: velocidad media / 15
        2: tiempo espera / 300
        3: num vehiculos / 30
        4: densidad (ocupacion) / 100
        5: hay_emergencia (0 o 1)
      + fase_actual_normalizada (0-1)
      + paso_normalizado (0-1)
    """

    def __init__(self, cfg_file, gui=False, config=None):
        self.cfg_file = cfg_file
        self.gui = gui
        self.config = config or {}

        # Parametros del entorno
        env_cfg = self.config.get("environment", {})
        self.semaforo_id = (env_cfg.get("traffic_light_ids") or ["center"])[0]
        self.max_steps = env_cfg.get("max_steps", 1000)
        self.delta_time = env_cfg.get("delta_time", 10)
        self.min_green = env_cfg.get("min_green_time", 10)
        self.max_green = env_cfg.get("max_green_time", 60)
        self.yellow_time = env_cfg.get("yellow_time", 3)
        self.warmup_steps = env_cfg.get("warmup_steps", 50)

        # Edges de entrada
        edges_cfg = env_cfg.get("edges", {})
        self.edges = {
            "N": edges_cfg.get("N", "north_in"),
            "S": edges_cfg.get("S", "south_in"),
            "E": edges_cfg.get("E", "east_in"),
            "W": edges_cfg.get("W", "west_in"),
        }

        # Pesos de recompensa (desde YAML)
        rw = self.config.get("reward", {})
        self.rw_queue   = rw.get("queue_length_weight", -0.25)
        self.rw_wait    = rw.get("wait_time_weight", -0.15)
        self.rw_through = rw.get("throughput_weight", 0.5)
        self.rw_speed   = rw.get("speed_weight", 0.1)
        self.rw_phase   = rw.get("phase_change_penalty", -0.8)
        self.rw_emerg   = rw.get("emergency_weight", -5.0)
        self.rw_clip_min = rw.get("clip_min", -10.0)
        self.rw_clip_max = rw.get("clip_max", 10.0)

        # Estado interno
        self.conectado = False
        self.paso = 0
        self.fase_actual = 0
        self.pasos_en_fase = 0
        self.prev_arrived = 0
        self.last_phase_change = False
        self._lane_ids_cache = None

    # -- Conexion --

    def reset(self):
        """Inicia o reinicia una simulacion. Devuelve estado inicial."""
        if self.conectado:
            try:
                traci.close()
            except Exception:
                pass

        sumo_cmd = "sumo-gui" if self.gui else "sumo"
        cmd = [
            sumo_cmd, "-c", self.cfg_file,
            "--step-length", "1",
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
        ]
        traci.start(cmd)
        self.conectado = True
        self.paso = 0
        self.fase_actual = 0
        self.pasos_en_fase = 0
        self.prev_arrived = 0
        self.last_phase_change = False
        self._lane_ids_cache = None

        # Warmup
        for _ in range(self.warmup_steps):
            traci.simulationStep()
            self.paso += 1

        return self._get_state()

    def close(self):
        if self.conectado:
            try:
                traci.close()
            except Exception:
                pass
            self.conectado = False

    # -- Lanes helper --

    def _get_lane_ids(self, edge):
        """Obtiene IDs de carriles validos para un edge."""
        if self._lane_ids_cache is None:
            self._lane_ids_cache = set(traci.lane.getIDList())
        lanes = []
        for i in range(4):
            lid = f"{edge}_{i}"
            if lid in self._lane_ids_cache:
                lanes.append(lid)
        return lanes

    # -- Estado (26D) --

    def _get_state(self):
        state = []

        for d in ["N", "S", "E", "W"]:
            edge = self.edges[d]
            lanes = self._get_lane_ids(edge)

            # Cola (vehiculos parados)
            try:
                cola = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            except Exception:
                cola = 0
            state.append(cola / 30.0)

            # Velocidad media
            try:
                vel = traci.edge.getLastStepMeanSpeed(edge)
            except Exception:
                vel = 0
            state.append(vel / 15.0)

            # Tiempo de espera
            try:
                espera = traci.edge.getWaitingTime(edge)
            except Exception:
                espera = 0
            state.append(min(espera, 300) / 300.0)

            # Numero de vehiculos
            try:
                n_vehs = traci.edge.getLastStepVehicleNumber(edge)
            except Exception:
                n_vehs = 0
            state.append(n_vehs / 30.0)

            # Densidad (ocupacion %)
            try:
                ocupacion = traci.edge.getLastStepOccupancy(edge)
            except Exception:
                ocupacion = 0
            state.append(ocupacion / 100.0)

            # Vehiculo de emergencia presente
            try:
                vehs = traci.edge.getLastStepVehicleIDs(edge)
                has_emerg = 0
                for v in vehs:
                    vtype = traci.vehicle.getTypeID(v)
                    if vtype in ("emergencia", "ambulancia", "policia", "bomberos"):
                        has_emerg = 1
                        break
            except Exception:
                has_emerg = 0
            state.append(float(has_emerg))

        # Fase actual normalizada
        state.append(self.fase_actual / 3.0)
        # Paso normalizado
        state.append(min(self.paso, self.max_steps) / self.max_steps)

        return np.array(state, dtype=np.float32)

    # -- Accion --

    def step(self, action):
        """
        Ejecuta accion y devuelve (next_state, reward, done).
        Acciones: 0=Mantener, 1=Fase N-S, 2=Fase E-O, 3=Extender
        """
        self.last_phase_change = False

        if action == 1 and self.fase_actual != 0 and self.pasos_en_fase >= self.min_green:
            self._cambiar_fase(0)
            self.last_phase_change = True
        elif action == 2 and self.fase_actual != 2 and self.pasos_en_fase >= self.min_green:
            self._cambiar_fase(2)
            self.last_phase_change = True

        # Forzar cambio si excede maximo
        if self.pasos_en_fase >= self.max_green * 10:
            nueva = 2 if self.fase_actual == 0 else 0
            self._cambiar_fase(nueva)
            self.last_phase_change = True

        # Avanzar delta_time pasos
        for _ in range(self.delta_time):
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
            self.paso += 1
            self.pasos_en_fase += 1

        reward = self._calc_reward()
        done = (self.paso >= self.max_steps) or (traci.simulation.getMinExpectedNumber() <= 0)
        next_state = self._get_state()

        return next_state, reward, done

    def _cambiar_fase(self, nueva_fase):
        """Cambia fase con transicion de amarillo."""
        try:
            traci.trafficlight.setPhase(self.semaforo_id, self.fase_actual + 1)
            for _ in range(self.yellow_time * 10):
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
                traci.simulationStep()
                self.paso += 1
            traci.trafficlight.setPhase(self.semaforo_id, nueva_fase)
            self.fase_actual = nueva_fase
            self.pasos_en_fase = 0
        except Exception:
            pass

    # -- Recompensa --

    def _calc_reward(self):
        """Calcula recompensa usando pesos del YAML."""
        total_queue = 0
        total_wait = 0
        total_speed = 0
        emerg_waiting = 0

        for edge in self.edges.values():
            lanes = self._get_lane_ids(edge)
            try:
                total_queue += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            except Exception:
                pass

            try:
                total_wait += traci.edge.getWaitingTime(edge)
            except Exception:
                pass

            try:
                total_speed += traci.edge.getLastStepMeanSpeed(edge)
            except Exception:
                pass

            # Emergencias esperando
            try:
                vehs = traci.edge.getLastStepVehicleIDs(edge)
                for v in vehs:
                    vtype = traci.vehicle.getTypeID(v)
                    if vtype in ("emergencia", "ambulancia", "policia", "bomberos"):
                        if traci.vehicle.getWaitingTime(v) > 5:
                            emerg_waiting += 1
            except Exception:
                pass

        # Throughput
        arrived = traci.simulation.getArrivedNumber()
        throughput = arrived - self.prev_arrived
        self.prev_arrived = arrived

        # Componer recompensa
        reward = 0.0
        reward += self.rw_queue * total_queue
        reward += self.rw_wait * (total_wait / 100.0)
        reward += self.rw_through * throughput
        reward += self.rw_speed * (total_speed / 4.0)
        reward += self.rw_emerg * emerg_waiting

        if self.last_phase_change:
            reward += self.rw_phase

        return np.clip(reward, self.rw_clip_min, self.rw_clip_max)


# =============================================================================
# LOOP DE ENTRENAMIENTO
# =============================================================================

def entrenar_pro(config, scenario="simple", num_episodes=None, gui=False,
                 use_muse=False):
    """Ejecuta entrenamiento ATLAS Pro, opcionalmente con MUSE."""

    cfg_file = ESCENARIOS.get(scenario)
    if not cfg_file:
        print(f"ERROR: Escenario '{scenario}' no encontrado.")
        print(f"  Disponibles: {list(ESCENARIOS.keys())}")
        return
    if not os.path.exists(cfg_file):
        print(f"ERROR: No existe {cfg_file}")
        print(f"  Ejecuta primero: python crear_todo.py")
        return

    episodes = num_episodes or config.get("total_episodes", 50)
    agent_cfg = config.get("agent", {})

    # Crear agente (26D estado, 4 acciones)
    agent = AgenteDuelingDDQN(config=agent_cfg)

    # Fine-tuning: cargar modelo previo si existe
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"agente_pro_{scenario}.pt")

    if agent.load(model_path):
        logger.info(f"MODELO CARGADO: {model_path} (Fine-tuning)")
    else:
        logger.info(f"Modelo nuevo (no se encontro {model_path})")

    # MUSE: inicializar metacognicion
    muse = None
    muse_path = os.path.join(MODEL_DIR, f"muse_{scenario}.pt")
    if use_muse and MUSE_DISPONIBLE:
        muse = MUSEController(agent, config)
        if muse.load(muse_path):
            logger.info(f"MUSE CARGADO: {muse_path}")
        else:
            logger.info("MUSE iniciado desde cero")
    elif use_muse and not MUSE_DISPONIBLE:
        logger.warning("MUSE solicitado pero no disponible (falta muse_metacognicion.py)")

    # Crear entorno Pro (26D)
    env = EntornoSUMO_Pro(cfg_file, gui=gui, config=config)

    # Header
    print()
    print("=" * 70)
    if muse:
        print("  ATLAS Pro + MUSE - Entrenamiento con Metacognicion")
    else:
        print("  ATLAS Pro - Entrenamiento Avanzado")
    print("=" * 70)
    print(f"  Escenario:   {scenario} ({cfg_file})")
    print(f"  Episodios:   {episodes}")
    print(f"  Algoritmo:   {agent_cfg.get('algorithm', 'dueling_ddqn')}")
    print(f"  Red:         {agent_cfg.get('hidden_dims', [256,256,128])}")
    print(f"  LR:          {agent_cfg.get('lr', 0.0003)}")
    print(f"  Noisy Nets:  {agent_cfg.get('use_noisy_nets', True)}")
    print(f"  N-Step:      {agent_cfg.get('n_step', 3)}")
    print(f"  Buffer:      {agent_cfg.get('buffer_size', 100000)}")
    print(f"  Device:      {agent.device}")
    print(f"  Modelo:      {model_path}")
    if muse:
        print(f"  MUSE:        ACTIVADO ({muse_path})")
    print("=" * 70)
    print()

    # Training loop
    all_rewards = []
    best_reward = -float("inf")
    save_interval = config.get("save_interval", 5)

    # MUSE stats
    muse_fallbacks = 0
    muse_novel_count = 0

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            total_loss = 0.0
            loss_count = 0
            step_count = 0
            ep_fallbacks = 0
            ep_novels = 0

            while True:
                # Seleccionar accion: con MUSE o sin
                if muse:
                    action = muse.act(state)
                    diag = muse.last_diagnosis
                    if diag["strategy"] != "rl_agent":
                        ep_fallbacks += 1
                    if diag["is_novel"]:
                        ep_novels += 1
                else:
                    action = agent.select_action(state)

                next_state, reward, done = env.step(action)

                # Almacenar transicion y entrenar agente RL
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()

                # MUSE: observar resultado
                if muse:
                    muse.observe(state, action, reward, next_state, done)

                total_reward += reward
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

                state = next_state
                step_count += 1

                # Progreso cada 50 pasos
                if step_count % 50 == 0:
                    avg_l = total_loss / max(loss_count, 1)
                    if muse:
                        diag = muse.last_diagnosis
                        print(f"  Ep {ep:3d} | Step {step_count:4d} | "
                              f"Reward: {total_reward:8.1f} | "
                              f"Loss: {avg_l:.4f} | "
                              f"Comp: {diag['competence']:.2f} "
                              f"Nov: {diag['novelty']:.2f} "
                              f"Strat: {diag['strategy']}")
                    else:
                        print(f"  Ep {ep:3d} | Step {step_count:4d} | "
                              f"Reward: {total_reward:8.1f} | "
                              f"Loss: {avg_l:.4f}")

                if done:
                    break

            env.close()
            all_rewards.append(total_reward)
            avg_loss = total_loss / max(loss_count, 1)
            muse_fallbacks += ep_fallbacks
            muse_novel_count += ep_novels

            # Media reciente
            recent = all_rewards[-10:] if len(all_rewards) >= 10 else all_rewards
            avg_recent = np.mean(recent)

            print(f"\n>>> RESULTADO EPISODIO {ep}/{episodes}: "
                  f"Reward = {total_reward:.1f} | Loss = {avg_loss:.4f}")
            if muse:
                print(f">>> MUSE: fallbacks={ep_fallbacks} | "
                      f"situaciones_nuevas={ep_novels} | "
                      f"competencia={muse.last_diagnosis['competence']:.3f}")
            print(f">>> MEDIA RECIENTE ({len(recent)} eps): {avg_recent:.1f}")
            print()

            # Guardar periodicamente (con proteccion contra fallos de disco)
            if ep % save_interval == 0:
                try:
                    agent.save(model_path)
                    if muse:
                        muse.save(muse_path)
                    logger.info(f"Progreso guardado en {model_path}")
                except OSError as save_err:
                    logger.warning(f"No se pudo guardar (disco?): {save_err}")
                    logger.warning("Continuando entrenamiento sin guardar...")

            # Mejor modelo
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = os.path.join(MODEL_DIR, f"best_pro_{scenario}.pt")
                try:
                    agent.save(best_path)
                except OSError:
                    pass

    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

    # Guardar final (reintentar hasta 3 veces)
    for intento in range(3):
        try:
            agent.save(model_path)
            if muse:
                muse.save(muse_path)
            break
        except OSError as e:
            logger.warning(f"Guardado final intento {intento+1}/3 fallo: {e}")
            time.sleep(2)

    print()
    print("=" * 70)
    print(f"  ENTRENAMIENTO COMPLETADO")
    print(f"  Episodios:       {len(all_rewards)}")
    if all_rewards:
        print(f"  Mejor Reward:    {best_reward:.1f}")
        print(f"  Media Final:     {np.mean(all_rewards[-10:]):.1f}")
    print(f"  Modelo guardado: {model_path}")
    if muse:
        diag = muse.get_diagnosis()
        print(f"  MUSE guardado:   {muse_path}")
        print(f"  MUSE Stats:")
        print(f"    Total fallbacks:     {muse_fallbacks}")
        print(f"    Situaciones nuevas:  {muse_novel_count}")
        print(f"    Confianza final:     {diag['performance']['confidence']:.3f}")
        strats = diag['strategy_stats']['percentages']
        for s, p in strats.items():
            if p > 0:
                print(f"    Estrategia {s}: {p}%")
    print("=" * 70)
    print()

    return all_rewards


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ATLAS Pro - Entrenamiento RL")
    parser.add_argument("--config", default="train_config.yaml",
                        help="Archivo de configuracion YAML")
    parser.add_argument("--scenario", default="simple",
                        choices=list(ESCENARIOS.keys()),
                        help="Escenario de simulacion")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Numero de episodios (sobreescribe YAML)")
    parser.add_argument("--gui", action="store_true",
                        help="Mostrar interfaz grafica de SUMO")
    parser.add_argument("--muse", action="store_true",
                        help="Activar metacognicion MUSE (auto-evaluacion + fallback)")
    args = parser.parse_args()

    # Cargar YAML
    if not os.path.exists(args.config):
        print(f"ERROR: No se encontro {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"\n  Configuracion cargada: {args.config}")
    if args.muse:
        if MUSE_DISPONIBLE:
            print("  MUSE: ACTIVADO (metacognicion habilitada)")
        else:
            print("  ADVERTENCIA: --muse solicitado pero muse_metacognicion.py no encontrado")
            print("  Continuando sin MUSE...")

    entrenar_pro(
        config=config,
        scenario=args.scenario,
        num_episodes=args.episodes,
        gui=args.gui,
        use_muse=args.muse,
    )


if __name__ == "__main__":
    main()
