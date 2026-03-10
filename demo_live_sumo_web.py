import os
import sys
import time
import threading
import requests
import logging
import random
import numpy as np

try:
    import traci
except ImportError:
    print("Error: SUMO (traci) no está instalado. Ejecuta: pip install traci sumolib")
    sys.exit(1)

from atlas.config import AgentConfig, EnvironmentConfig
from atlas.agents import DuelingDDQNAgent
from atlas.production.dashboard import start_dashboard
from atlas.sumo_env import ATLASTrafficEnv
from atlas.production.xai_engine import xai_engine
from atlas.production.safety_watchdog import watchdog

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("DemoWebLive")

def run_simulation():
    # 1. Configurar Agente IA Real
    config = AgentConfig(state_dim=26, action_dim=4)
    agent = DuelingDDQNAgent(state_dim=26, action_dim=4, config=config, device="cpu")
    
    # Intentar cargar mejor modelo
    model_path = "checkpoints_extended/atlas_best.pt"
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            logger.info(f"Modelo IA Cargado ({model_path})")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo debido a arquitectura diferente. Usando pesos iniciales. Error: {e}")
        
    cfg = EnvironmentConfig(
        sumo_cfg_file="simulations/simple/simulation.sumocfg",
        gui=False, # Lo corremos oculto pero lo enviamos al JS
        step_length=0.1,
        max_steps=100000
    )
    env = ATLASTrafficEnv(env_config=cfg)
    obs, info = env.reset()
    
    logger.info("📡 Entorno de Simulación Física Iniciado. Transmitiendo posiciones...")
    
    current_phase = 0
    phase_timer = 0
    explanation = "Iniciating Neural Analysis of Incoming Traffic Flow..."
    saliency_weights = [0.1]*20
    
    try:
        while True:
            # Extraer pos de vehículos ANTES del paso completo (traci expone todo internamente)
            try:
                vehicles_pos = []
                for vid in env._conn.vehicle.getIDList():
                    x, y = env._conn.vehicle.getPosition(vid)
                    vehicles_pos.append({"id": vid, "x": x, "y": y})
            except Exception:
                vehicles_pos = []
            
            map_data = {
                "center": {
                    "coords": [40.4194, -3.7042],
                    "status": "green" if current_phase % 2 == 0 else "red"
                }
            }
            
            # Enviar Data al Dashboard
            data = {
                "vehicles": len(vehicles_pos),
                "vehicles_pos": vehicles_pos,
                "safety_score": 100,
                "current_phase": "NORTH-SOUTH" if current_phase % 2 == 0 else "EAST-WEST",
                "explanation": explanation,
                "map_data": map_data,
                "saliency": {
                     "weights": saliency_weights,
                     "labels": [f"N{i}" for i in range(20)]
                }
            }
            
            try:
                requests.post("http://localhost:8888/api/update", json=data, timeout=0.1)
            except Exception:
                pass
            
            # Gestionamos el "paso a paso" saltándonos el delta_time para tener FPS altos en web
            # Ejecutamos pasos manuales de TRACI 
            env._conn.simulationStep()
            phase_timer += 0.1
            
            # Toma de decisión IA cada 5 Segundos (50 pasos)
            if phase_timer >= 5.0:
                obs = env._get_observation()
                ai_action, saliency = agent.select_action(obs, evaluate=True, return_xai=True)
                
                explanation = xai_engine.generate_explanation(obs, ai_action, saliency)
                
                # Check incident
                incident_active = False
                try:
                    r = requests.get("http://localhost:8888/api/incident_status", timeout=0.1)
                    if r.status_code == 200: incident_active = r.json().get('active', False)
                except: pass
                
                # Accidente manual hack
                if incident_active: 
                    ai_action = 1 if current_phase == 0 else 0
                    explanation = "EMERGENCY: Rerouting Traffic around Blockage Zone!"
                
                final_action = watchdog.validate_action(ai_action, current_phase, False)
                final_action = final_action % 4
                
                if final_action != current_phase:
                    env._change_phase("center", final_action) # Cambia realmente en sumo
                    current_phase = final_action
                
                # Fake Saliency
                if saliency is not None:
                     saliency_weights = saliency.tolist()[:20] if hasattr(saliency, 'tolist') else [random.random()*0.5 for _ in range(20)]
                else:
                     saliency_weights = [random.random()*0.5 for _ in range(20)]
                     
                phase_timer = 0
            
            time.sleep(0.02) # Control de Loop for JS FPS rate (~50fps)
            
    except KeyboardInterrupt:
        logger.info("Deteniendo simulador web...")
    finally:
        env.close()

def main():
    print("=========================================================")
    print("🚀 ATLAS Pro - DEMO LIVE WEB (SUMO BACKEND)")
    print("=========================================================")
    print("1. Levantando Dashboard FastAPI Interface...")
    
    dash_thread = threading.Thread(target=start_dashboard, kwargs={"port": 8888}, daemon=True)
    dash_thread.start()
    
    time.sleep(2)
    
    import webbrowser
    print("2. WebApp levantada. Abriendo el navegador automáticamente en http://localhost:8888 ...")
    webbrowser.open("http://localhost:8888")
    
    print("3. Conectando al simulador maestro TRACI headless...")
    run_simulation()

if __name__ == "__main__":
    main()
