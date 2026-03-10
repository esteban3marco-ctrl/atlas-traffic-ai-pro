import os
import sys
import time
import threading
import requests
import logging
import random
import uuid
import numpy as np

from atlas.config import AgentConfig
from atlas.agents import DuelingDDQNAgent
from atlas.production.dashboard import start_dashboard
from atlas.production.xai_engine import xai_engine
from atlas.production.safety_watchdog import watchdog

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("DemoWebStandalone")

class SimpleVehicle:
    def __init__(self, v_id, origin, lane, spawn_pos):
        self.id = v_id
        self.origin = origin # N, S, E, W
        self.lane = lane # 0, 1 -> offset laterally
        self.pos = spawn_pos # distance from center (start at 100)
        self.speed = 10.0 + random.uniform(-2, 2)
        self.max_speed = 15.0
        self.accel = 2.0
        self.decel = 4.0
        self.stopped = False

    def get_xy(self):
        # Distancias y offsets para que coincidan con la vista satélite de Madrid simulada
        offset = 4 + self.lane * 3 # lateral offset
        if self.origin == 'N':
            return (-offset, self.pos)
        elif self.origin == 'S':
            return (offset, -self.pos)
        elif self.origin == 'E':
            return (self.pos, offset)
        elif self.origin == 'W':
            return (-self.pos, -offset)

class StandaloneEnv:
    def __init__(self):
        self.vehicles = []
        self.current_phase = 0 # 0: NS Green, 1: NS Yellow, 2: EW Green, 3: EW Yellow
        self.step_length = 0.1
        self.time = 0.0
        self.congestion_multiplier = 1.0
        
    def step(self, phase):
        self.current_phase = phase
        self.time += self.step_length
        next_vehicles = []
        
        # Spawn logic (Generación procedimental de tráfico ficticio)
        spawn_rate = 0.3 * self.congestion_multiplier
        if random.random() < spawn_rate:
            origin = random.choice(['N', 'S', 'E', 'W'])
            
            # Si hay congestión manual, inyectamos más por el norte (accidente)
            if self.congestion_multiplier > 1.5 and random.random() < 0.6:
                origin = 'N'
                
            lane = random.choice([0, 1])
            self.vehicles.append(SimpleVehicle(f"v_{uuid.uuid4().hex[:6]}", origin, lane, 100.0))
            
        lanes = {}
        for v in self.vehicles:
            k = (v.origin, v.lane)
            if k not in lanes: lanes[k] = []
            lanes[k].append(v)
            
        for k in lanes:
            # Sort by position (smaller pos means closer to crossing the intersection)
            lanes[k].sort(key=lambda x: x.pos)
            
            for i, v in enumerate(lanes[k]):
                leader = lanes[k][i-1] if i > 0 else None
                
                target_speed = v.max_speed
                # Traffic light logic
                # Stop line is at pos = 12.0
                dist_to_light = v.pos - 12.0
                
                facing_red = False
                if v.origin in ['N', 'S']:
                    if self.current_phase in [2, 3]: facing_red = True
                else:
                    if self.current_phase in [0, 1]: facing_red = True
                    
                if facing_red and 0 < dist_to_light < 40:
                    # Frena suavemente para detenerse en la línea
                    target_speed = min(target_speed, max(0, dist_to_light * 0.4))
                
                # Coche de delante
                if leader is not None:
                    dist_to_leader = v.pos - leader.pos
                    if 0 < dist_to_leader < 15:
                        target_speed = min(target_speed, leader.speed, max(0, dist_to_leader * 0.5 - 2.0))
                        
                # Aceleración / Deceleración cinemática ficticia
                if v.speed < target_speed:
                    v.speed = min(target_speed, v.speed + v.accel * self.step_length)
                elif v.speed > target_speed:
                    v.speed = max(target_speed, v.speed - v.decel * self.step_length)
                    
                v.pos -= v.speed * self.step_length
                
                # Despawn
                if v.pos > -100:
                    next_vehicles.append(v)
                    
        self.vehicles = next_vehicles
        
    def get_vehicles_pos(self):
        res = []
        for v in self.vehicles:
            x, y = v.get_xy()
            res.append({"id": v.id, "x": x, "y": y, "origin": v.origin})
        return res
        
    def get_obs(self):
        # Construye un vector de observación 26 de dimensiones pseudo-reales
        obs = [0]*26
        queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        speeds = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        counts = {'N': 1e-5, 'S': 1e-5, 'E': 1e-5, 'W': 1e-5}
        for v in self.vehicles:
            if v.pos > 0:
                o = v.origin
                if v.speed < 1.0: queues[o] += 1
                speeds[o] += v.speed
                counts[o] += 1
                
        # Normalizaciones aproximadas para Dueling DDQN
        obs[0] = queues['N'] / 10.0; obs[1] = speeds['N'] / counts['N'] / 15.0; obs[4] = queues['N'] / 15.0
        obs[5] = queues['S'] / 10.0; obs[6] = speeds['S'] / counts['S'] / 15.0; obs[9] = queues['S'] / 15.0
        obs[10] = queues['E'] / 10.0; obs[11] = speeds['E'] / counts['E'] / 15.0; obs[14] = queues['E'] / 15.0
        obs[15] = queues['W'] / 10.0; obs[16] = speeds['W'] / counts['W'] / 15.0; obs[19] = queues['W'] / 15.0
        
        # Phase (OneHot 20-23)
        obs[20 + self.current_phase] = 1.0
        # Timing proxies
        obs[24] = 0.5  # Time in phase
        obs[25] = 0.5  # Time of day
        return np.array(obs, dtype=np.float32)

def run_simulation():
    # 1. Configurar Agente IA 
    config = AgentConfig(state_dim=26, action_dim=4)
    agent = DuelingDDQNAgent(state_dim=26, action_dim=4, config=config, device="cpu")
    
    # Intentar cargar modelo
    model_path = "checkpoints_extended/atlas_best.pt"
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            logger.info(f"Modelo IA Cargado ({model_path})")
        except: pass
        
    env = StandaloneEnv()
    
    logger.info("📡 Motor Físico Standalone (No SUMO) Inicializado. Transmitiendo...")
    
    phase_timer = 0
    explanation = "Iniciating Neural Analysis of Incoming Traffic Flow..."
    saliency_weights = [0.1]*20
    
    try:
        while True:
            # 2. Física "Fake" Step
            env.step(env.current_phase)
            vehicles_pos = env.get_vehicles_pos()
            
            map_data = {
                "center": {
                    "coords": [40.4194, -3.7042],
                    "status": "green" if env.current_phase % 2 == 0 else "red"
                }
            }
            
            data = {
                "vehicles": len(vehicles_pos),
                "vehicles_pos": vehicles_pos,
                "safety_score": 100,
                "current_phase": "NORTH-SOUTH" if env.current_phase % 2 == 0 else "EAST-WEST",
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
            
            phase_timer += 0.1
            
            # IA Decision
            if phase_timer >= 5.0:
                obs = env.get_obs()
                ai_action, saliency = agent.select_action(obs, evaluate=True, return_xai=True)
                explanation = xai_engine.generate_explanation(obs, ai_action, saliency)
                
                incident_active = False
                try:
                    r = requests.get("http://localhost:8888/api/incident_status", timeout=0.1)
                    if r.status_code == 200: incident_active = r.json().get('active', False)
                except: pass
                
                # Forzamos tráfico si hay accidente
                if incident_active:
                    env.congestion_multiplier = 2.0
                    explanation = "EMERGENCY: Rerouting High Density Traffic!"
                    if env.current_phase == 0: ai_action = 1
                else:
                    env.congestion_multiplier = 1.0
                
                final_action = watchdog.validate_action(ai_action, env.current_phase, False) % 4
                
                if final_action != env.current_phase:
                    # Simulate Yellow Phase
                    env.current_phase = final_action
                    
                if saliency is not None:
                     saliency_weights = saliency.tolist()[:20] if hasattr(saliency, 'tolist') else [random.random()*0.5 for _ in range(20)]
                else:
                     saliency_weights = [random.random()*0.5 for _ in range(20)]
                     
                phase_timer = 0
            
            time.sleep(0.02) # Control de Loop para ~50fps en canvas
            
    except KeyboardInterrupt:
        logger.info("Deteniendo simulador standalone...")

def main():
    print("=========================================================")
    print("🤖 ATLAS Pro - DEMO FULL NATIVE (STANDALONE PHYSICS)")
    print("=========================================================")
    print("1. Levantando Dashboard FastAPI (100% Python, No SUMO)...")
    
    dash_thread = threading.Thread(target=start_dashboard, kwargs={"port": 8888}, daemon=True)
    dash_thread.start()
    
    time.sleep(2)
    
    import webbrowser
    print("2. Abriendo navegador automáticamente. (http://localhost:8888)")
    webbrowser.open("http://localhost:8888")
    
    print("3. Arrancando sistema kinemático procedimental...")
    run_simulation()

if __name__ == "__main__":
    main()
