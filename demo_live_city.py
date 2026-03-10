
"""
ATLAS Pro — Live City Deployment Demo
======================================
Launches the full production stack:
Dashboard with Real-time Map + Conversational XAI + MARL Coordination.
"""

import os
import sys
import time
import threading
import logging
import numpy as np
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from atlas.config import AgentConfig, EnvironmentConfig
from atlas.production.inference_engine import InferenceEngine, ProductionConfig
from atlas.production.dashboard import start_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("ATLAS.Demo")

def run_production_demo():
    print("\n" + "="*70)
    print("🚀 ATLAS PRO — LIVE CITY DEPLOYMENT DEMO")
    print("="*70)
    print("\nIniciando sistema de producción inteligente...")
    
    # 1. Configuration for Multiple Intersections in Madrid Gran Vía area
    env_cfg = EnvironmentConfig(
        is_multi_agent=True,
        traffic_light_ids=["gran_via_1", "gran_via_2", "gran_via_3", "callao"],
        map_coordinates={
            "gran_via_1": (40.4194, -3.7042),
            "gran_via_2": (40.4201, -3.7058),
            "gran_via_3": (40.4210, -3.7075),
            "callao": (40.4203, -3.7051)
        },
        gui=False
    )
    
    agent_cfg = AgentConfig(
        use_qmix=True,
        n_agents=4
    )
    
    prod_config = ProductionConfig(
        mode="demo", # Uses simulated cameras and controllers for the demo
        decision_interval=3.0,
        model_path="" # Use random weights for demo stability
    )
    
    # In demo mode, we'll manually inject the environment settings for the map
    # Since we can't easily modify the dataclass right now, we'll ensure
    # the engine can access these or we'll mock them.
    class MockEnv:
        def __init__(self, ids, coords):
            self.traffic_light_ids = ids
            self.map_coordinates = coords
            
    prod_config.env = MockEnv(
        ["gran_via_1", "gran_via_2", "gran_via_3", "callao"],
        {
            "gran_via_1": (40.4194, -3.7042),
            "gran_via_2": (40.4201, -3.7058),
            "gran_via_3": (40.4210, -3.7075),
            "callao": (40.4203, -3.7051)
        }
    )
    
    # 2. Start Dashboard in a separate thread
    print("📡 Levantando Dashboard en http://localhost:8888 ...")
    dashboard_thread = threading.Thread(
        target=start_dashboard, 
        kwargs={"port": 8888},
        daemon=True
    )
    dashboard_thread.start()
    
    time.sleep(2) # Wait for server to set up
    
    # 3. Start Inference Engine
    print("🧠 Inicializando Motor de Inferencia MARL + Conversational XAI...")
    engine = InferenceEngine(prod_config)
    
    if engine.initialize():
        print("\n" + "-"*70)
        print("✅ SISTEMA ONLINE")
        print("📍 Ciudad: Madrid (Gran Vía)")
        print("🤖 IA: QMIX Coordinated Fleet")
        print("🗣️ XAI: Activado (Explicaciones en lenguaje natural)")
        print("🗺️ MAPA: Activado (Leaflet Real-time)")
        print("-"*70)
        print("\nAbriendo Dashboard... (Por favor, abre http://localhost:8888 en tu navegador)")
        
        try:
            # Run the engine (this will run until interrupted)
            engine.run()
        except KeyboardInterrupt:
            print("\n⚠️ Deteniendo demo...")
            engine.shutdown()
    else:
        print("❌ Error al inicializar el motor.")

if __name__ == "__main__":
    run_production_demo()
