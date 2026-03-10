import os
import sys
import time
import random
import logging

try:
    import traci
except ImportError:
    print("Error: SUMO (traci) no está instalado. Ejecuta: pip install traci sumolib")
    sys.exit(1)

from atlas.config import EnvironmentConfig
from atlas.sumo_env import ATLASTrafficEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("DemoSUMO")

def run_sumo_demo():
    print("=========================================================")
    print("🚦 ATLAS Pro - Demo Simulador SUMO en Tiempo Real")
    print("=========================================================")
    print("Iniciando entorno con Interfaz Gráfica (SUMO-GUI)...")
    
    # Configurar el entorno para usar la GUI
    cfg = EnvironmentConfig(
        sumo_cfg_file="simulations/simple/simulation.sumocfg",
        gui=True,
        step_length=0.1,
        max_steps=3600
    )
    
    try:
        env = ATLASTrafficEnv(env_config=cfg, render_mode="sumo_gui")
        obs, info = env.reset()
        
        print("✅ Simulación conectada. Se abrirá la ventana de SUMO-GUI.")
        print("💡 TIP: En la ventana de SUMO, ajusta el 'Delay' (ms) arriba a 50-100 para verlo a velocidad normal.")
        print("Pulsa Ctrl+C en esta terminal para detener la simulación.")
        
        # Bucle de simulación con un agente de prueba (heurístico/aleatorio)
        step = 0
        while True:
            # Seleccionamos una acción lógica para no chocar (la lógica interna ya tiene semáforo amarillo)
            # Para la demo, el semáforo cambiará basándose en la longitud de fase simulando un agente
            if (step % 20) == 0:
                action = random.choice([env.ACTION_SWITCH_NS, env.ACTION_SWITCH_EW, env.ACTION_MAINTAIN])
            else:
                action = env.ACTION_MAINTAIN
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Mostramos progreso en la terminal brevemente
            if step % 10 == 0:
                queues = info.get("traffic_data", {}).get("queues", {})
                logger.info(f"Paso {step*5} (aprox) | Colas actuales: {queues}")
                
            step += 1
            
            if terminated or truncated:
                print("Simulación terminada, reiniciando...")
                obs, info = env.reset()
                step = 0
                
            time.sleep(0.01) # Ligero delay para que python no sature la CPU
            
    except KeyboardInterrupt:
        print("\nDeteniendo simulación...")
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("=========================================================")
        print("Demo finalizada.")
        print("=========================================================")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sumo_cfg = os.path.join(current_dir, "simulations", "simple", "simulation.sumocfg")
    
    if not os.path.exists(sumo_cfg):
        print(f"❌ Error: No se encontró el archivo de configuración {sumo_cfg}")
        print("Asegúrate de ejecutar este script desde la raíz del proyecto.")
    else:
        run_sumo_demo()
