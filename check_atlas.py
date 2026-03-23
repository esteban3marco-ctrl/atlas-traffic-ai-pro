# check_atlas.py
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ATLAS.Check")

def check():
    try:
        import fastapi
        logger.info("✅ FastAPI: OK")
    except ImportError:
        logger.error("❌ FastAPI: NO ENCONTRADO")

    try:
        import traci
        import sumolib
        logger.info("✅ TraCI/sumolib: OK")
    except ImportError:
        logger.error("❌ TraCI/sumolib: NO ENCONTRADO")

    try:
        from atlas.production.inference_engine import InferenceEngine, ProductionConfig
        logger.info("✅ InferenceEngine: OK")
    except ImportError as e:
        logger.error(f"❌ InferenceEngine: FALLO AL IMPORTAR - {e}")

    try:
        sumo_binary = sumolib.checkBinary('sumo')
        logger.info(f"✅ SUMO binary found at: {sumo_binary}")
    except Exception as e:
        logger.error(f"❌ SUMO binary: NO ENCONTRADO - {e}")

    # Check config
    cfg_path = "simulations/complejo/simulation.sumocfg"
    if os.path.exists(cfg_path):
        logger.info(f"✅ SUMO config: OK ({cfg_path})")
    else:
        logger.error(f"❌ SUMO config: NO ENCONTRADA ({cfg_path})")

if __name__ == "__main__":
    check()
