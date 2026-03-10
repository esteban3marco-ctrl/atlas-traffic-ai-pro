import os
import sys
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("ATLAS.Export")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.config import TrainingConfig
from atlas.agents import DuelingDDQNAgent

def main():
    logger.info("🚀 ATLAS Pro v2.5 SOTA Export Utility")
    
    # 1. Initialize Agent with V2 config
    config = TrainingConfig().agent
    config.use_transformer = True
    config.use_world_model = True
    
    agent = DuelingDDQNAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        config=config,
        device="cpu"
    )
    
    # 2. Mock load (or load best if exists)
    model_path = "checkpoints/atlas_dueling_ddqn_best.pt"
    if os.path.exists(model_path):
        agent.load(model_path)
        logger.info(f"✅ Loaded weights from {model_path}")
    else:
        logger.warning("⚠️ No existing weights found. Exporting architectural skeleton.")
    
    # 3. Export to Quantized TorchScript
    export_path = "checkpoints/atlas_v25_quantized.pt"
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        agent.export_for_production(export_path)
        logger.info(f"✨ SUCCESS: SOTA Model exported to {export_path}")
        logger.info("   Architecture: Transformer + Distributional + World Model")
        logger.info("   Format: TorchScript (INT8 Quantized)")
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
