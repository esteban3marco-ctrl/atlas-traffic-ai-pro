"""
ATLAS Pro — Command Line Interface
=====================================
Professional CLI for training, evaluation, and management.
"""

import os
import sys
import logging
import argparse
from pathlib import Path


def setup_logging(level: str = "INFO"):
    """Configure logging for ATLAS."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args):
    """Train an RL agent."""
    from atlas.config import TrainingConfig
    from atlas.trainer import Trainer
    
    if args.config and os.path.exists(args.config):
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
    
    # Override from args
    if args.episodes:
        config.total_episodes = args.episodes
    if args.algorithm:
        config.agent.algorithm = args.algorithm
    if args.device:
        config.device = args.device
    if args.sumo_cfg:
        config.environment.sumo_cfg_file = args.sumo_cfg
    if args.gui:
        config.environment.gui = True
    
    # Setup directories
    config.checkpoint_dir = args.output or "checkpoints"
    config.log_dir = args.logdir or "runs"
    
    trainer = Trainer(config)
    summary = trainer.train(resume_from=args.resume)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Reward: {summary['best_reward']:.1f}")
    print(f"  Mean Reward: {summary['mean_reward']:.1f}")
    print(f"  Time:        {summary['training_time_minutes']:.1f} min")
    print(f"{'='*60}")


def cmd_evaluate(args):
    """Evaluate a trained agent."""
    from atlas.config import TrainingConfig
    from atlas.trainer import Trainer
    
    config = TrainingConfig()
    if args.config and os.path.exists(args.config):
        config = TrainingConfig.load(args.config)
    
    if args.sumo_cfg:
        config.environment.sumo_cfg_file = args.sumo_cfg
    
    trainer = Trainer(config)
    
    if args.checkpoint:
        trainer.agent.load(args.checkpoint)
    
    results = trainer.benchmark_baselines(n_episodes=args.episodes or 10)
    
    print(f"\n{'='*60}")
    print("Benchmark Results")
    print(f"{'='*60}")
    for name, reward in sorted(results.items(), key=lambda x: -x[1]):
        marker = ">>>" if name == "atlas_ai" else "   "
        print(f"  {marker} {name:20s}: {reward:.1f}")
    print(f"{'='*60}")


def cmd_export(args):
    """Export model to ONNX."""
    from atlas.config import AgentConfig
    from atlas.agents import DuelingDDQNAgent
    
    agent = DuelingDDQNAgent(config=AgentConfig())
    agent.load(args.checkpoint)
    
    output = args.output or args.checkpoint.replace('.pt', '.onnx')
    agent.export_onnx(output)
    print(f"Exported to {output}")


def cmd_dashboard(args):
    """Start the web dashboard."""
    from atlas.dashboard.app import run_dashboard
    
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    
    print(f"\n  ATLAS Pro Dashboard")
    print(f"  http://localhost:{port}\n")
    
    run_dashboard(host=host, port=port)


def cmd_production(args):
    """Start the production inference engine."""
    from atlas.production.inference_engine import run_production
    
    print(f"\n  🚀 ATLAS Pro — Production Engine")
    print(f"  Mode:       {args.mode.upper()}")
    print(f"  Controller: {args.controller}")
    print(f"  Model:      {args.model}\n")
    
    run_production(
        mode=args.mode,
        model_path=args.model,
        controller_type=args.controller,
        decision_interval=args.interval
    )



def cmd_config(args):
    """Generate or display configuration."""
    from atlas.config import TrainingConfig
    
    if args.action == "generate":
        config = TrainingConfig()
        output = args.output or "atlas_config.yaml"
        config.save(output)
        print(f"Config saved to {output}")
    
    elif args.action == "show":
        if args.file and os.path.exists(args.file):
            config = TrainingConfig.load(args.file)
        else:
            config = TrainingConfig()
        
        import yaml
        print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="ATLAS Pro — Advanced Traffic Light AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atlas train --episodes 500 --algorithm dueling_ddqn
  atlas train --config myconfig.yaml --resume checkpoints/best.pt
  atlas evaluate --checkpoint checkpoints/best.pt --episodes 20
  atlas export --checkpoint checkpoints/best.pt
  atlas dashboard --port 8000
  atlas config generate --output myconfig.yaml
        """,
    )
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # --- train ---
    train_p = subparsers.add_parser("train", help="Train an RL agent")
    train_p.add_argument("--episodes", type=int, help="Number of training episodes")
    train_p.add_argument("--algorithm", choices=["dueling_ddqn", "ppo"], help="RL algorithm")
    train_p.add_argument("--config", type=str, help="Config YAML file")
    train_p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Compute device")
    train_p.add_argument("--sumo-cfg", type=str, help="SUMO config file path")
    train_p.add_argument("--gui", action="store_true", help="Show SUMO GUI")
    train_p.add_argument("--resume", type=str, help="Checkpoint to resume from")
    train_p.add_argument("--output", type=str, help="Checkpoint output directory")
    train_p.add_argument("--logdir", type=str, help="TensorBoard log directory")
    
    # --- evaluate ---
    eval_p = subparsers.add_parser("evaluate", help="Evaluate agent vs baselines")
    eval_p.add_argument("--checkpoint", type=str, help="Agent checkpoint path")
    eval_p.add_argument("--config", type=str, help="Config YAML file")
    eval_p.add_argument("--episodes", type=int, help="Evaluation episodes")
    eval_p.add_argument("--sumo-cfg", type=str, help="SUMO config file path")
    
    # --- export ---
    export_p = subparsers.add_parser("export", help="Export model to ONNX")
    export_p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to export")
    export_p.add_argument("--output", type=str, help="Output ONNX file path")
    
    # --- dashboard ---
    dash_p = subparsers.add_parser("dashboard", help="Start the web dashboard")
    dash_p.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    dash_p.add_argument("--port", type=int, default=8000, help="Port number")
    
    # --- config ---
    config_p = subparsers.add_parser("config", help="Manage configuration")
    config_p.add_argument("action", choices=["generate", "show"], help="Action to perform")
    config_p.add_argument("--file", type=str, help="Config file to show")
    config_p.add_argument("--output", type=str, help="Output file for generation")

    # --- production ---
    prod_p = subparsers.add_parser("production", help="Run in production mode")
    prod_p.add_argument("--mode", choices=["live", "shadow", "demo"], default="shadow", help="Operation mode")
    prod_p.add_argument("--model", default="checkpoints_extended/atlas_best.pt", help="Model path")
    prod_p.add_argument("--controller", default="simulated", choices=["simulated", "modbus", "rest_api", "gpio"], help="Hardware controller type")
    prod_p.add_argument("--interval", type=float, default=5.0, help="Seconds between decisions")
    
    args = parser.parse_args()
    
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "dashboard": cmd_dashboard,
        "config": cmd_config,
        "production": cmd_production,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
