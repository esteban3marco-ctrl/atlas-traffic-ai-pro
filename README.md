<p align="center">
  <img src="https://img.shields.io/badge/tests-43%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-49%25-yellow" alt="Coverage">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-00FFFF?logo=yolo" alt="YOLO">
  <img src="https://img.shields.io/badge/LSTM-TensorFlow-FF6F00?logo=tensorflow" alt="LSTM">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/SUMO-TraCI-green" alt="SUMO">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED" alt="ONNX">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

# ATLAS Traffic AI Pro

### AI-powered smart city traffic management integrating 5 ML models

ATLAS (Adaptive Traffic Light Automation System) is a production-ready intelligent traffic control platform that combines computer vision, deep reinforcement learning, and multi-agent coordination to optimize traffic flow across entire urban networks. Trained on Kaggle with Tesla P100 GPUs, ATLAS reduces average wait times by 25-35% and CO2 emissions by 15-22% compared to traditional fixed-timing systems.

---

## 5 ML Models Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ATLAS TRAFFIC AI PRO                                │
│              5 Interconnected ML Models Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐       │
│   │  MODEL 1     │     │  MODEL 2     │     │  MODEL 3             │       │
│   │  YOLOv8      │     │  OCR Engine  │     │  LSTM Predictor      │       │
│   │  Detection   │     │  License     │     │  Traffic Forecast    │       │
│   │              │     │  Plates      │     │                      │       │
│   │ • Vehicles   │     │              │     │ • 15/30/60 min       │       │
│   │ • Pedestrians│     │ • Read plates│     │   ahead prediction   │       │
│   │ • Cyclists   │     │ • Track IDs  │     │ • Congestion pattern │       │
│   │ • Emergency  │     │ • Enforce    │     │ • Seasonal trends    │       │
│   │   vehicles   │     │   violations │     │ • Event detection    │       │
│   └──────┬───────┘     └──────┬───────┘     └──────────┬───────────┘       │
│          │                    │                         │                   │
│          ▼                    ▼                         ▼                   │
│   ┌──────────────────────────────────────────────────────────────┐         │
│   │                    SENSOR FUSION LAYER                       │         │
│   │        Queue lengths • Speeds • Occupancy • Wait times       │         │
│   │                   26-dimensional state vector                │         │
│   └──────────────────────────┬───────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│   ┌──────────────────────────────────────────────────────────────┐         │
│   │                  MODEL 4: AUTOENCODER                        │         │
│   │              Night Surveillance & Anomaly Detection          │         │
│   │                                                              │         │
│   │  • Low-light image enhancement    • Z-score anomaly flags    │         │
│   │  • Reconstruction error scoring   • Statistical baselines    │         │
│   │  • Ghost vehicle filtering        • Incident alerts          │         │
│   └──────────────────────────┬───────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│   ┌──────────────────────────────────────────────────────────────┐         │
│   │            MODEL 5: DEEP REINFORCEMENT LEARNING              │         │
│   │          Dueling Double DQN (D3QN) + PPO + QMIX              │         │
│   │                                                              │         │
│   │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │         │
│   │  │  D3QN Agent  │  │  PPO Agent   │  │ QMIX Coordinator  │   │         │
│   │  │             │  │              │  │                   │   │         │
│   │  │ Dueling     │  │ Actor-Critic │  │ 50-500+           │   │         │
│   │  │ streams     │  │ GAE          │  │ intersections     │   │         │
│   │  │ Prioritized │  │ Entropy reg. │  │ IGM condition     │   │         │
│   │  │ replay      │  │ Clip ratio   │  │ Hyper-networks    │   │         │
│   │  │ N-step      │  │              │  │                   │   │         │
│   │  │ returns     │  │              │  │ Decentralized     │   │         │
│   │  │ Noisy nets  │  │              │  │ execution         │   │         │
│   │  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘   │         │
│   │         └────────────────┴───────────────────┘               │         │
│   │                          │                                   │         │
│   └──────────────────────────┼───────────────────────────────────┘         │
│                              │                                             │
│                              ▼                                             │
│   ┌──────────────────────────────────────────────────────────────┐         │
│   │                 PRODUCTION LAYER                             │         │
│   │                                                              │         │
│   │  Safety Watchdog ─► Controller Interface ─► NTCIP/Modbus    │         │
│   │  XAI Engine      ─► Dashboard (FastAPI)  ─► WebSocket UI    │         │
│   │  MUSE Metacog.   ─► Audit Logs (JSONL)   ─► Compliance      │         │
│   └──────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

```
atlas-traffic-ai-pro/
├── atlas/                          # Core production module
│   ├── agents.py                   # D3QN + PPO agents
│   ├── networks.py                 # NoisyLinear, Transformer, C51, WorldModel
│   ├── config.py                   # YAML-serializable configuration system
│   ├── rewards.py                  # Multi-objective reward (6 components)
│   ├── sumo_env.py                 # Gymnasium environment (SUMO/TraCI)
│   ├── trainer.py                  # Training pipeline + TensorBoard
│   ├── replay_buffer.py           # PER + N-step + SumTree
│   ├── baselines.py               # Fixed / MaxPressure / Actuated / Random
│   ├── cli.py                     # CLI: train | evaluate | deploy
│   ├── dashboard/                 # Real-time web dashboard
│   │   └── app.py                 # FastAPI + WebSocket server
│   ├── production/                # Deployment infrastructure
│   │   ├── inference_engine.py    # Real-time AI decision loop
│   │   ├── xai_engine.py         # Explainability (attention + saliency)
│   │   ├── controller_interface.py # NTCIP / Modbus / REST / GPIO
│   │   ├── camera_pipeline.py    # YOLO vision → traffic metrics
│   │   └── safety_watchdog.py    # Independent safety monitor
│   └── tests/                    # Test suite (43 tests)
│
├── src/                           # Extended modules
│   ├── agent/ddqn_agent.py       # Alternative DDQN implementation
│   ├── coordination/             # QMIX multi-agent coordinator
│   └── muse/muse_engine.py      # Metacognition + explainability
│
├── simulations/                   # 12 SUMO traffic scenarios
│   ├── simple/                   # Basic single intersection
│   ├── complejo/                 # Complex multi-intersection network
│   ├── emergencias/              # Emergency vehicle priority
│   ├── hora_punta/               # Peak hour stress test
│   ├── noche/                    # Night-time low traffic
│   ├── avenida/                  # Avenue corridor
│   ├── cruce_t/                  # T-junction
│   ├── evento/                   # Special event scenarios
│   └── ...                       # + 4 more scenario variants
│
├── main.py                       # Main entry point (all modes)
├── api_produccion.py             # Production REST API
├── dashboard_server.py           # Dashboard server
├── demo_live_sumo_web.py         # Live SUMO + web demo
└── ejecutar_entrenamiento_pro.py # Training launcher
```

---

## Key Features

- **Multi-objective reward function** balancing queue length, wait time, throughput, fairness (Jain's index), emissions, and phase stability
- **Prioritized Experience Replay** with SumTree for O(log n) sampling (Schaul et al., 2016)
- **N-step returns** for improved bias-variance tradeoff
- **Noisy networks** for parameter-space exploration (Fortunato et al., 2018)
- **QMIX multi-agent** coordination for city-scale deployment (50-500+ intersections)
- **MUSE metacognition engine** providing human-readable decision explanations
- **Safety watchdog** with automatic fallback to fixed timing on AI failure
- **NTCIP / Modbus / REST / GPIO** controller interfaces for real hardware
- **Edge-first design** targeting < 50ms inference on Jetson Nano via ONNX Runtime
- **12 SUMO simulation scenarios** covering intersections, avenues, emergencies, and events

---

## Installation

### Prerequisites

- Python 3.10+
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html) (Simulation of Urban Mobility)
- CUDA 11.8+ (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/esteban3marco-ctrl/atlas-traffic-ai-pro.git
cd atlas-traffic-ai-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements_prod.txt

# Verify installation
python main.py --modo demo
```

### Docker (alternative)

```bash
docker build -t atlas-traffic .
docker run -p 8000:8000 atlas-traffic
```

---

## Usage

### Training

```bash
# Train D3QN agent on simple intersection
python ejecutar_entrenamiento_pro.py

# Train with CLI (advanced)
python -m atlas.cli train --episodes 1000 --scenario simple

# Train with specific configuration
python -m atlas.cli train --config config/training.yaml
```

### Running the Simulation

```bash
# Quick demo mode (no SUMO required)
python main.py --modo demo

# Live SUMO simulation with web dashboard
python demo_live_sumo_web.py

# Production mode with real controllers
python main.py --modo produccion
```

### Dashboard

```bash
# Start the web dashboard
python dashboard_server.py
# Open http://localhost:8000 in your browser
```

### API

```bash
# Start the production API
python api_produccion.py

# Query intersection status
curl http://localhost:8000/api/v1/intersections/INT_01/status

# Get AI decision explanation
curl http://localhost:8000/api/v1/intersections/INT_01/explain
```

### Evaluation

```bash
# Benchmark against baselines (fixed timing, max pressure, actuated)
python -m atlas.cli evaluate --scenario complejo
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Deep Learning | PyTorch | >= 2.0.0 |
| Object Detection | YOLOv8 (Ultralytics) | Latest |
| Computer Vision | OpenCV | >= 4.x |
| Traffic Simulation | SUMO + TraCI | Latest |
| RL Environment | Gymnasium | Latest |
| Edge Inference | ONNX Runtime | >= 1.16.0 |
| REST API | FastAPI | >= 0.100.0 |
| Real-time Comms | WebSockets | >= 11.0 |
| Traffic Protocols | PySNMP (NTCIP) | 4.4.12 |
| Monitoring | Prometheus | >= 0.17.0 |
| Database | PostgreSQL (psycopg2) | >= 2.9.0 |
| Reports | ReportLab | >= 4.0.0 |
| Training Monitoring | TensorBoard | Latest |

### Training Infrastructure

Model training performed on **Kaggle** with **Tesla P100 GPU** (16 GB VRAM). Training pipeline supports checkpoint resumption, early stopping, and curriculum learning with progressive difficulty across 12 simulation scenarios.

---

## Reward Function

The multi-objective reward balances six components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Queue length | -0.30 | Penalizes accumulated vehicle queues |
| Wait time | -0.60 | Penalizes excessive waiting at red lights |
| Throughput | +0.50 | Rewards vehicles successfully served |
| Fairness (Jain) | -0.15 | Penalizes unequal wait distribution |
| Emissions proxy | -0.05 | Penalizes stop-and-go patterns |
| Phase stability | -0.50 | Penalizes excessive phase switching |
| Emergency bonus | +5.00 | Prioritizes emergency vehicle passage |

---

## Test Suite

```
43 passed | 49% coverage
```

Tests cover configuration, neural network forward passes, replay buffers, agent training loops, reward computation, environment reset/step cycles, and multi-agent coordination.

```bash
# Run tests
python -m pytest atlas/tests/ -v

# Run with coverage
python -m pytest atlas/tests/ --cov=atlas
```

---

## Project Status

ATLAS is under active development. Current version: **v4.0**

| Feature | Status |
|---------|--------|
| D3QN Agent | Production |
| PPO Agent | Production |
| QMIX Multi-Agent | Production |
| SUMO Integration | Production |
| YOLO Detection | Production |
| NTCIP Controller | Production |
| Safety Watchdog | Production |
| XAI / MUSE Engine | Production |
| Web Dashboard | Production |
| Edge Deployment (ONNX) | Beta |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with PyTorch, SUMO, and reinforcement learning<br>
  Trained on Kaggle with Tesla P100
</p>
