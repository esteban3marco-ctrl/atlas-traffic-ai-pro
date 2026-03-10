# ATLAS Pro — Guía de Despliegue en Producción 🚦

Esta guía detalla los pasos para desplegar ATLAS Pro en una intersección real utilizando visión por computador y controladores industriales.

## 1. Hardware Recomendado

### Unidad de Procesamiento (Edge AI)
Para que el sistema funcione en tiempo real con baja latencia, se recomienda:
*   **NVIDIA Jetson Orin Nano/NX**: Ideal para despliegue en poste. Consume poca energía y tiene aceleración para YOLOv8.
*   **Mini-PC con GPU NVIDIA** (ej. RTX 3050/4050): Para centros de control o armarios de tráfico refrigerados.
*   **Raspberry Pi 5 + Hailo-8 AI Kit**: Opción económica pero potente para detección básica.

### Cámaras
*   **Tipo**: Cámaras IP con soporte RTSP.
*   **Resolución**: 1080p (mínimo 720p).
*   **Lentes**: Angulares para cubrir todos los carriles de entrada.
*   **Visión Nocturna**: Infrarrojos (IR) potentes o cámaras de alta sensibilidad (DarkFighter/Starlight).

---

## 2. Diagrama de Conexión

```
📷 Cámara Norte (RTSP) \
📷 Cámara Sur   (RTSP)  --> [ 🖥️ Servidor ATLAS Pro ] --> [ 🚦 Controlador Físico ]
📷 Cámara Este  (RTSP) /            (Edge AI)               (NTCIP / Modbus / API)
📷 Cámara Oeste (RTSP)
```

---

## 3. Instalación de Software

En el servidor de producción:

```bash
# 1. Clonar y preparar entorno
git clone https://github.com/atlas-traffic-ai-pro
cd atlas-traffic-ai-pro
pip install -r requirements.txt
pip install ultralytics fastapi uvicorn pymodbus  # Dependencias de producción

# 2. Verificar modelo entrenado
# Copia tu mejor modelo a la unidad de producción
cp checkpoints_extended/atlas_best.pt ./production_model.pt
```

---

## 4. Configuración de Cámaras

Edita `atlas/production/inference_engine.py` o pasa los argumentos por CLI. Necesitas las URLs RTSP de tus cámaras:

```python
camera_sources = {
    "N": "rtsp://admin:password@192.168.1.50/stream1",
    "S": "rtsp://admin:password@192.168.1.51/stream1",
    "E": "rtsp://admin:password@192.168.1.52/stream1",
    "W": "rtsp://admin:password@192.168.1.53/stream1",
}
```

---

## 5. Ejecución en Modos

### Paso A: Modo Shadow (Validación Segura)
En este modo, la IA observa y toma decisiones en el log, pero **no toca el semáforo**. Úsalo durante la primera semana para comparar la IA con el sistema actual.

```bash
python -m atlas.cli production --mode shadow --model production_model.pt
```

### Paso B: Modo Live (Control Total)
Una vez validado, conecta ATLAS al controlador (ej. vía Modbus) para tomar el control.

```bash
python -m atlas.cli production --mode live --controller modbus --model production_model.pt
```

---

## 🧠 AI Engine: ATLAS Pro v2.5 "SOTA"

The system uses a cutting-edge Reinforcement Learning architecture:
- **Algorithms**: Rainbow DQN (Distributional, Noisy, Dueling, Double, N-step, PER)
- **Neural Backbone**: Multi-head Transformer Attention (State-of-the-art perception)
- **Explainability (XAI)**: Real-time feature saliency mapping (know *why* the AI acts)
- **World Model**: Latent reward prediction for predictive planning
- **Optimization**: INT8 Dynamic Quantization for Edge deployment (10x faster inference)
- **Robustness**: Domain Randomization (Sim-to-Real trained)

### Model Variants
1. `atlas_sota_v25.pt`: Full Transformer architecture (Recommended)
2. `atlas_quantized.pt`: High-speed INT8 version for Raspberry Pi / Jetson

---

## 6. Seguridad (Fail-Safe)

ATLAS Pro incluye capas de seguridad por diseño:
1.  **Watchdog**: Si el script de IA se detiene, el controlador físico vuelve automáticamente a su plan de tiempos fijo (Fallback).
2.  **Tiempos Mínimos**: La IA no puede cambiar un verde antes de los 10 segundos preconfigurados.
3.  **Conflictos**: El controlador de hardware (la placa física) siempre tiene la última palabra y bloquea estados verdes conflictivos (bloqueo mutuo).

---

## 7. Soporte de Protocolos

*   **Modbus TCP**: Estándar en controladores europeos.
*   **NTCIP**: Estándar en EE.UU.
*   **GPIO**: Para prototipado con relés directos.
*   **REST API**: Para integración con Smart City Platforms modernas.
