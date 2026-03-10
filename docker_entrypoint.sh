#!/bin/bash
# ==============================================================================
# ATLAS Pro - Docker Entrypoint
# ==============================================================================
set -e

echo "=================================================="
echo "  ATLAS Pro - Autonomous Traffic Light AI"
echo "  Environment: ${ATLAS_ENV:-development}"
echo "  Mode: $1"
echo "=================================================="

case "$1" in
  --api)
    echo "  Starting API + Dashboard on port 8080..."
    python api_produccion.py --host 0.0.0.0 --port 8080
    ;;
  --controller)
    echo "  Starting Traffic Controller..."
    echo "  Protocol: ${ATLAS_PROTOCOL:-ntcip}"
    echo "  Host: ${NTCIP_HOST:-192.168.1.100}"
    echo "  Model: ${ATLAS_MODEL:-modelos/agente_pro_heavy.pt}"
    python ntcip_adapter.py \
      --host "${NTCIP_HOST:-192.168.1.100}" \
      --port "${NTCIP_PORT:-161}" \
      --community "${NTCIP_COMMUNITY:-atlas}" \
      --model "${ATLAS_MODEL:-modelos/agente_pro_heavy.pt}" \
      --config config/train_config.yaml \
      --cycles 0
    ;;
  --monitor)
    echo "  Starting Anomaly Monitor..."
    python -c "
from anomalias_alertas import SistemaAlertas, HealthMonitor
import time, logging
logging.basicConfig(level=logging.INFO)
monitor = HealthMonitor()
print('  Monitor activo. Verificando cada 30s...')
while True:
    time.sleep(30)
"
    ;;
  --demo)
    echo "  Starting Demo Mode (simulated controller)..."
    python hardware_simulator.py --pattern balanced --speed 1x --duration 0
    ;;
  --test)
    echo "  Running tests..."
    python -m pytest tests_integracion.py -v
    ;;
  *)
    exec "$@"
    ;;
esac
