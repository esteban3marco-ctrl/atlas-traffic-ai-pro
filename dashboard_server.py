# -*- coding: utf-8 -*-
"""
ATLAS Pro - Dashboard Server v1.0
=================================
Este servidor actúa como el punto de entrada principal para el Dashboard 3D.
Optimizado para baja latencia (WebSocket @ 15Hz) y control de simulación.
"""

import os
import sys
import os
import sys
import logging
import uvicorn

# Configuración de Logging (DEBE SER LO PRIMERO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ATLAS.DashboardServer")

from api_produccion import app

def run_dashboard_server():
    """
    Inicia el servidor dedicado para el dashboard.
    Por defecto corre en el puerto 8000 para coincidir con la config del frontend.
    """
    port = int(os.environ.get("ATLAS_DASHBOARD_PORT", 8000))
    host = os.environ.get("ATLAS_DASHBOARD_HOST", "0.0.0.0")
    
    logger.info("=" * 50)
    logger.info("🚀 ATLAS Pro - Dashboard Server Iniciando")
    logger.info(f"📍 Host: {host} | Puerto: {port}")
    logger.info(f"📊 WebSocket Stream: ws://{host}:{port}/ws/traffic-stream")
    logger.info("=" * 50)
    
    # Iniciar Uvicorn cargando la app desde api_produccion
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_dashboard_server()
