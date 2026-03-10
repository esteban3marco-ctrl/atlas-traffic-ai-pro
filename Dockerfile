# ==============================================================================
# ATLAS Pro - Dockerfile para Despliegue Edge
# ==============================================================================
# Soporta: x86_64, ARM64 (Raspberry Pi 4, Jetson Nano)
# Imagen base: Python 3.10 slim (~120 MB base)
#
# Build:
#   docker build -t atlas-pro:latest .
#
# Run (produccion):
#   docker run -d --name atlas \
#     -p 8080:8080 \
#     -v ./modelos:/app/modelos \
#     -v ./config:/app/config \
#     -e NTCIP_HOST=192.168.1.100 \
#     atlas-pro:latest
#
# Run (demo/test):
#   docker run -d --name atlas-demo \
#     -p 8080:8080 \
#     atlas-pro:latest --demo
# ==============================================================================

FROM python:3.10-slim AS base

# Metadata
LABEL maintainer="Esteban Marco"
LABEL description="ATLAS Pro - Autonomous Traffic Light Adaptive System"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ATLAS_HOME=/app \
    ATLAS_ENV=production

WORKDIR /app

# Dependencias del sistema (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Stage: Dependencies ---
FROM base AS dependencies

# Copiar requirements primero (cache de Docker)
COPY requirements_prod.txt .

RUN pip install --no-cache-dir -r requirements_prod.txt

# --- Stage: Application ---
FROM dependencies AS application

# Copiar codigo fuente
COPY *.py ./
COPY train_config.yaml ./config/train_config.yaml

# Copiar dashboard
COPY production/ ./production/

# Crear directorios
RUN mkdir -p modelos config logs

# Copiar modelos si existen (opcional, se pueden montar como volumen)
# Usamos un .dockerignore-friendly approach: copiamos todo el directorio
# Los modelos .pt se pueden montar como volumen en producción
COPY modelos/ ./modelos/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Puerto de la API
EXPOSE 8080

# Entrypoint
COPY docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--api"]
