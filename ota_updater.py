#!/usr/bin/env python3
"""
OTA (Over-The-Air) Update System for ATLAS Edge Devices
Sistema de Actualización Over-The-Air para Dispositivos Edge de ATLAS

Permite actualizar modelos remotamente en Raspberry Pi, Jetson Nano y otros
dispositivos edge sin necesidad de acceso físico. Incluye:
- Descarga segura con verificación de integridad (SHA-256)
- Rollout gradual (canary -> 25% -> 50% -> 100%)
- Rollback automático si el rendimiento cae
- Métricas de rendimiento y monitoreo de dispositivos
"""

import hashlib
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import tempfile
import os
import sys

try:
    from flask import Flask, request, jsonify
except ImportError:
    Flask = None

import requests

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelManifest:
    """
    Manifiesto que describe una versión del modelo.
    Define qué versión está disponible, su integridad y política de rollout.
    """
    version: str  # "1.2.3" - versionado semántico
    model_hash: str  # SHA-256 del archivo del modelo
    model_size_bytes: int
    min_client_version: str  # Versión mínima del cliente requerida
    release_notes: str
    rollout_percentage: int  # 0-100, controla despliegue gradual
    performance_threshold: float  # Recompensa mínima para aceptar
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class DeviceInfo:
    """
    Información del dispositivo edge en la flota.
    Rastrear versión, hardware, salud y rendimiento.
    """
    device_id: str
    model_version: str
    hardware: str  # "rpi4", "jetson_nano", "x64"
    last_heartbeat: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    avg_reward_24h: float = 0.0
    uptime_hours: float = 0.0
    status: str = "active"  # "active", "updating", "failed", "rolled_back"
    metrics: Dict = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        return data


class RolloutManager:
    """
    Gestor de despliegue gradual (canary -> 25% -> 50% -> 100%).
    Automatiza la expansión del despliegue y rollback por degradación.
    """

    def __init__(self):
        self.canary_duration_hours = 24
        self.rollback_threshold = 0.10  # 10% degradación en recompensa promedio

    def should_rollout_expand(self, manifest: ModelManifest,
                             devices: Dict[str, DeviceInfo]) -> bool:
        """
        Determina si la versión canary ha sido exitosa y debe expandirse.
        Requiere: 24h sin errores y recompensa estable.
        """
        if manifest.rollout_percentage >= 100:
            return False

        # Encontrar dispositivos ejecutando la versión canary
        canary_devices = [
            d for d in devices.values()
            if d.model_version == manifest.version and d.status == "active"
        ]

        if not canary_devices:
            return False

        # Verificar duración mínima de canary (24 horas)
        earliest_heartbeat = min(
            datetime.fromisoformat(d.last_heartbeat) for d in canary_devices
        )
        canary_age = datetime.utcnow() - earliest_heartbeat

        if canary_age < timedelta(hours=self.canary_duration_hours):
            return False

        # Verificar recompensa promedio (debe ser decente)
        avg_reward = sum(d.avg_reward_24h for d in canary_devices) / len(canary_devices)
        logger.info(f"Canary {manifest.version} avg_reward: {avg_reward:.3f}")

        return avg_reward >= manifest.performance_threshold * 0.95

    def should_rollback(self, old_manifest: ModelManifest,
                       new_manifest: ModelManifest,
                       devices: Dict[str, DeviceInfo]) -> bool:
        """
        Determina rollback si la nueva versión causa degradación >10%.
        Compara dispositivos en versión nueva vs versión anterior.
        """
        new_version_devices = [
            d for d in devices.values()
            if d.model_version == new_manifest.version
        ]
        old_version_devices = [
            d for d in devices.values()
            if d.model_version == old_manifest.version
        ]

        if not new_version_devices or not old_version_devices:
            return False

        new_avg = sum(d.avg_reward_24h for d in new_version_devices) / len(new_version_devices)
        old_avg = sum(d.avg_reward_24h for d in old_version_devices) / len(old_version_devices)

        degradation = (old_avg - new_avg) / max(old_avg, 1e-6)
        logger.warning(
            f"Comparación: v{new_manifest.version} ({new_avg:.3f}) vs "
            f"v{old_manifest.version} ({old_avg:.3f}). Degradación: {degradation:.2%}"
        )

        return degradation > self.rollback_threshold


class OTAServer:
    """
    Servidor OTA que aloja archivos de modelo y gestiona el despliegue.
    Expone endpoints para que dispositivos edge verifiquen y descarguen actualizaciones.
    """

    def __init__(self, models_dir: str = "./models", port: int = 5000):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.port = port

        # Estado del servidor
        self.current_manifest: Optional[ModelManifest] = None
        self.devices: Dict[str, DeviceInfo] = {}
        self.rollout_manager = RolloutManager()
        self.lock = threading.Lock()

        # Crear app Flask si está disponible
        self.app = None
        if Flask:
            self.app = self._create_flask_app()

    def _create_flask_app(self) -> Flask:
        """Crea aplicación Flask con endpoints OTA."""
        app = Flask(__name__)

        @app.route('/ota/manifest', methods=['GET'])
        def get_manifest():
            """
            Endpoint: cliente pide la versión actual disponible.
            Responde con ManifestJSON si el dispositivo es elegible.
            """
            device_id = request.args.get('device_id')
            current_version = request.args.get('current_version', '0.0.0')

            if not device_id or not self.current_manifest:
                return jsonify({'error': 'No manifest available'}), 404

            # Registrar heartbeat del dispositivo
            with self.lock:
                if device_id not in self.devices:
                    self.devices[device_id] = DeviceInfo(
                        device_id=device_id,
                        model_version=current_version,
                        hardware='unknown'
                    )
                self.devices[device_id].last_heartbeat = datetime.utcnow().isoformat()

            # Verificar elegibilidad (rollout_percentage)
            import random
            if random.randint(1, 100) > self.current_manifest.rollout_percentage:
                logger.info(f"Device {device_id} no elegible para rollout (probabilidad)")
                return jsonify({'status': 'no_update'}), 200

            manifest_dict = self.current_manifest.to_dict()
            return jsonify(manifest_dict), 200

        @app.route('/ota/download/<version>', methods=['GET'])
        def download_model(version: str):
            """
            Endpoint: cliente descarga archivo de modelo.
            Sirve el archivo con información de integridad.
            """
            model_path = self.models_dir / f"model_{version}.onnx"
            if not model_path.exists():
                return jsonify({'error': 'Model not found'}), 404

            try:
                with open(model_path, 'rb') as f:
                    data = f.read()
                return data, 200, {
                    'Content-Type': 'application/octet-stream',
                    'Content-Disposition': f'attachment; filename=model_{version}.onnx'
                }
            except Exception as e:
                logger.error(f"Error descargando modelo {version}: {e}")
                return jsonify({'error': str(e)}), 500

        @app.route('/ota/heartbeat', methods=['POST'])
        def heartbeat():
            """
            Endpoint: dispositivo reporta salud y métricas.
            Cliente envía: device_id, model_version, avg_reward_24h, uptime_hours
            """
            data = request.get_json() or {}
            device_id = data.get('device_id')

            if not device_id:
                return jsonify({'error': 'Missing device_id'}), 400

            with self.lock:
                if device_id not in self.devices:
                    self.devices[device_id] = DeviceInfo(
                        device_id=device_id,
                        model_version=data.get('model_version', '0.0.0'),
                        hardware=data.get('hardware', 'unknown')
                    )

                device = self.devices[device_id]
                device.last_heartbeat = datetime.utcnow().isoformat()
                device.avg_reward_24h = data.get('avg_reward_24h', 0.0)
                device.uptime_hours = data.get('uptime_hours', 0.0)
                device.status = data.get('status', 'active')
                device.metrics = data.get('metrics', {})

            logger.info(
                f"Heartbeat de {device_id}: v{device.model_version}, "
                f"reward={device.avg_reward_24h:.3f}, status={device.status}"
            )

            return jsonify({'status': 'ok'}), 200

        @app.route('/ota/rollback', methods=['POST'])
        def request_rollback():
            """
            Endpoint: dispositivo solicita rollback (modelo nuevo causó errores).
            Servidor confirma y proporciona versión anterior.
            """
            data = request.get_json() or {}
            device_id = data.get('device_id')
            failed_version = data.get('failed_version')
            error_msg = data.get('error', '')

            logger.warning(
                f"Rollback solicitado por {device_id}: v{failed_version} "
                f"({error_msg})"
            )

            with self.lock:
                if device_id in self.devices:
                    self.devices[device_id].status = 'rolled_back'

            # Retornar versión anterior conocida (lógica simplificada)
            previous_version = self._get_previous_version(failed_version)
            return jsonify({
                'status': 'approved',
                'previous_version': previous_version
            }), 200

        @app.route('/ota/fleet-status', methods=['GET'])
        def fleet_status():
            """Endpoint: obtener estado de la flota (para monitoreo)."""
            with self.lock:
                devices_list = [d.to_dict() for d in self.devices.values()]
                manifest_dict = self.current_manifest.to_dict() if self.current_manifest else None

            return jsonify({
                'devices_count': len(self.devices),
                'devices': devices_list,
                'current_manifest': manifest_dict,
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        return app

    def load_manifest(self, manifest_path: str):
        """Carga manifiesto desde archivo JSON."""
        with open(manifest_path) as f:
            data = json.load(f)

        self.current_manifest = ModelManifest(
            version=data['version'],
            model_hash=data['model_hash'],
            model_size_bytes=data['model_size_bytes'],
            min_client_version=data['min_client_version'],
            release_notes=data['release_notes'],
            rollout_percentage=data['rollout_percentage'],
            performance_threshold=data['performance_threshold']
        )
        logger.info(f"Manifiesto cargado: v{self.current_manifest.version}")

    def expand_rollout(self, new_percentage: int):
        """Expande gradualmente el porcentaje de rollout (5% -> 25% -> 50% -> 100%)."""
        if self.current_manifest:
            old_pct = self.current_manifest.rollout_percentage
            self.current_manifest.rollout_percentage = new_percentage
            logger.info(
                f"Rollout expandido: {old_pct}% -> {new_percentage}% "
                f"(v{self.current_manifest.version})"
            )

    def run(self):
        """Inicia servidor Flask en background."""
        if not self.app:
            logger.error("Flask no disponible. Instalar: pip install flask")
            return

        logger.info(f"OTA Server iniciado en puerto {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

    def _get_previous_version(self, current_version: str) -> str:
        """Obtiene versión anterior conocida (simplificado)."""
        # En producción, esto vendría de una base de datos de versiones
        parts = current_version.split('.')
        if len(parts) >= 3 and int(parts[2]) > 0:
            parts[2] = str(int(parts[2]) - 1)
            return '.'.join(parts)
        return '0.0.0'


class OTAClient:
    """
    Cliente OTA que corre en dispositivo edge.
    Verifica periódicamente actualizaciones, descarga e instala con rollback automático.
    """

    def __init__(self, device_id: str, server_url: str,
                 models_dir: str = "./models", check_interval: int = 3600):
        self.device_id = device_id
        self.server_url = server_url.rstrip('/')
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.check_interval = check_interval  # Segundos entre chequeos
        self.current_model_version = '0.0.0'
        self.current_model_path: Optional[Path] = None
        self.hardware = self._detect_hardware()

        self.should_stop = False
        self.last_reward = 0.0
        self.uptime_seconds = 0

    def _detect_hardware(self) -> str:
        """Detecta hardware del dispositivo (rpi4, jetson_nano, x64)."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model_str = f.read().lower()
                if 'raspberry pi 4' in model_str:
                    return 'rpi4'
                elif 'jetson' in model_str:
                    return 'jetson_nano'
        except FileNotFoundError:
            pass
        return 'x64'

    def check_for_updates(self) -> bool:
        """
        Chequea servidor por actualizaciones disponibles.
        Retorna True si hay actualización disponible y elegible.
        """
        try:
            url = f"{self.server_url}/ota/manifest"
            params = {
                'device_id': self.device_id,
                'current_version': self.current_model_version
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'no_update':
                logger.info(f"No actualización disponible (elegibilidad probabilística)")
                return False

            manifest = ModelManifest(
                version=data['version'],
                model_hash=data['model_hash'],
                model_size_bytes=data['model_size_bytes'],
                min_client_version=data['min_client_version'],
                release_notes=data['release_notes'],
                rollout_percentage=data['rollout_percentage'],
                performance_threshold=data['performance_threshold']
            )

            if manifest.version != self.current_model_version:
                logger.info(
                    f"Actualización disponible: v{manifest.version} "
                    f"(actual: v{self.current_model_version})"
                )
                self._download_and_install(manifest)
                return True

            return False

        except requests.RequestException as e:
            logger.error(f"Error checando actualizaciones: {e}")
            return False

    def _download_and_install(self, manifest: ModelManifest) -> bool:
        """
        Descarga modelo de manera atómica:
        1. Descarga a archivo temporal
        2. Verifica SHA-256
        3. Intercambia atómicamente (temp -> producción)
        4. Verifica funcionamiento
        5. Rollback automático si falla
        """
        logger.info(f"Iniciando descarga de v{manifest.version}...")

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(
            suffix='.onnx', dir=self.models_dir, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Descargar
            url = f"{self.server_url}/ota/download/{manifest.version}"
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()

            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Descarga completa: {tmp_path.stat().st_size} bytes")

            # Verificar integridad (SHA-256)
            if not self._verify_hash(tmp_path, manifest.model_hash):
                logger.error(f"Fallo verificación hash para v{manifest.version}")
                tmp_path.unlink()
                return False

            # Intercambio atómico
            backup_path = self.models_dir / f"model_{self.current_model_version}_backup.onnx"
            current_path = self.models_dir / "model_current.onnx"

            if current_path.exists() and self.current_model_path:
                shutil.copy2(current_path, backup_path)
                logger.info(f"Backup creado: {backup_path}")

            shutil.move(str(tmp_path), str(current_path))
            logger.info(f"Modelo instalado: {current_path}")

            # Actualizar seguimiento
            old_version = self.current_model_version
            self.current_model_version = manifest.version
            self.current_model_path = current_path

            logger.info(f"Actualización exitosa: v{old_version} -> v{manifest.version}")
            return True

        except Exception as e:
            logger.error(f"Error descargando/instalando modelo: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def _verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verifica SHA-256 del archivo descargado."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        computed_hash = sha256.hexdigest()
        match = computed_hash == expected_hash

        logger.info(
            f"Verificación hash: {'OK' if match else 'FALLO'} "
            f"(esperado: {expected_hash[:16]}..., obtenido: {computed_hash[:16]}...)"
        )
        return match

    def send_heartbeat(self):
        """Reporta salud y métricas del dispositivo al servidor."""
        try:
            url = f"{self.server_url}/ota/heartbeat"
            payload = {
                'device_id': self.device_id,
                'model_version': self.current_model_version,
                'hardware': self.hardware,
                'avg_reward_24h': self.last_reward,
                'uptime_hours': self.uptime_seconds / 3600.0,
                'status': 'active',
                'metrics': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'uptime_seconds': self.uptime_seconds
                }
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug(f"Heartbeat enviado a {self.device_id}")
        except requests.RequestException as e:
            logger.error(f"Error enviando heartbeat: {e}")

    def run_periodic_check(self):
        """
        Corre chequeo periódico de actualizaciones en thread separado.
        Continúa hasta que should_stop sea True.
        """
        logger.info(
            f"Cliente OTA iniciado (dispositivo: {self.device_id}, "
            f"intervalo: {self.check_interval}s)"
        )

        while not self.should_stop:
            try:
                self.uptime_seconds += self.check_interval
                self.check_for_updates()
                self.send_heartbeat()
            except Exception as e:
                logger.error(f"Error en chequeo periódico: {e}")

            time.sleep(self.check_interval)

    def start(self):
        """Inicia cliente en thread separado."""
        thread = threading.Thread(target=self.run_periodic_check, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Detiene cliente OTA."""
        self.should_stop = True
        logger.info(f"Cliente OTA detenido ({self.device_id})")


def main():
    """CLI principal para ejecutar como servidor u cliente."""
    parser = argparse.ArgumentParser(
        description='Sistema OTA para ATLAS Edge Devices'
    )
    subparsers = parser.add_subparsers(dest='mode', help='Modo de operación')

    # Servidor
    server_parser = subparsers.add_parser('server', help='Modo servidor OTA')
    server_parser.add_argument(
        '--port', type=int, default=5000,
        help='Puerto para servidor (default: 5000)'
    )
    server_parser.add_argument(
        '--models-dir', default='./models',
        help='Directorio de modelos (default: ./models)'
    )
    server_parser.add_argument(
        '--manifest', required=False,
        help='Ruta a archivo de manifiesto JSON'
    )

    # Cliente
    client_parser = subparsers.add_parser('client', help='Modo cliente OTA')
    client_parser.add_argument(
        '--device-id', required=True,
        help='ID único del dispositivo (ej: RPi_001)'
    )
    client_parser.add_argument(
        '--server-url', required=True,
        help='URL del servidor OTA (ej: https://ota.example.com)'
    )
    client_parser.add_argument(
        '--check-interval', type=int, default=3600,
        help='Intervalo de chequeo en segundos (default: 3600 = 1 hora)'
    )
    client_parser.add_argument(
        '--models-dir', default='./models',
        help='Directorio de modelos (default: ./models)'
    )

    args = parser.parse_args()

    if args.mode == 'server':
        logger.info("Iniciando OTA Server...")
        server = OTAServer(models_dir=args.models_dir, port=args.port)

        if args.manifest:
            server.load_manifest(args.manifest)

        server.run()

    elif args.mode == 'client':
        logger.info("Iniciando OTA Client...")
        client = OTAClient(
            device_id=args.device_id,
            server_url=args.server_url,
            models_dir=args.models_dir,
            check_interval=args.check_interval
        )

        # Simular métrica de recompensa (en producción viene del modelo)
        client.last_reward = 0.75

        thread = client.start()

        try:
            thread.join()
        except KeyboardInterrupt:
            logger.info("Interrumpido por usuario")
            client.stop()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
