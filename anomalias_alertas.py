"""
ATLAS Pro - Sistema de Detección de Anomalías y Alertas
========================================================
Monitoreo inteligente del sistema:
- Detección de anomalías en tráfico (estadística + ML)
- Alertas en tiempo real por niveles de severidad
- Detección de patrones inusuales
- Health checks del sistema
- Registro de incidentes y métricas históricas
"""

import os
import json
import time
import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np

logger = logging.getLogger("ATLAS.Anomalias")


# =============================================================================
# TIPOS Y ENUMS
# =============================================================================

class SeveridadAlerta(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TipoAnomalia(Enum):
    COLA_EXCESIVA = "cola_excesiva"
    ESPERA_PROLONGADA = "espera_prolongada"
    CAIDA_THROUGHPUT = "caida_throughput"
    PATRON_INUSUAL = "patron_inusual"
    MODELO_DEGRADADO = "modelo_degradado"
    LATENCIA_ALTA = "latencia_alta"
    SENSOR_FALLO = "sensor_fallo"
    CONGESTION_PROPAGADA = "congestion_propagada"


@dataclass
class Alerta:
    """Estructura de una alerta"""
    id: str
    timestamp: str
    severidad: str
    tipo: str
    mensaje: str
    detalles: Dict
    interseccion: str = "global"
    resuelta: bool = False
    resolucion_timestamp: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class MetricaTrafico:
    """Snapshot de métricas de tráfico"""
    timestamp: str
    colas: Dict[str, int]
    esperas: Dict[str, float]
    throughput: float
    velocidad_media: float
    fase_actual: int
    latencia_ia_ms: float
    q_value: float = 0.0
    confianza: float = 0.0


# =============================================================================
# DETECTOR DE ANOMALÍAS ESTADÍSTICO
# =============================================================================

class DetectorAnomalias:
    """
    Detector de anomalías usando métodos estadísticos:
    - Z-Score para valores individuales
    - Moving Average para tendencias
    - IQR para outliers robustos
    - EWMA para detección adaptativa
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0,
                 iqr_factor: float = 1.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor

        # Historial de métricas
        self.history: Dict[str, deque] = {}
        self.ewma: Dict[str, float] = {}
        self.ewma_var: Dict[str, float] = {}
        self.alpha = 0.1  # Factor de suavizado EWMA

    def update(self, metric_name: str, value: float) -> Optional[Dict]:
        """
        Actualiza una métrica y detecta anomalías.

        Returns:
            Dict con info de anomalía si se detecta, None si normal
        """
        if metric_name not in self.history:
            self.history[metric_name] = deque(maxlen=self.window_size)
            self.ewma[metric_name] = value
            self.ewma_var[metric_name] = 0.0

        self.history[metric_name].append(value)
        values = np.array(self.history[metric_name])

        # Actualizar EWMA
        old_ewma = self.ewma[metric_name]
        self.ewma[metric_name] = self.alpha * value + (1 - self.alpha) * old_ewma
        diff = value - old_ewma
        self.ewma_var[metric_name] = (1 - self.alpha) * (self.ewma_var[metric_name] + self.alpha * diff ** 2)

        if len(values) < 10:
            return None

        anomaly = None

        # Z-Score
        mean = np.mean(values)
        std = np.std(values)
        if std > 0:
            z_score = abs(value - mean) / std
            if z_score > self.z_threshold:
                anomaly = {
                    'method': 'z_score',
                    'metric': metric_name,
                    'value': value,
                    'z_score': float(z_score),
                    'mean': float(mean),
                    'std': float(std)
                }

        # IQR
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        if value < lower or value > upper:
            if anomaly is None:
                anomaly = {
                    'method': 'iqr',
                    'metric': metric_name,
                    'value': value,
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr)
                }
            else:
                anomaly['iqr_confirmed'] = True

        # EWMA
        ewma_std = np.sqrt(self.ewma_var[metric_name])
        if ewma_std > 0:
            ewma_z = abs(value - self.ewma[metric_name]) / ewma_std
            if ewma_z > self.z_threshold:
                if anomaly is None:
                    anomaly = {
                        'method': 'ewma',
                        'metric': metric_name,
                        'value': value,
                        'ewma': float(self.ewma[metric_name]),
                        'ewma_z': float(ewma_z)
                    }
                else:
                    anomaly['ewma_confirmed'] = True

        return anomaly

    def get_statistics(self, metric_name: str) -> Optional[Dict]:
        """Estadísticas actuales de una métrica"""
        if metric_name not in self.history or len(self.history[metric_name]) < 2:
            return None

        values = np.array(self.history[metric_name])
        return {
            'metric': metric_name,
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'ewma': float(self.ewma.get(metric_name, 0)),
            'trend': float(np.polyfit(range(len(values)), values, 1)[0])
        }


# =============================================================================
# SISTEMA DE ALERTAS
# =============================================================================

class SistemaAlertas:
    """
    Sistema de alertas con niveles de severidad y callbacks.
    """

    def __init__(self, log_dir: str = "logs/alertas",
                 max_alertas: int = 10000):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.alertas: List[Alerta] = []
        self.alertas_activas: List[Alerta] = []
        self.max_alertas = max_alertas
        self.callbacks: Dict[str, List[Callable]] = {
            'info': [], 'warning': [], 'critical': [], 'emergency': []
        }
        self.alert_counter = 0
        self.cooldowns: Dict[str, float] = {}
        self.cooldown_seconds = 60  # Mínimo 60s entre alertas del mismo tipo

        logger.info("SistemaAlertas inicializado")

    def registrar_callback(self, severidad: str, callback: Callable):
        """Registra callback para una severidad"""
        if severidad in self.callbacks:
            self.callbacks[severidad].append(callback)

    def emitir_alerta(self, severidad: SeveridadAlerta, tipo: TipoAnomalia,
                     mensaje: str, detalles: Dict = None,
                     interseccion: str = "global") -> Optional[Alerta]:
        """
        Emite una nueva alerta si no está en cooldown.
        """
        # Cooldown check
        cooldown_key = f"{tipo.value}_{interseccion}"
        now = time.time()
        if cooldown_key in self.cooldowns:
            if now - self.cooldowns[cooldown_key] < self.cooldown_seconds:
                return None
        self.cooldowns[cooldown_key] = now

        self.alert_counter += 1
        alerta = Alerta(
            id=f"ALERT-{self.alert_counter:06d}",
            timestamp=datetime.now().isoformat(),
            severidad=severidad.value,
            tipo=tipo.value,
            mensaje=mensaje,
            detalles=detalles or {},
            interseccion=interseccion
        )

        self.alertas.append(alerta)
        self.alertas_activas.append(alerta)

        # Logging
        log_msg = f"[{severidad.value.upper()}] {tipo.value}: {mensaje}"
        if severidad == SeveridadAlerta.EMERGENCY:
            logger.critical(log_msg)
        elif severidad == SeveridadAlerta.CRITICAL:
            logger.error(log_msg)
        elif severidad == SeveridadAlerta.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Callbacks
        for callback in self.callbacks.get(severidad.value, []):
            try:
                callback(alerta)
            except Exception as e:
                logger.error(f"Error en callback de alerta: {e}")

        # Persistir
        self._persistir_alerta(alerta)

        # Limpieza
        if len(self.alertas) > self.max_alertas:
            self.alertas = self.alertas[-self.max_alertas:]

        return alerta

    def resolver_alerta(self, alert_id: str):
        """Marca una alerta como resuelta"""
        for alerta in self.alertas_activas:
            if alerta.id == alert_id:
                alerta.resuelta = True
                alerta.resolucion_timestamp = datetime.now().isoformat()
                self.alertas_activas.remove(alerta)
                logger.info(f"Alerta resuelta: {alert_id}")
                return True
        return False

    def obtener_activas(self, severidad: str = None) -> List[Dict]:
        """Obtiene alertas activas, opcionalmente filtradas por severidad"""
        alertas = self.alertas_activas
        if severidad:
            alertas = [a for a in alertas if a.severidad == severidad]
        return [a.to_dict() for a in alertas]

    def obtener_historial(self, horas: int = 24, tipo: str = None) -> List[Dict]:
        """Obtiene historial de alertas"""
        cutoff = datetime.now() - timedelta(hours=horas)
        result = []
        for alerta in reversed(self.alertas):
            ts = datetime.fromisoformat(alerta.timestamp)
            if ts < cutoff:
                break
            if tipo and alerta.tipo != tipo:
                continue
            result.append(alerta.to_dict())
        return result

    def resumen(self) -> Dict:
        """Resumen del estado de alertas"""
        activas_por_severidad = {}
        for alerta in self.alertas_activas:
            sev = alerta.severidad
            activas_por_severidad[sev] = activas_por_severidad.get(sev, 0) + 1

        return {
            'total_alertas': len(self.alertas),
            'alertas_activas': len(self.alertas_activas),
            'por_severidad': activas_por_severidad,
            'ultima_alerta': self.alertas[-1].to_dict() if self.alertas else None
        }

    def _persistir_alerta(self, alerta: Alerta):
        """Guarda alerta en archivo"""
        fecha = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(self.log_dir, f"alertas_{fecha}.jsonl")
        with open(filepath, 'a') as f:
            f.write(json.dumps(alerta.to_dict(), ensure_ascii=False) + "\n")


# =============================================================================
# MONITOR DE SALUD DEL SISTEMA
# =============================================================================

class HealthMonitor:
    """
    Monitor de salud del sistema ATLAS.
    Verifica componentes y emite alertas automáticas.
    """

    def __init__(self, alertas: SistemaAlertas, config: Dict = None):
        self.alertas = alertas
        self.config = config or self._default_config()
        self.detector = DetectorAnomalias(
            window_size=self.config.get('window_size', 100),
            z_threshold=self.config.get('z_threshold', 3.0)
        )
        self.last_check = time.time()
        self.metrics_history: List[MetricaTrafico] = []
        self.is_running = False

    def _default_config(self) -> Dict:
        return {
            'window_size': 100,
            'z_threshold': 3.0,
            'max_queue': 50,
            'max_wait_time': 180,
            'min_throughput': 10,
            'max_latency_ms': 200,
            'check_interval': 5,
            'congestion_threshold': 0.8
        }

    def process_metrics(self, metrics: MetricaTrafico):
        """
        Procesa métricas y detecta anomalías automáticamente.
        """
        self.metrics_history.append(metrics)

        # 1. Verificar colas
        for direction, queue in metrics.colas.items():
            anomaly = self.detector.update(f"cola_{direction}", queue)
            if anomaly:
                self._check_queue_alert(direction, queue, anomaly)

        # 2. Verificar esperas
        for direction, wait in metrics.esperas.items():
            anomaly = self.detector.update(f"espera_{direction}", wait)
            if anomaly:
                self._check_wait_alert(direction, wait, anomaly)

        # 3. Verificar throughput
        anomaly = self.detector.update("throughput", metrics.throughput)
        if anomaly and metrics.throughput < self.config['min_throughput']:
            self.alertas.emitir_alerta(
                SeveridadAlerta.WARNING,
                TipoAnomalia.CAIDA_THROUGHPUT,
                f"Throughput bajo: {metrics.throughput:.1f} veh/min",
                {'throughput': metrics.throughput, 'anomaly': anomaly}
            )

        # 4. Verificar latencia
        anomaly = self.detector.update("latencia", metrics.latencia_ia_ms)
        if metrics.latencia_ia_ms > self.config['max_latency_ms']:
            self.alertas.emitir_alerta(
                SeveridadAlerta.CRITICAL,
                TipoAnomalia.LATENCIA_ALTA,
                f"Latencia IA: {metrics.latencia_ia_ms:.0f}ms (máx: {self.config['max_latency_ms']}ms)",
                {'latencia_ms': metrics.latencia_ia_ms}
            )

        # 5. Verificar confianza del modelo
        if metrics.confianza > 0 and metrics.confianza < 0.3:
            self.alertas.emitir_alerta(
                SeveridadAlerta.WARNING,
                TipoAnomalia.MODELO_DEGRADADO,
                f"Confianza del modelo baja: {metrics.confianza:.1%}",
                {'confianza': metrics.confianza, 'q_value': metrics.q_value}
            )

    def _check_queue_alert(self, direction: str, queue: int, anomaly: Dict):
        if queue > self.config['max_queue'] * 1.5:
            self.alertas.emitir_alerta(
                SeveridadAlerta.EMERGENCY,
                TipoAnomalia.CONGESTION_PROPAGADA,
                f"Congestión severa en {direction}: {queue} vehículos",
                {'direction': direction, 'queue': queue, 'anomaly': anomaly}
            )
        elif queue > self.config['max_queue']:
            self.alertas.emitir_alerta(
                SeveridadAlerta.CRITICAL,
                TipoAnomalia.COLA_EXCESIVA,
                f"Cola excesiva en {direction}: {queue} vehículos",
                {'direction': direction, 'queue': queue, 'anomaly': anomaly}
            )

    def _check_wait_alert(self, direction: str, wait: float, anomaly: Dict):
        if wait > self.config['max_wait_time']:
            sev = SeveridadAlerta.CRITICAL if wait > self.config['max_wait_time'] * 1.5 \
                else SeveridadAlerta.WARNING
            self.alertas.emitir_alerta(
                sev,
                TipoAnomalia.ESPERA_PROLONGADA,
                f"Espera prolongada en {direction}: {wait:.0f}s",
                {'direction': direction, 'wait_time': wait, 'anomaly': anomaly}
            )

    def health_check(self) -> Dict:
        """Verificación completa de salud del sistema"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'overall': 'healthy',
            'components': {},
            'metrics_summary': {},
            'active_alerts': len(self.alertas.alertas_activas)
        }

        # Verificar métricas recientes
        if self.metrics_history:
            recent = self.metrics_history[-10:]
            avg_latency = np.mean([m.latencia_ia_ms for m in recent])
            avg_throughput = np.mean([m.throughput for m in recent])

            status['metrics_summary'] = {
                'avg_latency_ms': float(avg_latency),
                'avg_throughput': float(avg_throughput),
                'samples': len(recent)
            }

            if avg_latency > self.config['max_latency_ms']:
                status['overall'] = 'degraded'
                status['components']['inference'] = 'slow'
            else:
                status['components']['inference'] = 'ok'

            if avg_throughput < self.config['min_throughput']:
                status['overall'] = 'degraded'
                status['components']['traffic_flow'] = 'poor'
            else:
                status['components']['traffic_flow'] = 'ok'
        else:
            status['components']['metrics'] = 'no_data'

        # Alertas activas
        emergency_alerts = [a for a in self.alertas.alertas_activas
                          if a.severidad == 'emergency']
        if emergency_alerts:
            status['overall'] = 'critical'

        return status

    def get_statistics(self) -> Dict:
        """Estadísticas de todas las métricas monitoreadas"""
        stats = {}
        for metric_name in self.detector.history:
            stat = self.detector.get_statistics(metric_name)
            if stat:
                stats[metric_name] = stat
        return stats


# =============================================================================
# EJEMPLO
# =============================================================================

def ejemplo_anomalias():
    """Demo del sistema de anomalías y alertas"""
    print("\n" + "=" * 70)
    print("🔔 ATLAS Pro - Detección de Anomalías y Alertas")
    print("=" * 70)

    alertas = SistemaAlertas()
    monitor = HealthMonitor(alertas)

    # Callback de ejemplo
    def on_critical(alerta):
        print(f"  🚨 CRITICAL: {alerta.mensaje}")

    alertas.registrar_callback('critical', on_critical)
    alertas.registrar_callback('emergency', on_critical)

    # Simular métricas normales
    print("\n📊 Simulando tráfico normal...")
    for i in range(50):
        metrics = MetricaTrafico(
            timestamp=datetime.now().isoformat(),
            colas={'norte': np.random.randint(5, 25), 'sur': np.random.randint(5, 20),
                   'este': np.random.randint(3, 15), 'oeste': np.random.randint(3, 15)},
            esperas={'norte': np.random.uniform(20, 60), 'sur': np.random.uniform(15, 50),
                    'este': np.random.uniform(10, 40), 'oeste': np.random.uniform(10, 40)},
            throughput=np.random.uniform(30, 60),
            velocidad_media=np.random.uniform(25, 45),
            fase_actual=i % 2,
            latencia_ia_ms=np.random.uniform(10, 50),
            confianza=np.random.uniform(0.6, 0.95)
        )
        monitor.process_metrics(metrics)

    # Simular anomalías
    print("\n⚠️  Simulando anomalías...")
    for i in range(10):
        metrics = MetricaTrafico(
            timestamp=datetime.now().isoformat(),
            colas={'norte': np.random.randint(60, 100), 'sur': np.random.randint(40, 70),
                   'este': np.random.randint(5, 15), 'oeste': np.random.randint(5, 15)},
            esperas={'norte': np.random.uniform(200, 300), 'sur': np.random.uniform(150, 250),
                    'este': np.random.uniform(10, 40), 'oeste': np.random.uniform(10, 40)},
            throughput=np.random.uniform(5, 15),
            velocidad_media=np.random.uniform(5, 15),
            fase_actual=0,
            latencia_ia_ms=np.random.uniform(100, 300),
            confianza=np.random.uniform(0.1, 0.3)
        )
        monitor.process_metrics(metrics)

    # Resumen
    print(f"\n📋 Resumen de alertas:")
    resumen = alertas.resumen()
    print(f"   Total: {resumen['total_alertas']}")
    print(f"   Activas: {resumen['alertas_activas']}")
    print(f"   Por severidad: {resumen['por_severidad']}")

    # Health check
    print(f"\n🏥 Health Check:")
    health = monitor.health_check()
    print(f"   Estado: {health['overall']}")
    print(f"   Componentes: {health['components']}")

    print("\n✅ Demo completada")


if __name__ == "__main__":
    ejemplo_anomalias()
