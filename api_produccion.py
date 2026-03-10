"""
ATLAS Pro - API de Producción v4.0 (FastAPI + WebSocket + Auth)
================================================================
Servidor de producción completo con:
- API REST para control, configuración y métricas
- WebSocket para dashboard en tiempo real
- Autenticación JWT con roles (admin/operador/visor)
- Histórico de datos con TimescaleDB
- Generación de reportes PDF
- Simulación de datos realista para demo
- Integración con todos los módulos ATLAS Pro
"""

import os
import sys
import json
import time
import asyncio
import logging
import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger("ATLAS.API")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Header, Depends
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_DISPONIBLE = True
except ImportError:
    FASTAPI_DISPONIBLE = False
    logger.warning("FastAPI no disponible. Instalar con: pip install fastapi uvicorn")

try:
    from auth import AuthManager, init_auth, get_auth, get_current_user, require_role, ROLES
    AUTH_DISPONIBLE = True
except ImportError:
    AUTH_DISPONIBLE = False
    logger.warning("Módulo auth no disponible")


# =============================================================================
# IMPORTAR MÓDULOS ATLAS
# =============================================================================

try:
    from sistema_seguridad import ControladorSeguridad, ConfiguracionSeguridad
    SEGURIDAD_DISPONIBLE = True
except ImportError:
    SEGURIDAD_DISPONIBLE = False

try:
    from anomalias_alertas import SistemaAlertas, HealthMonitor, MetricaTrafico, SeveridadAlerta, TipoAnomalia
    ANOMALIAS_DISPONIBLE = True
except ImportError:
    ANOMALIAS_DISPONIBLE = False

try:
    from motor_xai import MotorXAI
    XAI_DISPONIBLE = True
except ImportError:
    XAI_DISPONIBLE = False

try:
    from checkpoint_manager import CheckpointManager
    CHECKPOINTS_DISPONIBLE = True
except ImportError:
    CHECKPOINTS_DISPONIBLE = False

try:
    from muse_metacognicion import MUSEController
    MUSE_DISPONIBLE = True
except ImportError:
    MUSE_DISPONIBLE = False


# =============================================================================
# SIMULACIÓN REALISTA DE TRÁFICO
# =============================================================================

class TrafficSimulator:
    """
    Simula tráfico realista con patrones diurnos, eventos y variabilidad.
    Genera datos coherentes para el dashboard de demostración.
    """

    def __init__(self):
        self.time_offset = 0
        self.event_active = False
        self.event_start = None
        self.incident_active = False

        # Historial circular para gráficos
        self.history_size = 120  # 2 minutos a 1 dato/seg
        self.throughput_history = deque(maxlen=self.history_size)
        self.wait_history = deque(maxlen=self.history_size)
        self.queue_history = deque(maxlen=self.history_size)
        self.reward_history = deque(maxlen=self.history_size)
        self.co2_history = deque(maxlen=self.history_size)

        # Modelos entrenados (simulación de rendimiento de cada ronda)
        self.scenarios = {
            "normal": {"baseline": 45, "ai_improvement": 0.32},
            "avenida": {"baseline": 180, "ai_improvement": 0.41},
            "evento": {"baseline": -15, "ai_improvement": 0.18},
            "heavy": {"baseline": 80, "ai_improvement": 0.28},
            "noche": {"baseline": 350, "ai_improvement": 0.65},
            "emergencias": {"baseline": 300, "ai_improvement": 0.55},
        }

        self.current_scenario = "normal"
        self._step = 0

    def _time_factor(self) -> float:
        """Factor multiplicador según hora del día (simula patrones reales)"""
        # Simula hora acelerada (1 min real = 1 hora simulada)
        hour = (self._step / 60) % 24
        # Picos: 8-9am, 17-19pm
        if 7.5 <= hour <= 9.5:
            return 1.8 + 0.3 * math.sin((hour - 7.5) * math.pi / 2)
        elif 16.5 <= hour <= 19.5:
            return 1.9 + 0.4 * math.sin((hour - 16.5) * math.pi / 3)
        elif 22 <= hour or hour <= 5:
            return 0.3 + 0.1 * math.sin(hour * math.pi / 12)
        else:
            return 1.0

    def _event_factor(self) -> float:
        """Factor de evento especial"""
        if self.event_active:
            elapsed = (self._step - self.event_start) if self.event_start else 0
            if elapsed > 300:  # Eventos duran ~5 min
                self.event_active = False
                return 1.0
            return 1.5 + 0.5 * math.sin(elapsed * math.pi / 150)
        # Probabilidad de nuevo evento
        if random.random() < 0.002:  # ~cada 8 min
            self.event_active = True
            self.event_start = self._step
            self.current_scenario = "evento"
            return 1.5
        return 1.0

    def generate_step(self) -> Dict:
        """Genera un paso de simulación con datos realistas"""
        self._step += 1
        tf = self._time_factor()
        ef = self._event_factor()
        combined = tf * ef

        # Auto-change scenario based on conditions
        if not self.event_active:
            hour = (self._step / 60) % 24
            if 22 <= hour or hour <= 5:
                self.current_scenario = "noche"
            elif combined > 1.6:
                self.current_scenario = "heavy"
            else:
                self.current_scenario = random.choice(["normal", "avenida"])

        # Throughput (vehículos/ciclo) — IA mejora flujo
        base_throughput = 25 + 15 * combined
        ai_bonus = base_throughput * self.scenarios[self.current_scenario]["ai_improvement"]
        throughput = base_throughput + ai_bonus + random.gauss(0, 3)
        throughput = max(5, throughput)

        # Wait time (segundos promedio)
        base_wait = 15 + 25 * combined
        ai_reduction = base_wait * 0.35  # IA reduce 35% espera
        avg_wait = base_wait - ai_reduction + random.gauss(0, 4)
        avg_wait = max(3, avg_wait)

        # Queues por dirección
        queue_base = int(8 * combined)
        queues = {
            "N": max(0, queue_base + random.randint(-3, 8)),
            "S": max(0, queue_base + random.randint(-3, 6)),
            "E": max(0, int(queue_base * 0.7) + random.randint(-2, 5)),
            "W": max(0, int(queue_base * 0.7) + random.randint(-2, 5)),
        }
        total_queue = sum(queues.values())

        # Reward (recompensa del agente)
        reward = throughput * 0.8 - total_queue * 0.3 - avg_wait * 0.2 + random.gauss(0, 2)

        # CO2 reduction estimate (kg/hora)
        co2_reduction = throughput * 0.015 * (1 + self.scenarios[self.current_scenario]["ai_improvement"])

        # Latencia de inferencia (ms)
        latency = 8 + random.gauss(0, 2) + (3 if combined > 1.5 else 0)
        latency = max(2, latency)

        # Confianza del modelo
        scenario_data = self.scenarios[self.current_scenario]
        base_confidence = 0.7 + scenario_data["ai_improvement"] * 0.3
        confidence = min(0.99, max(0.4, base_confidence + random.gauss(0, 0.05)))

        # Detections (vehículos detectados por sensores)
        detections = int(throughput * 2.5 + random.randint(-5, 15))

        # MUSE metrics
        muse_interventions = 1 if random.random() < 0.03 else 0
        muse_competence = min(1.0, 0.75 + self._step * 0.0001 + random.gauss(0, 0.02))

        # Fase actual del semáforo
        phase = (self._step // 30) % 4  # Cambia cada 30 seg
        phase_names = ["N-S Verde", "N-S Ámbar", "E-O Verde", "E-O Ámbar"]

        # Decisión IA
        actions = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]
        if queues["N"] + queues["S"] > queues["E"] + queues["W"]:
            action_idx = 1
        elif queues["E"] + queues["W"] > queues["N"] + queues["S"]:
            action_idx = 2
        elif avg_wait > 35:
            action_idx = 3
        else:
            action_idx = 0
        decision = actions[action_idx]

        # Guardar historial
        self.throughput_history.append(round(throughput, 1))
        self.wait_history.append(round(avg_wait, 1))
        self.queue_history.append(total_queue)
        self.reward_history.append(round(reward, 2))
        self.co2_history.append(round(co2_reduction, 3))

        hour = (self._step / 60) % 24

        return {
            "type": "metrics",
            "timestamp": datetime.now().isoformat(),
            "simulated_hour": round(hour, 1),
            "scenario": self.current_scenario,
            "phase": phase,
            "phase_name": phase_names[phase],
            "throughput": round(throughput, 1),
            "avg_wait": round(avg_wait, 1),
            "total_queue": total_queue,
            "queues": queues,
            "reward": round(reward, 2),
            "co2_reduction": round(co2_reduction, 3),
            "latency_ms": round(latency, 1),
            "confidence": round(confidence, 3),
            "detections": detections,
            "decision": decision,
            "action_index": action_idx,
            "event_active": self.event_active,
            "incident_active": self.incident_active,
            "muse": {
                "interventions": muse_interventions,
                "competence": round(muse_competence, 3),
            },
            "traffic_factor": round(combined, 2),
        }

    def get_history(self) -> Dict:
        """Retorna todo el historial para inicialización de gráficos"""
        return {
            "type": "history",
            "throughput": list(self.throughput_history),
            "wait": list(self.wait_history),
            "queue": list(self.queue_history),
            "reward": list(self.reward_history),
            "co2": list(self.co2_history),
        }

    def get_scenario_performance(self) -> Dict:
        """Rendimiento por escenario (datos de entrenamiento reales de ATLAS)"""
        return {
            "normal": {
                "best_reward": 45.2, "ronda": "R2",
                "improvement_vs_fixed": "+32%", "episodes_trained": 1100
            },
            "avenida": {
                "best_reward": 257.4, "ronda": "R4",
                "improvement_vs_fixed": "+41%", "episodes_trained": 1350
            },
            "evento": {
                "best_reward": 5.8, "ronda": "R4",
                "improvement_vs_fixed": "+18%", "episodes_trained": 1200
            },
            "heavy": {
                "best_reward": 118.1, "ronda": "R4",
                "improvement_vs_fixed": "+28%", "episodes_trained": 1150
            },
            "noche": {
                "best_reward": 773.5, "ronda": "R3",
                "improvement_vs_fixed": "+65%", "episodes_trained": 950
            },
            "emergencias": {
                "best_reward": 424.4, "ronda": "R4",
                "improvement_vs_fixed": "+55%", "episodes_trained": 1050
            },
        }


# =============================================================================
# ESTADO GLOBAL DEL SISTEMA
# =============================================================================

class AtlasSystemState:
    """Estado global del sistema ATLAS"""

    def __init__(self):
        self.mode = "ia_activa"
        self.phase = 0
        self.phase_time = 0
        self.start_time = datetime.now()
        self.total_decisions = 0
        self.current_metrics = {}
        self.websocket_clients: Set[WebSocket] = set()
        self.simulator = TrafficSimulator()

        # Alertas recientes
        self.alerts: List[Dict] = []
        self.alert_id_counter = 0

        # Inicializar subsistemas
        if SEGURIDAD_DISPONIBLE:
            self.seguridad = ControladorSeguridad()
        else:
            self.seguridad = None

        if ANOMALIAS_DISPONIBLE:
            self.alertas_system = SistemaAlertas()
            self.health_monitor = HealthMonitor(self.alertas_system)
        else:
            self.alertas_system = None
            self.health_monitor = None

        if CHECKPOINTS_DISPONIBLE:
            self.checkpoint_manager = CheckpointManager()
        else:
            self.checkpoint_manager = None

        logger.info("AtlasSystemState v3.0 inicializado")

    def add_alert(self, severity: str, message: str, source: str = "system"):
        self.alert_id_counter += 1
        alert = {
            "id": self.alert_id_counter,
            "severity": severity,
            "message": message,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "resolved": False
        }
        self.alerts.append(alert)
        # Keep last 100
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        return alert

    async def broadcast(self, message: Dict):
        """Envía mensaje a todos los clientes WebSocket"""
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.add(client)
        self.websocket_clients -= disconnected


# =============================================================================
# API FASTAPI
# =============================================================================

if FASTAPI_DISPONIBLE:

    app = FastAPI(
        title="ATLAS Pro API",
        description="API de producción para el sistema de control de semáforos inteligente ATLAS",
        version="3.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Montar archivos estáticos
    production_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "production")
    os.makedirs(production_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=production_dir), name="static")

    # Estado global
    system = AtlasSystemState()

    # --------- DASHBOARD ---------

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Sirve el dashboard de producción"""
        dashboard_path = os.path.join(production_dir, "dashboard.html")
        if os.path.exists(dashboard_path):
            return FileResponse(dashboard_path)
        return HTMLResponse("""
        <html><body style='background:#0a0e17;color:#fff;font-family:sans-serif;
        display:flex;justify-content:center;align-items:center;height:100vh'>
        <h1>ATLAS Pro - Dashboard no encontrado en /production/dashboard.html</h1>
        </body></html>""")

    # --------- WEBSOCKET ---------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket para datos en tiempo real"""
        await websocket.accept()
        system.websocket_clients.add(websocket)
        logger.info(f"WebSocket conectado. Clientes: {len(system.websocket_clients)}")

        # Enviar historial al conectar
        try:
            await websocket.send_json(system.simulator.get_history())
        except Exception:
            pass

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message.get('type') == 'command':
                    await handle_command(message, websocket)
        except WebSocketDisconnect:
            system.websocket_clients.discard(websocket)
            logger.info(f"WebSocket desconectado. Clientes: {len(system.websocket_clients)}")

    async def handle_command(message: Dict, websocket: WebSocket):
        """Procesa comandos del dashboard"""
        cmd = message.get('command')

        if cmd == 'change_mode':
            new_mode = message.get('mode', 'ia_activa')
            system.mode = new_mode
            await system.broadcast({'type': 'mode_change', 'mode': new_mode})

        elif cmd == 'manual_phase':
            phase = message.get('phase', 0)
            system.phase = phase
            await system.broadcast({'type': 'phase_change', 'phase': phase})

        elif cmd == 'get_status':
            await websocket.send_json({
                'type': 'status',
                'mode': system.mode,
                'phase': system.phase,
                'uptime': (datetime.now() - system.start_time).total_seconds(),
                'total_decisions': system.total_decisions,
                'clients': len(system.websocket_clients)
            })

        elif cmd == 'get_history':
            await websocket.send_json(system.simulator.get_history())

        elif cmd == 'get_scenarios':
            await websocket.send_json({
                'type': 'scenarios',
                'data': system.simulator.get_scenario_performance()
            })

    # --------- API REST ENDPOINTS ---------

    @app.get("/api/status")
    async def get_status():
        """Estado general del sistema"""
        return {
            "system": "ATLAS Pro",
            "version": "3.0.0",
            "mode": system.mode,
            "phase": system.phase,
            "uptime_seconds": round((datetime.now() - system.start_time).total_seconds(), 1),
            "total_decisions": system.total_decisions,
            "websocket_clients": len(system.websocket_clients),
            "current_scenario": system.simulator.current_scenario,
            "event_active": system.simulator.event_active,
            "modules": {
                "seguridad": SEGURIDAD_DISPONIBLE,
                "anomalias": ANOMALIAS_DISPONIBLE,
                "xai": XAI_DISPONIBLE,
                "checkpoints": CHECKPOINTS_DISPONIBLE,
                "muse": MUSE_DISPONIBLE,
            }
        }

    @app.get("/api/health")
    async def health_check():
        """Health check del sistema"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system.start_time).total_seconds() / 3600, 2),
        }
        if system.health_monitor:
            try:
                health.update(system.health_monitor.health_check())
            except Exception:
                pass
        return health

    @app.get("/api/metrics")
    async def get_metrics():
        """Métricas actuales"""
        return system.current_metrics

    @app.get("/api/metrics/history")
    async def get_metrics_history():
        """Historial de métricas para gráficos"""
        return system.simulator.get_history()

    @app.get("/api/scenarios")
    async def get_scenarios():
        """Rendimiento por escenario de entrenamiento"""
        return system.simulator.get_scenario_performance()

    @app.get("/api/alerts")
    async def get_alerts(
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = Query(default=50, le=200)
    ):
        """Obtener alertas"""
        alerts = system.alerts
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a["resolved"] == resolved]
        return {"alerts": alerts[-limit:], "total": len(alerts)}

    @app.post("/api/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: int):
        """Resolver alerta"""
        for alert in system.alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                return {"message": f"Alerta {alert_id} resuelta"}
        raise HTTPException(404, f"Alerta {alert_id} no encontrada")

    @app.get("/api/model/info")
    async def get_model_info():
        """Información del modelo actual"""
        return {
            "architecture": "Dueling DDQN + PER + Noisy Networks",
            "state_dim": 26,
            "action_dim": 4,
            "network": "[512, 256, 256, 128]",
            "training_rounds": 5,
            "metacognition": "MUSE v2",
            "best_scenarios": system.simulator.get_scenario_performance(),
            "actions": {
                0: "Mantener Fase",
                1: "Cambiar a N-S",
                2: "Cambiar a E-O",
                3: "Extender Fase"
            }
        }

    @app.get("/api/infrastructure")
    async def get_infrastructure():
        """Estado de infraestructura"""
        return {
            "protocols": {
                "ntcip": {"status": "available", "version": "NTCIP 1202"},
                "utmc": {"status": "available", "version": "EN 12675"},
            },
            "sensors": {
                "inductive_loops": True,
                "cameras": True,
                "radar": True,
                "bluetooth": True,
                "gps": True
            },
            "deployment": {
                "docker": True,
                "ota_updates": True,
                "edge_devices": ["Raspberry Pi 4", "Jetson Nano", "x64 Server"]
            }
        }

    @app.post("/api/mode/{mode}")
    async def change_mode(mode: str):
        """Cambiar modo de operación"""
        valid_modes = ['ia_activa', 'fallback', 'manual', 'mantenimiento']
        if mode not in valid_modes:
            raise HTTPException(400, f"Modo inválido. Opciones: {valid_modes}")
        system.mode = mode
        await system.broadcast({'type': 'mode_change', 'mode': mode})
        return {"message": f"Modo cambiado a {mode}", "mode": mode}

    @app.get("/api/statistics")
    async def get_statistics():
        """Estadísticas completas"""
        return {
            "uptime_hours": round(
                (datetime.now() - system.start_time).total_seconds() / 3600, 2
            ),
            "total_decisions": system.total_decisions,
            "current_scenario": system.simulator.current_scenario,
            "alerts_total": len(system.alerts),
            "alerts_unresolved": len([a for a in system.alerts if not a["resolved"]]),
            "avg_throughput_2min": round(
                np.mean(list(system.simulator.throughput_history) or [0]), 1
            ),
            "avg_wait_2min": round(
                np.mean(list(system.simulator.wait_history) or [0]), 1
            ),
            "avg_reward_2min": round(
                np.mean(list(system.simulator.reward_history) or [0]), 2
            ),
        }

    # --------- AUTENTICACIÓN ---------

    if AUTH_DISPONIBLE:
        # Inicializar auth
        db_url = os.environ.get("ATLAS_DB_URL")
        jwt_secret = os.environ.get("ATLAS_JWT_SECRET", "atlas-pro-jwt-secret-2026")
        auth_manager = init_auth(secret_key=jwt_secret, db_url=db_url)

        from fastapi import Request

        @app.post("/api/auth/login")
        async def auth_login(request: Request):
            """Autenticar usuario y obtener JWT token"""
            body = await request.json()
            username = body.get("username", "")
            password = body.get("password", "")
            if not username or not password:
                raise HTTPException(400, "Se requiere username y password")
            result = auth_manager.login(username, password)
            if result is None:
                raise HTTPException(401, "Credenciales inválidas")
            return result

        @app.get("/api/auth/me")
        async def auth_me(authorization: str = Header(None)):
            """Obtener datos del usuario autenticado"""
            user = await get_current_user(authorization)
            full_user = auth_manager.user_store.get_user(user["username"])
            return full_user or user

        @app.get("/api/auth/roles")
        async def auth_roles():
            """Listar roles disponibles y sus permisos"""
            return ROLES

        @app.get("/api/users")
        async def list_users(authorization: str = Header(None)):
            """Listar usuarios (solo admin)"""
            user = await get_current_user(authorization)
            if user["role"] != "admin":
                raise HTTPException(403, "Solo administradores pueden ver usuarios")
            return {"users": auth_manager.user_store.list_users()}

        @app.post("/api/users")
        async def create_user(request: Request, authorization: str = Header(None)):
            """Crear usuario (solo admin)"""
            user = await get_current_user(authorization)
            if user["role"] != "admin":
                raise HTTPException(403, "Solo administradores pueden crear usuarios")
            body = await request.json()
            ok = auth_manager.user_store.create_user(
                username=body.get("username", ""),
                password=body.get("password", ""),
                role=body.get("role", "visor"),
                nombre=body.get("nombre", ""),
                email=body.get("email", ""),
            )
            if not ok:
                raise HTTPException(400, "No se pudo crear el usuario (ya existe o rol inválido)")
            return {"message": f"Usuario {body['username']} creado"}

        @app.put("/api/users/{username}")
        async def update_user(username: str, request: Request, authorization: str = Header(None)):
            """Actualizar usuario (solo admin)"""
            user = await get_current_user(authorization)
            if user["role"] != "admin":
                raise HTTPException(403, "Solo administradores pueden editar usuarios")
            body = await request.json()
            ok = auth_manager.user_store.update_user(
                username=username,
                role=body.get("role"),
                activo=body.get("activo"),
                nombre=body.get("nombre"),
            )
            if not ok:
                raise HTTPException(404, f"Usuario {username} no encontrado")
            return {"message": f"Usuario {username} actualizado"}

    # --------- HISTÓRICO (TimescaleDB) ---------

    @app.get("/api/history/hourly")
    async def get_hourly_history(
        hours: int = Query(default=24, le=168),
        intersection_id: str = Query(default="INT_001"),
    ):
        """Resumen por hora de las últimas N horas"""
        db_url = os.environ.get("ATLAS_DB_URL")
        if db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(db_url)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            time_bucket('1 hour', timestamp) AS hora,
                            AVG(reward) AS avg_reward,
                            AVG(avg_speed) AS avg_speed,
                            AVG(queue_length) AS avg_queue,
                            AVG(wait_time) AS avg_wait,
                            SUM(vehicle_count) AS total_vehicles,
                            COUNT(*) FILTER (WHERE has_emergency) AS emergencias
                        FROM metricas_trafico
                        WHERE timestamp > NOW() - INTERVAL '%s hours'
                          AND intersection_id = %s
                        GROUP BY hora
                        ORDER BY hora DESC
                    """, (hours, intersection_id))
                    columns = [d[0] for d in cur.description]
                    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
                conn.close()
                return {"data": rows, "hours": hours, "source": "timescaledb"}
            except Exception as e:
                logger.warning(f"Error consultando TimescaleDB: {e}")

        # Fallback: generar datos simulados
        data = []
        now = datetime.now()
        for i in range(hours):
            hour = now - timedelta(hours=i)
            h = hour.hour
            # Simular patrón diurno
            if 7 <= h <= 9 or 17 <= h <= 19:
                factor = 1.8
            elif 22 <= h or h <= 5:
                factor = 0.3
            else:
                factor = 1.0
            data.append({
                "hora": hour.replace(minute=0, second=0).isoformat(),
                "avg_reward": round(50 * factor + random.gauss(0, 10), 2),
                "avg_speed": round(35 / factor + random.gauss(0, 3), 1),
                "avg_queue": round(8 * factor + random.gauss(0, 2), 1),
                "avg_wait": round(20 * factor + random.gauss(0, 5), 1),
                "total_vehicles": int(800 * factor + random.gauss(0, 50)),
                "emergencias": random.randint(0, 2) if factor > 1.5 else 0,
            })
        return {"data": data, "hours": hours, "source": "simulation"}

    @app.get("/api/history/daily")
    async def get_daily_history(
        days: int = Query(default=30, le=90),
        intersection_id: str = Query(default="INT_001"),
    ):
        """Resumen diario de los últimos N días"""
        db_url = os.environ.get("ATLAS_DB_URL")
        if db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(db_url)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            time_bucket('1 day', timestamp) AS dia,
                            AVG(reward) AS avg_reward,
                            AVG(avg_speed) AS avg_speed,
                            AVG(queue_length) AS avg_queue,
                            AVG(wait_time) AS avg_wait,
                            SUM(vehicle_count) AS total_vehicles,
                            COUNT(*) FILTER (WHERE has_emergency) AS emergencias
                        FROM metricas_trafico
                        WHERE timestamp > NOW() - INTERVAL '%s days'
                          AND intersection_id = %s
                        GROUP BY dia
                        ORDER BY dia DESC
                    """, (days, intersection_id))
                    columns = [d[0] for d in cur.description]
                    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
                conn.close()
                return {"data": rows, "days": days, "source": "timescaledb"}
            except Exception as e:
                logger.warning(f"Error consultando TimescaleDB: {e}")

        # Fallback: datos simulados
        data = []
        now = datetime.now()
        for i in range(days):
            day = now - timedelta(days=i)
            is_weekend = day.weekday() >= 5
            factor = 0.7 if is_weekend else 1.0
            data.append({
                "dia": day.strftime("%Y-%m-%d"),
                "avg_reward": round(55 * factor + random.gauss(0, 8), 2),
                "avg_speed": round(32 * (1 + (1 - factor) * 0.3) + random.gauss(0, 2), 1),
                "avg_queue": round(10 * factor + random.gauss(0, 2), 1),
                "avg_wait": round(22 * factor + random.gauss(0, 4), 1),
                "total_vehicles": int(18000 * factor + random.gauss(0, 1000)),
                "emergencias": random.randint(0, 5),
            })
        return {"data": data, "days": days, "source": "simulation"}

    @app.get("/api/history/actions")
    async def get_actions_log(
        limit: int = Query(default=100, le=1000),
        source_filter: Optional[str] = None,
    ):
        """Log de acciones para auditoría"""
        db_url = os.environ.get("ATLAS_DB_URL")
        if db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(db_url)
                with conn.cursor() as cur:
                    query = """
                        SELECT timestamp, intersection_id, action, previous_phase,
                               time_in_phase, reward, model_version, muse_strategy, source
                        FROM acciones_log
                    """
                    params = []
                    if source_filter:
                        query += " WHERE source = %s"
                        params.append(source_filter)
                    query += " ORDER BY timestamp DESC LIMIT %s"
                    params.append(limit)
                    cur.execute(query, params)
                    columns = [d[0] for d in cur.description]
                    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
                conn.close()
                return {"actions": rows, "source": "timescaledb"}
            except Exception as e:
                logger.warning(f"Error consultando acciones: {e}")

        # Fallback: simular
        actions_names = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]
        sources = ["rl_agent", "rl_agent", "rl_agent", "rl_agent", "fallback", "manual"]
        strategies = ["exploit", "exploit", "explore", "exploit", "fallback", "muse_override"]
        data = []
        now = datetime.now()
        for i in range(min(limit, 100)):
            ts = now - timedelta(seconds=i * 30)
            action_idx = random.randint(0, 3)
            data.append({
                "timestamp": ts.isoformat(),
                "intersection_id": "INT_001",
                "action": action_idx,
                "action_name": actions_names[action_idx],
                "previous_phase": random.randint(0, 3),
                "time_in_phase": round(random.uniform(10, 60), 1),
                "reward": round(random.gauss(50, 30), 2),
                "model_version": "v1.0.0",
                "muse_strategy": random.choice(strategies),
                "source": random.choice(sources),
            })
        return {"actions": data, "source": "simulation"}

    # --------- REPORTES PDF ---------

    @app.get("/api/reports/daily")
    async def generate_daily_report(
        date: Optional[str] = None,
        intersection_id: str = Query(default="INT_001"),
    ):
        """Generar reporte PDF diario"""
        try:
            from reportes_pdf import generar_reporte_diario
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")

            # Obtener datos
            history_resp = await get_hourly_history(hours=24, intersection_id=intersection_id)
            alerts_resp = await get_alerts(limit=50)
            stats_resp = await get_statistics()
            scenarios_resp = await get_scenarios()

            pdf_bytes = generar_reporte_diario(
                fecha=date,
                intersection_id=intersection_id,
                datos_horarios=history_resp["data"],
                alertas=alerts_resp["alerts"],
                estadisticas=stats_resp,
                escenarios=scenarios_resp,
            )

            return StreamingResponse(
                iter([pdf_bytes]),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=atlas_report_{date}.pdf"
                }
            )
        except ImportError:
            raise HTTPException(501, "Módulo reportes_pdf no disponible")
        except Exception as e:
            raise HTTPException(500, f"Error generando reporte: {str(e)}")

    # --------- SIMULACIÓN EN TIEMPO REAL ---------

    async def simulation_loop():
        """Loop de simulación para demo con datos realistas"""
        while True:
            metrics = system.simulator.generate_step()
            system.current_metrics = metrics
            system.total_decisions += 1
            system.phase = metrics["phase"]

            # Broadcast datos principales
            await system.broadcast(metrics)

            # Generar alertas periódicas
            if random.random() < 0.01:  # ~cada 100 seg
                severities = ["info", "warning", "critical"]
                messages = [
                    "Sensor norte: latencia elevada (>50ms)",
                    "Cola excesiva detectada en dirección E",
                    "MUSE intervino: competencia baja en escenario evento",
                    "Pico de tráfico detectado: +40% sobre media",
                    "Actualización OTA disponible: v1.2.3",
                    "Ciclo de semáforo optimizado automáticamente",
                ]
                sev = random.choice(severities)
                msg = random.choice(messages)
                alert = system.add_alert(sev, msg, "simulation")
                await system.broadcast({"type": "alert", **alert})

            # XAI broadcast
            xai_data = {
                'type': 'xai',
                'decision': metrics["decision"],
                'action_index': metrics["action_index"],
                'confidence': metrics["confidence"],
                'explanation': f'Optimizando {metrics["scenario"]} — factor tráfico: {metrics["traffic_factor"]}x',
                'scenario': metrics["scenario"],
            }
            await system.broadcast(xai_data)

            await asyncio.sleep(1)

    @app.on_event("startup")
    async def startup_event():
        """Iniciar loop de simulación"""
        asyncio.create_task(simulation_loop())
        logger.info("ATLAS Pro API v3.0 iniciada — Dashboard disponible en /")

    # --------- MAIN ---------

    def start_server(host: str = "0.0.0.0", port: int = 8000):
        """Inicia el servidor"""
        print()
        print("=" * 62)
        print("  ATLAS Pro v3.0 — Sistema de Control de Trafico Inteligente")
        print("=" * 62)
        print(f"\n  Dashboard:  http://localhost:{port}")
        print(f"  API Docs:   http://localhost:{port}/docs")
        print(f"  WebSocket:  ws://localhost:{port}/ws")
        print(f"  Metricas:   http://localhost:{port}/api/metrics")
        print(f"\n  Modulos activos:")
        print(f"    Seguridad:     {'SI' if SEGURIDAD_DISPONIBLE else 'NO'}")
        print(f"    Anomalias:     {'SI' if ANOMALIAS_DISPONIBLE else 'NO'}")
        print(f"    XAI:           {'SI' if XAI_DISPONIBLE else 'NO'}")
        print(f"    Checkpoints:   {'SI' if CHECKPOINTS_DISPONIBLE else 'NO'}")
        print(f"    MUSE:          {'SI' if MUSE_DISPONIBLE else 'NO'}")
        print(f"    Auth JWT:      {'SI' if AUTH_DISPONIBLE else 'NO'}")
        print()

        uvicorn.run(app, host=host, port=port, log_level="info")

else:
    def start_server(*args, **kwargs):
        print("FastAPI no disponible. Instalar con: pip install fastapi uvicorn")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS Pro API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto (default: 8000)")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
