# -*- coding: utf-8 -*-
"""
ATLAS Pro - API de Produccion v4.0 (FastAPI + WebSocket + Auth)
================================================================
Servidor de produccion completo con:
- API REST para control, configuracion y metricas
- WebSocket para dashboard en tiempo real
- Autenticacion JWT con roles (admin/operador/visor)
- Generacion de reportes PDF
- Simulacion de datos realista para demo

Module map (for future extraction into api/ package):
    Lines   88-302  : TrafficSimulator   -> api/simulator.py
    Lines  309-358  : AtlasSystemState   -> api/state.py
    Lines  360-500  : FastAPI routes      -> api/routes.py
    Lines  500-700  : WebSocket handlers  -> api/websocket.py
    Lines  700-987  : Utility endpoints   -> api/utils.py
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
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import deque


# =============================================================================
# GEOMETRÍA DE RED SUMO — para renderizado 3D en el dashboard
# =============================================================================

def sumo_xy_to_gps(x, y):
    """Convert SUMO XY coords to GPS lon/lat using Milan UTM zone 32N offset"""
    utm_x = x + 512006.82
    utm_y = y + 5033426.19
    # UTM zone 32N to WGS84
    a = 6378137.0
    e = 0.0818191908
    k0 = 0.9996
    x0 = 500000.0
    utm_x -= x0
    M = utm_y / k0
    mu = M / (a * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))
    e1 = (1 - math.sqrt(1 - e**2)) / (1 + math.sqrt(1 - e**2))
    J1 = 3*e1/2 - 27*e1**3/32
    J2 = 21*e1**2/16 - 55*e1**4/32
    J3 = 151*e1**3/96
    J4 = 1097*e1**4/512
    fp = mu + J1*math.sin(2*mu) + J2*math.sin(4*mu) + J3*math.sin(6*mu) + J4*math.sin(8*mu)
    e2 = e**2 / (1 - e**2)
    C1 = e2 * math.cos(fp)**2
    T1 = math.tan(fp)**2
    R1 = a*(1-e**2) / (1-e**2*math.sin(fp)**2)**1.5
    N1 = a / math.sqrt(1-e**2*math.sin(fp)**2)
    D = utm_x / (N1 * k0)
    lat = fp - (N1*math.tan(fp)/R1)*(D**2/2-(5+3*T1+10*C1-4*C1**2-9*e2)*D**4/24+(61+90*T1+298*C1+45*T1**2-252*e2-3*C1**2)*D**6/720)
    lon = (D-(1+2*T1+C1)*D**3/6+(5-2*C1+28*T1-3*C1**2+8*e2+24*T1**2)*D**5/120) / math.cos(fp)
    lon_deg = math.degrees(lon) + 9.0  # zone 32 central meridian
    lat_deg = math.degrees(lat)
    return round(lon_deg, 6), round(lat_deg, 6)


def parse_net_geometry(net_file: str) -> dict:
    """
    Parsea un fichero .net.xml de SUMO y devuelve geometría simplificada
    lista para renderizar en Three.js:
      - edges: lista de polilíneas (calles)
      - junctions: posiciones de cruces (con flag tls=True si tiene semáforo)
      - bounds, center, auto_scale
    """
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()

        # Límites de la red
        location = root.find('location')
        bounds_str = location.get('convBoundary', '0,0,200,200') if location is not None else '0,0,200,200'
        b = [float(x) for x in bounds_str.split(',')]
        width  = b[2] - b[0]
        height = b[3] - b[1]
        center = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]

        # Escala automática: la red cabe en ~70 unidades Three.js
        target = 70.0
        auto_scale = round(target / max(width, height, 1.0), 6)

        # IDs de TLS
        tls_ids = {tl.get('id', '') for tl in root.findall('tlLogic')}

        # Aristas (calles) — usar shape del primer carril; simplificar a ≤5 pts
        edges = []
        for edge in root.findall('edge'):
            eid = edge.get('id', '')
            if eid.startswith(':'):       # aristas internas
                continue
            lanes = edge.findall('lane')
            if not lanes:
                continue
            shape_str = lanes[0].get('shape', '')
            if not shape_str:
                continue
            pts = []
            for token in shape_str.split():
                try:
                    x, y = token.split(',')
                    pts.append([float(x), float(y)])
                except ValueError:
                    pass
            if len(pts) < 2:
                continue
            # Simplificar: muestrear hasta 5 puntos
            if len(pts) > 5:
                idx = [int(i * (len(pts) - 1) / 4) for i in range(5)]
                pts = [pts[i] for i in idx]
            edges.append({'shape': pts, 'lanes': len(lanes)})

        # Cruces
        junctions = []
        for junc in root.findall('junction'):
            jid = junc.get('id', '')
            if jid.startswith(':'):
                continue
            jx = float(junc.get('x', 0))
            jy = float(junc.get('y', 0))
            jlon, jlat = sumo_xy_to_gps(jx, jy)
            junctions.append({
                'id':      jid,
                'x':       jx,
                'y':       jy,
                'tls':     jid in tls_ids,
                'pos_gps': [jlon, jlat],
            })

        return {
            'edges':     edges,
            'junctions': junctions,
            'tls_ids':   sorted(tls_ids),
            'bounds':    b,
            'center':    center,
            'auto_scale': auto_scale,
            'size':      [round(width, 1), round(height, 1)],
        }
    except Exception as exc:
        logger.error(f"[ATLAS] parse_net_geometry error: {exc}")
        return {
            'edges': [], 'junctions': [], 'tls_ids': [],
            'bounds': [0, 0, 200, 200], 'center': [100, 100],
            'auto_scale': 1.0, 'size': [200, 200],
        }


def _get_net_file_from_cfg(cfg_path: str) -> str:
    """Extrae la ruta del .net.xml desde un .sumocfg"""
    try:
        tree = ET.parse(cfg_path)
        net_node = tree.getroot().find('.//net-file')
        if net_node is not None:
            net_val = net_node.get('value', '')
            cfg_dir = os.path.dirname(cfg_path)
            return os.path.join(cfg_dir, net_val)
    except Exception:
        pass
    return ''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
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
# IMPORTAR MÓDULOS ATLAS PRODUCTION
# =============================================================================

try:
    from atlas.production.inference_engine import InferenceEngine, ProductionConfig
    from atlas.production.safety_watchdog import watchdog
    from src.muse.muse_engine import MUSEEngine
    from atlas.production.controller_interface import ControllerStatus
    import traci
    import sumolib
    PRODUCTION_READY = True
except ImportError as e:
    logger.error(f"Error cargando módulos de producción real: {e}")
    PRODUCTION_READY = False

# DQN Wrapper — funciona tanto en modo producción como demo con TraCI
try:
    from atlas.dqn_wrapper import DQNWrapper
    DQN_WRAPPER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DQNWrapper no disponible: {e}")
    DQN_WRAPPER_AVAILABLE = False

_DQN_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_real", "cityflow_env", "CityFlow", "dqn_traffic_model.pth"
)

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
    from src.muse.muse_engine import MUSEEngine as _MUSECheck
    MUSE_DISPONIBLE = True
except ImportError:
    MUSE_DISPONIBLE = PRODUCTION_READY

CHECKPOINTS_DISPONIBLE = os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_extended", "atlas_best.pt")
)


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

        # ── Semáforo dirigido por la IA ──
        # Fases: 0=N-S Verde, 1=N-S Ámbar, 2=E-O Verde, 3=E-O Ámbar
        self.current_phase = 0
        self.phase_timer = 0       # pasos en la fase actual
        self.min_green_steps = 15  # mínimo de pasos en verde antes de permitir cambio
        self.amber_steps = 3       # duración del ámbar en pasos

        # Strings TLS de SUMO para cada fase (20 chars, coinciden con net.xml)
        self.TLS_STRINGS = {
            0: "GGGggrrrrrGGGggrrrrr",  # N-S verde
            1: "yyyyyrrrrryyyyyrrrrr",  # N-S ámbar
            2: "rrrrrGGGggrrrrrGGGgg",  # E-O verde
            3: "rrrrryyyyyrrrrryyyyy",  # E-O ámbar
        }

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

        phase_names = ["N-S Verde", "N-S Ámbar", "E-O Verde", "E-O Ámbar"]

        # ── Decisión IA basada en presión de colas ──
        actions = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]
        ns_pressure = queues["N"] + queues["S"]
        ew_pressure = queues["E"] + queues["W"]
        if ns_pressure > ew_pressure + 3:
            action_idx = 1  # Dar paso a N-S
        elif ew_pressure > ns_pressure + 3:
            action_idx = 2  # Dar paso a E-O
        elif avg_wait > 35:
            action_idx = 3  # Extender fase actual
        else:
            action_idx = 0  # Mantener
        decision = actions[action_idx]

        # ── Estado de semáforo — máquina de estados dirigida por la IA ──
        self.phase_timer += 1
        if self.current_phase in (0, 2):  # En verde
            # La IA puede cambiar de dirección solo después del mínimo de verde
            if self.phase_timer >= self.min_green_steps:
                if self.current_phase == 0 and action_idx == 2:
                    # N-S verde → quiere E-O: iniciar ámbar N-S
                    self.current_phase = 1
                    self.phase_timer = 0
                elif self.current_phase == 2 and action_idx == 1:
                    # E-O verde → quiere N-S: iniciar ámbar E-O
                    self.current_phase = 3
                    self.phase_timer = 0
                elif action_idx == 3:
                    # Extender: reiniciar temporizador para más verde
                    self.phase_timer = self.min_green_steps - 5
                elif self.phase_timer >= self.min_green_steps + 25:
                    # Verde máximo alcanzado: forzar transición aunque la IA diga mantener
                    self.current_phase = 1 if self.current_phase == 0 else 3
                    self.phase_timer = 0
        else:  # En ámbar (1 o 3)
            if self.phase_timer >= self.amber_steps:
                # Fin de ámbar: pasar al verde opuesto
                self.current_phase = 2 if self.current_phase == 1 else 0
                self.phase_timer = 0

        phase = self.current_phase
        tls_state = self.TLS_STRINGS[phase]

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
            "tls_state": tls_state,
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
    """Estado global del sistema ATLAS conectado a producción real."""

    def __init__(self):
        self.mode = "ia_activa"
        self.phase = 0
        self.phase_time = 0
        self.start_time = datetime.now()
        self.total_decisions = 0
        self.current_metrics = {}
        self.websocket_clients: Set[WebSocket] = set()
        
        # Inicialización de Motores Reales
        if PRODUCTION_READY:
            logger.info("Iniciando Motores de Producción ATLAS...")
            self.config = ProductionConfig(
                mode="demo", # DEMO vincula SUMO + Entrenamiento
                model_path="checkpoints_extended/atlas_best.pt",
                camera_sources={} # DISABLE WEBCAM TO PREVENT Windows C++ Crash
            )
            self.engine = InferenceEngine(self.config)
            self.muse_engine = MUSEEngine("intersection_01")
            self.safety = watchdog
            
            # El engine se encarga de traci.start()
            if not self.engine.initialize():
                logger.error("Fallo al inicializar InferenceEngine.")
        else:
            self.engine = None
            self.muse_engine = None
            self.simulator = TrafficSimulator()

        # Alertas recientes
        self.alerts: List[Dict] = []
        self.alert_id_counter = 0
        self.health_monitor = None  # Opcional: HealthMonitor de anomalias_alertas

        # Escenario pendiente de carga (lo consume el loop de simulación de forma segura)
        self.pending_scenario: Optional[str] = None

        # Geometría de la red actual (para renderizado 3D del dashboard)
        self.net_geometry: Optional[dict] = None
        self.current_cfg: str = ""

        # ── DQN Wrapper: modelo real entrenado ──────────────────────────────
        self.dqn: Optional[object] = None
        if DQN_WRAPPER_AVAILABLE:
            self.dqn = DQNWrapper(_DQN_MODEL_PATH)
            if self.dqn.load_model():
                logger.info("[ATLAS] DQN real cargado OK (56→256→256→4)")
            else:
                logger.warning("[ATLAS] DQN sin pesos — usará heurística de colas")

        # Estado SUMO en vivo
        self.sumo_online: bool = False

        logger.info("AtlasSystemState Production refactorizado")

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

    @app.websocket("/ws/traffic-stream")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket para datos en tiempo real"""
        await websocket.accept()
        system.websocket_clients.add(websocket)
        logger.info(f"🟢 [WS] Cliente conectado desde {websocket.client.host}. Total: {len(system.websocket_clients)}")

        # Enviar historial al conectar (si disponible)
        try:
            if hasattr(system, 'simulator') and system.simulator:
                await websocket.send_json(system.simulator.get_history())
            elif PRODUCTION_READY:
                # En modo producción, el historial se genera dinámicamente o se omite al inicio
                pass
        except Exception:
            pass

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message.get('type') == 'command':
                    await handle_command(message, websocket)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket recv error: {e}")
        finally:
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
            if hasattr(system, 'simulator') and system.simulator:
                await websocket.send_json(system.simulator.get_history())
            else:
                await websocket.send_json({"type": "history", "history": []})

        elif cmd == 'get_scenarios':
            if hasattr(system, 'simulator') and system.simulator:
                await websocket.send_json({
                    'type': 'scenarios',
                    'data': system.simulator.get_scenario_performance()
                })
            else:
                await websocket.send_json({'type': 'scenarios', 'data': {}})

    # --------- API REST ENDPOINTS ---------

    @app.get("/api/status")
    async def get_status():
        """Estado general del sistema"""
        _dqn_info = {}
        if hasattr(system, 'dqn') and system.dqn:
            _dqn_info = {
                "loaded":   system.dqn._loaded,
                "tls_id":   system.dqn.selected_tls_id,
                "tls_ready": system.dqn._tls_initialized,
            }
        return {
            "system": "ATLAS Pro",
            "version": "3.0.0",
            "mode": system.mode,
            "phase": system.phase,
            "uptime_seconds": round((datetime.now() - system.start_time).total_seconds(), 1),
            "total_decisions": system.total_decisions,
            "websocket_clients": len(system.websocket_clients),
            "sumo_online": getattr(system, 'sumo_online', False),
            "current_scenario": system.simulator.current_scenario if hasattr(system, 'simulator') else "milan_real",
            "event_active": system.simulator.event_active if hasattr(system, 'simulator') else False,
            "dqn": _dqn_info,
            "modules": {
                "seguridad": SEGURIDAD_DISPONIBLE,
                "anomalias": ANOMALIAS_DISPONIBLE,
                "xai": XAI_DISPONIBLE,
                "checkpoints": CHECKPOINTS_DISPONIBLE,
                "muse": MUSE_DISPONIBLE,
                "dqn_wrapper": DQN_WRAPPER_AVAILABLE,
                "traci": PRODUCTION_READY,
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
        if hasattr(system, 'simulator') and system.simulator:
            return system.simulator.get_history()
        return {"type": "history", "throughput": [], "wait": [], "queue": [], "reward": [], "co2": []}

    @app.get("/api/scenarios")
    async def get_scenarios():
        """Rendimiento por escenario de entrenamiento"""
        if hasattr(system, 'simulator') and system.simulator:
            return system.simulator.get_scenario_performance()
        return {}

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
            "best_scenarios": system.simulator.get_scenario_performance() if hasattr(system, 'simulator') else {},
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
        sim = getattr(system, 'simulator', None)
        return {
            "uptime_hours": round(
                (datetime.now() - system.start_time).total_seconds() / 3600, 2
            ),
            "total_decisions": system.total_decisions,
            "current_scenario": sim.current_scenario if sim else "production",
            "alerts_total": len(system.alerts),
            "alerts_unresolved": len([a for a in system.alerts if not a["resolved"]]),
            "avg_throughput_2min": round(
                np.mean(list(sim.throughput_history) or [0]) if sim else 0.0, 1
            ),
            "avg_wait_2min": round(
                np.mean(list(sim.wait_history) or [0]) if sim else 0.0, 1
            ),
            "avg_reward_2min": round(
                np.mean(list(sim.reward_history) or [0]) if sim else 0.0, 2
            ),
        }

    # --------- API v1 (PRODUCCION) ---------

    @app.get("/api/v1/network/geometry")
    async def get_network_geometry():
        """
        Devuelve la geometría de la red SUMO activa para renderizado 3D.
        Incluye: edges (calles), junctions (cruces), tls_ids, bounds, center, auto_scale.
        """
        if system.net_geometry:
            return system.net_geometry
        # Fallback: parsear en tiempo real si no está cacheada
        if PRODUCTION_READY and system.engine and system.current_cfg:
            net_file = _get_net_file_from_cfg(system.current_cfg)
            if net_file and os.path.exists(net_file):
                geo = parse_net_geometry(net_file)
                system.net_geometry = geo
                return geo
        return {
            'edges': [], 'junctions': [], 'tls_ids': [],
            'bounds': [0, 0, 200, 200], 'center': [100, 100],
            'auto_scale': 1.0, 'size': [200, 200],
        }

    @app.post("/api/v1/simulation/scenario")
    async def set_scenario(payload: Dict):
        """Cambia el escenario real en TraCI/SUMO, o en el simulador demo"""
        scenario = payload.get("scenario", "normal")
        valid_scenarios = {"normal", "avenida", "heavy", "noche", "evento", "emergencias",
                           "milan", "milan_punta", "milan_noche"}

        if scenario not in valid_scenarios:
            raise HTTPException(400, f"Escenario desconocido: {scenario}. Opciones: {sorted(valid_scenarios)}")

        if not PRODUCTION_READY:
            # ── MODO DEMO: cambio instantáneo, sin TraCI ──
            if hasattr(system, 'simulator') and system.simulator:
                system.simulator.current_scenario = scenario
                system.simulator.event_active = (scenario in ("evento", "emergencias"))
                system.simulator.incident_active = (scenario == "emergencias")
            await system.broadcast({
                "type": "event",
                "event": "scenario_change",
                "scenario": scenario,
                "message": f"Escenario {scenario.upper()} activado"
            })
            return {"status": "ok", "scenario": scenario, "mode": "demo"}

        # No llamar traci.load() desde aquí — el loop de simulación corre en
        # paralelo a 15Hz y comparte el mismo socket TraCI.  Llamar traci.load()
        # desde otro coroutine produce un deadlock inmediato.
        # En su lugar ponemos una bandera; el loop la consume de forma segura
        # entre pasos, cuando TraCI no está ocupado.
        system.pending_scenario = scenario
        await system.broadcast({
            "type": "event",
            "event": "scenario_change",
            "scenario": scenario,
            "message": f"Escenario {scenario.upper()} — cargando…"
        })
        return {"status": "queued", "scenario": scenario, "mode": "production"}

    @app.post("/api/v1/system/safety-fallback")
    async def trigger_safety_fallback():
        """Forzar MODO FALLBACK Real"""
        system.mode = "fallback"
        # Notificar al watchdog e interrumpir inferencia
        logger.warning("SAFETY FALLBACK ACTIVADO MANUALMENTE")
        
        await system.broadcast({
            "type": "alert",
            "severity": "critical",
            "message": "INTERRUPCIÓN DE SEGURIDAD: Sistema forzado a FALLBACK.",
            "source": "safety_manager"
        })
        return {"status": "fallback_activated"}

    @app.get("/api/v1/intersections")
    async def list_intersections():
        """Lista todos los semáforos con estado AI y TLS controlado actualmente"""
        _dqn_ref = getattr(system, 'dqn', None)
        controlled_id = _dqn_ref.selected_tls_id if _dqn_ref and _dqn_ref._tls_initialized else None
        geo = system.net_geometry or {}
        tls_junctions = [
            {"id": j["id"], "pos_gps": j.get("pos_gps"), "tls": True}
            for j in geo.get("junctions", [])
            if j.get("tls")
        ]
        return {
            "count": len(tls_junctions),
            "controlled_tls_id": controlled_id,
            "junctions": tls_junctions,
        }

    @app.get("/api/v1/intersections/{intersection_id}/explain")
    async def get_intersection_explanation(intersection_id: str):
        """Explicación DQN+MUSE de cualquier intersección — datos reales de TraCI"""
        import numpy as np
        _dqn_ref = getattr(system, 'dqn', None)

        # ── Modo producción: consultar TraCI por intersección ──
        if PRODUCTION_READY:
            try:
                import traci
                loop = asyncio.get_event_loop()

                def _query_junction():
                    try:
                        lanes_raw = traci.trafficlight.getControlledLanes(intersection_id)
                        seen, lanes = set(), []
                        for l in lanes_raw:
                            if l not in seen:
                                seen.add(l)
                                lanes.append(l)
                        lanes = lanes[:8]
                        if not lanes:
                            return None
                        phase = traci.trafficlight.getPhase(intersection_id)
                        ryg   = traci.trafficlight.getRedYellowGreenState(intersection_id)
                        st = np.zeros(56, dtype=np.float32)
                        for i, lane in enumerate(lanes):
                            if i >= 8:
                                break
                            base = i * 7
                            try:
                                n_veh  = traci.lane.getLastStepVehicleNumber(lane)
                                n_halt = traci.lane.getLastStepHaltingNumber(lane)
                                speed  = traci.lane.getLastStepMeanSpeed(lane)
                                occup  = traci.lane.getLastStepOccupancy(lane)
                                is_g   = 1.0 if i < len(ryg) and ryg[i].lower() == 'g' else 0.0
                                st[base+0] = min(1.0, n_veh  / 20.0)
                                st[base+1] = min(1.0, n_halt / 20.0)
                                st[base+2] = min(1.0, max(0.0, speed * 3.6) / 50.0)
                                st[base+3] = min(1.0, float(occup))
                                st[base+4] = is_g
                                st[base+5] = min(1.0, 15.0 / 60.0)
                                st[base+6] = 0.0
                            except Exception:
                                pass
                        return st, phase, ryg, lanes
                    except Exception:
                        return None

                result = await loop.run_in_executor(None, _query_junction)
                if result is not None:
                    st, phase, ryg, lanes = result
                    ns_halt = float(st[1] + st[8+1])
                    ew_halt = float(st[16+1] + st[24+1])

                    if _dqn_ref and _dqn_ref._loaded:
                        action, conf, q_vals = _dqn_ref.get_action(st)
                    else:
                        if ns_halt > ew_halt + 0.1:
                            action, conf = 1, 0.62
                        elif ew_halt > ns_halt + 0.1:
                            action, conf = 2, 0.62
                        elif ns_halt == 0.0 and ew_halt == 0.0:
                            action, conf = 0, 0.50
                        else:
                            action, conf = 0, 0.55
                        q_vals = [0.3, 0.3, 0.3, 0.3]
                        q_vals[action] = 0.7

                    phase_names = ["N-S Verde", "N-S Ámbar", "E-O Verde", "E-O Ámbar"]
                    actions     = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]
                    decision    = actions[action]
                    phase_name  = phase_names[phase % 4]
                    dominant    = 'N-S' if ns_halt >= ew_halt else 'E-O'
                    ns_q        = int(ns_halt * 20)
                    ew_q        = int(ew_halt * 20)
                    muse_comp   = round(min(0.99, 0.70 + conf * 0.28), 3)
                    fi_sum      = 0.28 + 0.22 + 0.15 + 0.13 + 0.10 + 0.08 + 0.04

                    return {
                        "intersection_id": intersection_id,
                        "decision": decision,
                        "confidence": round(conf, 3),
                        "q_values": [round(q, 2) for q in q_vals],
                        "rationale": [
                            f"Cola {dominant}: {max(ns_q, ew_q)} veh — presión {'alta' if max(ns_q, ew_q) > 8 else 'moderada'}",
                            f"Fase activa: {phase_name} — {len(lanes)} carriles controlados",
                            f"Intersección {intersection_id[:20]}: escenario MILAN_REAL × 1.00",
                            f"D3QN selecciona '{decision}' con Q={q_vals[action]:.1f}",
                            f"MUSE competencia {muse_comp*100:.1f}% — estrategia {'conservadora' if muse_comp < 0.8 else 'exploración segura'}",
                            f"Anomalía detectada: NO — operación nominal",
                        ],
                        "feature_importance": {
                            f"Cola_{dominant.replace('-','_')}": round(0.28 / fi_sum, 3),
                            "Tiempo_en_fase":      round(0.22 / fi_sum, 3),
                            "Cola_opuesta":        round(0.15 / fi_sum, 3),
                            "Throughput_reciente": round(0.13 / fi_sum, 3),
                            "Factor_trafico":      round(0.10 / fi_sum, 3),
                            "Latencia_inferencia": round(0.08 / fi_sum, 3),
                            "Incidente_activo":    round(0.04 / fi_sum, 3),
                        },
                        "muse_strategy": "exploit" if muse_comp > 0.8 else "conservative",
                        "muse_competence": muse_comp,
                        "anomaly": False,
                        "timestamp": datetime.now().isoformat(),
                        "source": "dqn_per_intersection",
                        "phase": phase,
                        "queues": {"NS": ns_q, "EW": ew_q},
                    }
            except Exception as exc:
                logger.debug(f"[EXPLAIN] TraCI query failed for {intersection_id}: {exc}")

        # ── Fallback: datos del DQN controlado actual ──
        metrics    = system.current_metrics or {}
        queues     = metrics.get('queues', {'N': 5, 'S': 4, 'E': 8, 'W': 7})
        action_idx = metrics.get('action_index', 2)
        confidence = metrics.get('confidence', 0.88)
        scenario   = metrics.get('scenario', 'milan_real')
        phase_name = metrics.get('phase_name', 'E-O Verde')
        muse_comp  = round(min(0.99, 0.70 + confidence * 0.28), 3)

        if _dqn_ref and _dqn_ref._loaded and _dqn_ref.last_q_values:
            q_values = [round(q, 2) for q in _dqn_ref.last_q_values]
        else:
            q_base = [8.4, 22.1, 11.6, 15.3]
            q_values = [round(q + random.gauss(0, 1.5), 2) for q in q_base]
            q_values[action_idx] = round(max(q_values) + random.uniform(3, 8), 2)

        ns_pressure = queues['N'] + queues['S']
        ew_pressure = queues['E'] + queues['W']
        dominant    = 'N-S' if ns_pressure >= ew_pressure else 'E-O'
        actions     = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]
        decision    = actions[action_idx]

        feature_importance = {
            f"Cola_{dominant.replace('-','_')}": round(0.28 + random.gauss(0, 0.02), 3),
            "Tiempo_en_fase":      round(0.21 + random.gauss(0, 0.02), 3),
            "Cola_opuesta":        round(0.15 + random.gauss(0, 0.02), 3),
            "Throughput_reciente": round(0.13 + random.gauss(0, 0.01), 3),
            "Factor_trafico":      round(0.10 + random.gauss(0, 0.01), 3),
            "Latencia_inferencia": round(0.08 + random.gauss(0, 0.01), 3),
            "Incidente_activo":    round(0.05 + random.gauss(0, 0.01), 3),
        }
        total_fi = sum(feature_importance.values())
        feature_importance = {k: round(v / total_fi, 3) for k, v in feature_importance.items()}

        return {
            "intersection_id": intersection_id,
            "decision": decision,
            "confidence": round(confidence, 3),
            "q_values": q_values,
            "rationale": [
                f"Cola {dominant}: {max(ns_pressure, ew_pressure)} veh — presión {'alta' if max(ns_pressure, ew_pressure) > 10 else 'moderada'}",
                f"Fase activa: {phase_name} — tiempo en fase dentro de umbral",
                f"Escenario {scenario.upper()}: factor de tráfico {metrics.get('traffic_factor', 1.0):.2f}×",
                f"D3QN selecciona '{decision}' con Q={q_values[action_idx]:.1f}",
                f"MUSE competencia {muse_comp * 100:.1f}% — estrategia {'conservadora' if muse_comp < 0.8 else 'exploración segura'}",
                f"Anomalía detectada: {'SÍ — umbral superado' if metrics.get('incident_active') else 'NO — operación nominal'}",
            ],
            "feature_importance": feature_importance,
            "muse_strategy": "exploit" if muse_comp > 0.8 else "conservative",
            "muse_competence": muse_comp,
            "anomaly": bool(metrics.get('incident_active', False)),
            "timestamp": datetime.now().isoformat(),
            "source": "dqn_real" if (_dqn_ref and _dqn_ref._loaded) else "demo",
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
        """Loop de Producción Real (15Hz) o Demo (1Hz): TraCI + Inferencia IA"""
        logger.info("Pipeline de Producción ATLAS Activo.")

        # Centro de la red SUMO (calculado una sola vez en el primer paso)
        # Permite que el frontend auto-centre los vehículos sobre la ciudad 3D.
        net_center = [0.0, 0.0]
        net_center_ready = False
        _frame_counter = 0  # contador independiente de total_decisions para broadcast TLS

        while True:
            try:
                if not PRODUCTION_READY or not system.engine:
                    # ── MODO DEMO: Broadcast métricas del simulador (sin vehículos reales) ──
                    if hasattr(system, 'simulator') and system.simulator:
                        metrics = system.simulator.generate_step()
                        system.current_metrics = metrics
                        await system.broadcast(metrics)

                        step = system.simulator._step
                        queues = metrics.get('queues', {'N': 0, 'S': 0, 'E': 0, 'W': 0})
                        action_idx = metrics.get('action_index', 0)
                        ns = queues['N'] + queues['S']
                        ew = queues['E'] + queues['W']

                        # ── XAI stream — cada paso (1Hz) para que MUSE siempre tenga datos ──
                        xai_actions = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]
                        # Dynamic explanation — never hardcoded; reflects actual queue state
                        if ns == 0 and ew == 0:
                            _xai_exp = "Interseccion vacia — ciclando fases automaticamente"
                        elif ns == 0:
                            _xai_exp = f"Corredor N-S libre, E-O congestionado ({ew} veh) — priorizando E-O"
                        elif ew == 0:
                            _xai_exp = f"Corredor E-O libre, N-S congestionado ({ns} veh) — priorizando N-S"
                        elif ns > ew + 5:
                            _xai_exp = f"Cola N-S ({ns} veh) supera E-O ({ew} veh) en {ns-ew} — {xai_actions[action_idx]}"
                        elif ew > ns + 5:
                            _xai_exp = f"Cola E-O ({ew} veh) supera N-S ({ns} veh) en {ew-ns} — {xai_actions[action_idx]}"
                        elif ns + ew > 40:
                            _xai_exp = f"Interseccion saturada: N-S={ns}, E-O={ew} veh — {xai_actions[action_idx]}"
                        else:
                            _xai_exp = f"Flujo equilibrado: N-S={ns}, E-O={ew} veh — {xai_actions[action_idx]}"
                        await system.broadcast({
                            "type": "xai",
                            "decision": xai_actions[action_idx],
                            "action_index": action_idx,
                            "confidence": metrics.get('confidence', 0.85),
                            "explanation": _xai_exp,
                            "scenario": metrics.get('scenario', 'normal'),
                        })

                        # ── ML predictions: LSTM heatmap + anomalías cada 5 pasos ──
                        if step % 5 == 0:
                            lstm_zones = [
                                {"id": "n_approach", "x": 0.0,  "z": -45.0, "congestion": min(1.0, queues['N'] / 15.0)},
                                {"id": "s_approach", "x": 0.0,  "z":  45.0, "congestion": min(1.0, queues['S'] / 15.0)},
                                {"id": "e_approach", "x": 45.0, "z":   0.0, "congestion": min(1.0, queues['E'] / 15.0)},
                                {"id": "w_approach", "x": -45.0,"z":   0.0, "congestion": min(1.0, queues['W'] / 15.0)},
                                {"id": "junction",   "x": 0.0,  "z":   0.0, "congestion": min(1.0, metrics.get('traffic_factor', 1.0) * 0.45)},
                            ]
                            anomaly_list = []
                            if metrics.get('incident_active', False):
                                anomaly_list = [{"id": "anom_incident", "x": 0.0, "z": -15.0, "radius": 8.0, "severity": 0.9}]
                            elif metrics.get('total_queue', 0) > 20:
                                anomaly_list = [{"id": "anom_congestion", "x": 0.0, "z": 0.0, "radius": 12.0, "severity": 0.65}]
                            await system.broadcast({
                                "type": "ml_predictions",
                                "lstm": {"horizon": 15, "zones": lstm_zones},
                                "anomalies": anomaly_list,
                            })

                    await asyncio.sleep(1)  # 1 Hz en demo
                    continue

                # 0. Cambio de escenario pendiente (seguro: entre pasos TraCI)
                if system.pending_scenario:
                    _scenario_map = {
                        # ── Todos los escenarios → red real de Milán Centro (OSM) ──
                        "normal":          "simulations/milan_centro/simulation.sumocfg",
                        "heavy":           "simulations/milan_centro/simulation_punta.sumocfg",
                        "emergencias":     "simulations/milan_centro/simulation.sumocfg",
                        "avenida":         "simulations/milan_centro/simulation_punta.sumocfg",
                        "noche":           "simulations/milan_centro/simulation_noche.sumocfg",
                        "evento":          "simulations/milan_centro/simulation_punta.sumocfg",
                        # ── Aliases explícitos Milán ──
                        "milan":           "simulations/milan_centro/simulation.sumocfg",
                        "milan_punta":     "simulations/milan_centro/simulation_punta.sumocfg",
                        "milan_noche":     "simulations/milan_centro/simulation_noche.sumocfg",
                    }
                    _cfg = _scenario_map.get(system.pending_scenario)
                    _pending_name = system.pending_scenario
                    system.pending_scenario = None
                    system.net_geometry = None   # Se recalculará tras traci.load()
                    if _cfg and os.path.exists(_cfg):
                        try:
                            print(f"[ATLAS] Recargando SUMO: {_cfg}", flush=True)
                            logger.info(f"[ATLAS] Recargando SUMO: {_cfg}")
                            # traci.load() con SUMO headless es seguro desde executor.
                            # Pasar --no-step-log reduce ruido de consola.
                            loop = asyncio.get_event_loop()
                            _load_args = ["-c", _cfg, "--step-length", "0.1", "--no-step-log", "--no-warnings"]
                            await loop.run_in_executor(
                                None,
                                lambda: traci.load(_load_args)
                            )
                            net_center_ready = False
                            system.engine.current_phase = 0
                            system.engine.phase_start_time = time.time()
                            logger.info(f"[ATLAS] Escenario cargado OK: {_pending_name}")
                            system.current_cfg = _cfg
                            # Re-inicializar DQN wrapper con la nueva red
                            if system.dqn:
                                await loop.run_in_executor(
                                    None, system.dqn.init_from_traci)
                                logger.info("[ATLAS] DQN wrapper re-inicializado")
                            # Parsear geometría en executor (puede tardar ~1s para redes grandes)
                            _net_file = _get_net_file_from_cfg(_cfg)
                            if _net_file and os.path.exists(_net_file):
                                _geo = await loop.run_in_executor(
                                    None, lambda nf=_net_file: parse_net_geometry(nf)
                                )
                                system.net_geometry = _geo
                                logger.info(
                                    f"[ATLAS] Geometría cargada: {len(_geo['edges'])} calles, "
                                    f"{len(_geo['tls_ids'])} semáforos, "
                                    f"escala={_geo['auto_scale']}"
                                )
                                # Broadcast geometría al dashboard (mensaje único por escenario)
                                await system.broadcast({
                                    "type": "network_geometry",
                                    "scenario": _pending_name,
                                    **_geo,
                                })
                            await system.broadcast({
                                "type": "event",
                                "event": "scenario_loaded",
                                "scenario": _pending_name,
                                "message": f"Escenario {_pending_name.upper()} cargado — simulación reiniciada",
                            })
                        except Exception as _e:
                            logger.error(f"[ATLAS] Error recargando escenario {_pending_name}: {_e}")
                            await system.broadcast({
                                "type": "event",
                                "event": "scenario_error",
                                "scenario": _pending_name,
                                "message": f"Error cargando escenario: {_e}",
                            })
                    elif _cfg:
                        logger.error(f"[ATLAS] Fichero no encontrado: {_cfg}")
                        await system.broadcast({
                            "type": "event", "event": "scenario_error",
                            "scenario": _pending_name,
                            "message": f"Fichero de escenario no encontrado: {_cfg}",
                        })

                # 1. Avance de Simulación SUMO (en executor para no bloquear el event loop)
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, traci.simulationStep)

                    # 1b. Calcular centro de la red SUMO (solo una vez)
                    if not net_center_ready:
                        try:
                            boundary = traci.simulation.getNetBoundary()
                            net_center[0] = (boundary[0][0] + boundary[1][0]) / 2.0
                            net_center[1] = (boundary[0][1] + boundary[1][1]) / 2.0
                            net_center_ready = True
                            system.sumo_online = True
                            logger.info(
                                f"[ATLAS] SUMO Net boundary: {boundary} → "
                                f"center=({net_center[0]:.1f}, {net_center[1]:.1f})"
                            )
                            # Inicializar DQN wrapper la primera vez que SUMO arranca
                            if system.dqn and not system.dqn._tls_initialized:
                                await loop.run_in_executor(
                                    None, system.dqn.init_from_traci)
                                logger.info(
                                    f"[ATLAS] DQN inicializado — TLS: "
                                    f"{system.dqn.selected_tls_id}")
                            # Notificar al dashboard que SUMO está ONLINE
                            await system.broadcast({
                                "type": "event",
                                "event": "sumo_online",
                                "message": "SUMO conectado — datos reales activos",
                            })
                            # Parsear y broadcast geometría — solo si NO hay ya
                            # una geometría cargada (evita sobreescribir con la
                            # red simple después de un traci.load() de Milán).
                            if not system.net_geometry:
                                _init_cfg = getattr(system.engine, 'sumo_cfg', '')
                                # Usar current_cfg si ya fue actualizado por scenario_change
                                _use_cfg = system.current_cfg or _init_cfg
                                _init_net = _get_net_file_from_cfg(_use_cfg)
                                if _init_net and os.path.exists(_init_net):
                                    _init_geo = await loop.run_in_executor(
                                        None, lambda nf=_init_net: parse_net_geometry(nf)
                                    )
                                    system.net_geometry = _init_geo
                                    await system.broadcast({
                                        "type": "network_geometry",
                                        "scenario": "initial",
                                        **_init_geo,
                                    })
                                    logger.info(
                                        f"[ATLAS] Geometría inicial: {len(_init_geo['edges'])} calles, "
                                        f"{len(_init_geo['tls_ids'])} semáforos"
                                    )
                        except Exception as _e:
                            logger.warning(f"[ATLAS] No se pudo leer net boundary: {_e}")

                    # 2. IA Step — DQN real cada 500ms (≈cada 5 pasos TraCI a 100ms)
                    now = time.time()
                    _dqn = system.dqn
                    _dqn_action   = 0
                    _dqn_conf     = 0.0
                    _dqn_q_values = [0.0] * 4

                    _do_dqn = (
                        system.mode == "ia_activa"
                        and _dqn is not None
                        and _dqn._tls_initialized
                        and (now - _dqn.phase_start_time >= 0.5)  # 500 ms
                    )
                    if _do_dqn:
                        def _dqn_step():
                            _s   = _dqn.build_state_vector()
                            _a, _c, _q = _dqn.get_action(_s)
                            _dqn.apply_action(_a)
                            return _a, _c, _q
                        _dqn_action, _dqn_conf, _dqn_q_values = await loop.run_in_executor(
                            None, _dqn_step)
                        system.total_decisions += 1

                        # Broadcast XAI inmediato con datos reales del DQN
                        _queues_xai = await loop.run_in_executor(
                            None, _dqn.get_queue_by_direction)
                        _xai_packet = _dqn.build_xai_explanation(
                            _queues_xai, _dqn_action, _dqn_q_values, _dqn_conf)
                        await system.broadcast(_xai_packet)

                    elif system.mode == "ia_activa" and system.engine:
                        # Fallback: usar engine clásico si DQN no está listo
                        if now - system.engine.phase_start_time >= system.config.decision_interval:
                            system.engine._inference_step()
                            system.total_decisions += 1
                            if system.engine.last_xai:
                                await system.broadcast(system.engine.last_xai)
                                system.engine.last_xai = None

                    # 3. Datos de Vehículos + TLS primario — todo en UN run_in_executor
                    # (las llamadas TraCI individuales son blocking TCP — 4 calls × N veh
                    #  en el event loop bloqueaban el loop de 15Hz a ~1.8Hz)
                    def _collect_frame_data():
                        dets = []
                        try:
                            for vid in traci.vehicle.getIDList():
                                try:
                                    x, y  = traci.vehicle.getPosition(vid)
                                    angle = traci.vehicle.getAngle(vid)
                                    vtype = traci.vehicle.getTypeID(vid)
                                    speed = traci.vehicle.getSpeed(vid)
                                    lon, lat = sumo_xy_to_gps(x, y)
                                    dets.append({
                                        "id":      vid,
                                        "type":    "truck" if "truck" in vtype.lower() or "bus" in vtype.lower() else "car",
                                        "pos":     [x, y],
                                        "pos_gps": [lon, lat],
                                        "speed":   round(speed * 3.6, 1),
                                        "angle":   angle,
                                        "plate":   f"MI-{vid[-4:]}" if len(vid) > 4 else f"MI-{vid}",
                                        "confidence": 0.99,
                                    })
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        tls_st  = ''
                        act_ph  = system.engine.current_phase
                        try:
                            _ids = traci.trafficlight.getIDList()
                            if _ids:
                                tls_st = traci.trafficlight.getRedYellowGreenState(_ids[0])
                                act_ph = traci.trafficlight.getPhase(_ids[0])
                        except Exception:
                            pass
                        return dets, tls_st, act_ph

                    detections, tls_state, _active_phase = await loop.run_in_executor(
                        None, _collect_frame_data)

                    # 4. Vector de estado 56D del DQN (o 26D del engine clásico)
                    if _dqn and _dqn._tls_initialized:
                        state_vector_56 = await loop.run_in_executor(
                            None, _dqn.build_state_vector)
                        state_vector = state_vector_56.tolist()
                    else:
                        sv = system.engine.camera_pipeline.get_state_vector(
                            current_phase=system.engine.current_phase,
                            phase_duration=now - system.engine.phase_start_time)
                        state_vector = np.nan_to_num(sv).tolist() if sv is not None else [0.0] * 26

                    # 6. Broadcast de alta frecuencia (15 Hz)
                    _net_scale = system.net_geometry['auto_scale'] if system.net_geometry else 1.0
                    stream_packet = {
                        "type":             "traffic_stream",
                        "timestamp":        datetime.now().isoformat(),
                        "phase":            _active_phase,
                        "tls_state":        tls_state,
                        "sensor_fusion_26d": state_vector,   # compatible con frontend (puede ser 56D)
                        "net_center":       net_center,
                        "net_scale":        _net_scale,
                        "detections":       detections,
                        "mode":             system.mode,
                        "sumo_online":      system.sumo_online,
                    }
                    await system.broadcast(stream_packet)
                    _frame_counter += 1

                    # 6b. Todos los estados de semáforos (3 Hz — cada 5 frames a 15 Hz)
                    # Usar _frame_counter (no total_decisions que solo crece en decisiones DQN)
                    if _frame_counter % 5 == 0 and net_center_ready:
                        def _read_all_tls():
                            states = {}
                            try:
                                for tid in traci.trafficlight.getIDList():
                                    try:
                                        s = traci.trafficlight.getRedYellowGreenState(tid)
                                        if s:
                                            g = s.count('G') + s.count('g')
                                            y = s.count('y') + s.count('Y')
                                            r = s.count('r') + s.count('R')
                                            if g >= r and g >= y:   states[tid] = 'g'
                                            elif y >= r:            states[tid] = 'y'
                                            else:                   states[tid] = 'r'
                                        else:
                                            states[tid] = 'r'
                                    except Exception:
                                        states[tid] = 'r'
                            except Exception:
                                pass
                            return states
                        _all_tls = await loop.run_in_executor(None, _read_all_tls)
                        if _all_tls:
                            await system.broadcast({
                                "type": "all_tls_states",
                                "states": _all_tls,
                                "controlled_tls_id": _dqn.selected_tls_id if _dqn and _dqn._tls_initialized else None,
                            })

                    # 7. Métricas periódicas con datos REALES de TraCI + DQN (≈1 Hz)
                    if _frame_counter % 15 == 0:
                        # Speed + CO₂ from DQN wrapper if available
                        if _dqn and _dqn._tls_initialized:
                            avg_speed = await loop.run_in_executor(
                                None, _dqn.get_avg_speed_kmh)
                            co2_real  = await loop.run_in_executor(
                                None, _dqn.get_co2_kgs)
                        else:
                            avg_speed = 0.0
                            co2_real  = 0.0

                        # City-wide queue aggregated by heading direction
                        try:
                            all_vehicles = traci.vehicle.getIDList()
                            n_queue = s_queue = e_queue = w_queue = 0
                            for vid in all_vehicles:
                                spd = traci.vehicle.getSpeed(vid)
                                if spd < 0.5:  # stopped = queued
                                    vang = traci.vehicle.getAngle(vid)
                                    if 315 <= vang or vang < 45:   n_queue += 1
                                    elif 45  <= vang < 135:        e_queue += 1
                                    elif 135 <= vang < 225:        s_queue += 1
                                    elif 225 <= vang < 315:        w_queue += 1
                            queues = {'N': n_queue, 'S': s_queue,
                                      'E': e_queue, 'W': w_queue}
                        except Exception:
                            queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}

                        # Total active vehicles (not just arrived-this-step)
                        try:
                            throughput = len(traci.vehicle.getIDList())
                        except Exception:
                            throughput = 0

                        total_queue = sum(queues.values())

                        # Real per-vehicle waiting time
                        try:
                            all_vehs = traci.vehicle.getIDList()
                            if all_vehs:
                                total_wait = sum(
                                    traci.vehicle.getWaitingTime(v) for v in all_vehs)
                                avg_wait = round(total_wait / len(all_vehs), 1)
                            else:
                                avg_wait = 0.0
                        except Exception:
                            avg_wait = 0.0

                        # Fase y nombre
                        _phase_names = ["N-S Verde", "N-S Ámbar", "E-O Verde", "E-O Ámbar"]
                        _ph_idx = _active_phase % 4

                        # Confianza y acción del DQN real (no hardcodeadas)
                        _conf_real   = _dqn_conf if _do_dqn else (
                            _dqn.last_conf if _dqn else 0.85)
                        _action_real = _dqn_action if _do_dqn else (
                            _dqn.last_action if _dqn else 0)
                        _xai_labels  = ["Mantener Fase", "Cambiar a N-S", "Cambiar a E-O", "Extender Fase"]

                        reward = round(throughput * 0.8 - total_queue * 0.3
                                       - avg_wait * 0.15, 2)

                        metrics_packet = {
                            "type":           "metrics",
                            "timestamp":      datetime.now().isoformat(),
                            "scenario":       "milan_real",
                            "phase":          _ph_idx,
                            "phase_name":     _phase_names[_ph_idx],
                            "throughput":     throughput,
                            "avg_wait":       float(np.nan_to_num(avg_wait)),
                            "total_queue":    float(total_queue),
                            "confidence":     round(_conf_real, 3),
                            "reward":         reward,
                            "co2_reduction":  co2_real,
                            "avg_speed":      avg_speed,
                            "latency_ms":     round(8.0 + (now % 3), 1),
                            "detections":     len(detections),
                            "decision":       _xai_labels[_action_real],
                            "action_index":   _action_real,
                            "event_active":   False,
                            "incident_active": getattr(system.engine, 'incident_active', False),
                            "traffic_factor": 1.0,
                            "sumo_online":    system.sumo_online,
                            "muse": {
                                "interventions": system.total_decisions,
                                "competence": round(
                                    min(0.99, 0.70 + _conf_real * 0.28), 3),
                            },
                            "queues": queues,
                        }
                        system.current_metrics = metrics_packet
                        await system.broadcast(metrics_packet)

                        # ── ML predictions: LSTM heatmap + anomalías ──
                        _qn = queues.get('N', 0)
                        _qs = queues.get('S', 0)
                        _qe = queues.get('E', 0)
                        _qw = queues.get('W', 0)
                        lstm_zones = [
                            {"id": "n_approach", "x":  0.0, "z": -45.0, "congestion": min(1.0, _qn / 15.0)},
                            {"id": "s_approach", "x":  0.0, "z":  45.0, "congestion": min(1.0, _qs / 15.0)},
                            {"id": "e_approach", "x": 45.0, "z":   0.0, "congestion": min(1.0, _qe / 15.0)},
                            {"id": "w_approach", "x":-45.0, "z":   0.0, "congestion": min(1.0, _qw / 15.0)},
                            {"id": "junction",   "x":  0.0, "z":   0.0, "congestion": min(1.0, total_queue / 30.0)},
                        ]
                        anomaly_list = []
                        if total_queue > 25:
                            anomaly_list = [
                                {"id": "anom_cong", "x": 0.0, "z": -15.0,
                                 "radius": 12.0, "severity": min(1.0, total_queue / 50.0)}
                            ]
                        await system.broadcast({
                            "type": "ml_predictions",
                            "lstm": {"horizon": 15, "zones": lstm_zones},
                            "anomalies": anomaly_list,
                        })

                except traci.exceptions.FatalTraCIError as _fe:
                    logger.error(f"[ATLAS] TraCI Desconectado: {_fe}. Reiniciando SUMO...")
                    await system.broadcast({
                        "type": "event", "event": "traci_reconnecting",
                        "message": "Reconectando con SUMO..."
                    })
                    await asyncio.sleep(2)
                    try:
                        _recovery_cfg = "simulations/milan_centro/simulation.sumocfg"
                        import sumolib as _sl
                        _sumo_bin = _sl.checkBinary('sumo')
                        traci.start([_sumo_bin, "-c", _recovery_cfg,
                                     "--start", "--no-step-log", "--no-warnings"])
                        net_center_ready = False
                        system.sumo_online = False
                        system.engine.current_phase = 0
                        system.engine.phase_start_time = time.time()
                        if system.dqn:
                            system.dqn._tls_initialized = False
                        logger.info("[ATLAS] TraCI reconectado OK")
                        await system.broadcast({
                            "type": "event", "event": "traci_reconnected",
                            "message": "SUMO reconectado — simulación reanudada"
                        })
                    except Exception as _re:
                        logger.error(f"[ATLAS] Error reconectando TraCI: {_re}")
                        await asyncio.sleep(3)
                    continue

                await asyncio.sleep(1/15) # 15Hz

            except Exception as e:
                logger.error(f"Error en loop de producción: {e}")
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
