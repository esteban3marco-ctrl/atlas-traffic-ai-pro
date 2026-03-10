"""
ATLAS Pro - Integraciones de Datos Externos
============================================
Módulo de integración con fuentes externas de datos para
mejorar la predicción y adaptación del sistema de tráfico.

Fuentes:
1. Eventos públicos (calendario, APIs de eventos)
2. Datos de navegación (Waze, Google Traffic, TomTom)
3. Transporte público (GTFS, buses, tranvías)
4. Datos de sensores IoT (contadores, cámaras, radar)

Se integra con: weather_integration.py, onda_verde.py, multi_interseccion.py
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger("ATLAS.DatosExternos")

try:
    import requests
    REQUESTS_DISPONIBLE = True
except ImportError:
    REQUESTS_DISPONIBLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EventCategory(Enum):
    """Categorías de eventos que afectan al tráfico"""
    SPORTS = "sports"           # Partidos, carreras
    CONCERT = "concert"         # Conciertos, festivales
    CONFERENCE = "conference"   # Conferencias, ferias
    PROTEST = "protest"         # Manifestaciones
    CONSTRUCTION = "construction"  # Obras viales
    HOLIDAY = "holiday"         # Festivos
    MARKET = "market"           # Mercados, ferias
    EMERGENCY = "emergency"     # Emergencias
    OTHER = "other"


@dataclass
class PublicEvent:
    """Evento público que puede afectar al tráfico"""
    event_id: str
    name: str
    category: EventCategory
    location_lat: float
    location_lon: float
    radius_m: float                 # Radio de impacto (metros)
    start_time: datetime
    end_time: datetime
    expected_attendance: int = 0    # Asistentes esperados
    traffic_impact: float = 0.5     # Impacto estimado (0=nulo, 1=severo)
    road_closures: List[str] = field(default_factory=list)  # IDs de vías cerradas
    source: str = "manual"

    @property
    def is_active(self) -> bool:
        now = datetime.now()
        # Considerar impacto 1h antes y 30min después
        return (self.start_time - timedelta(hours=1)) <= now <= (self.end_time + timedelta(minutes=30))

    @property
    def impact_phase(self) -> str:
        """Fase del evento: pre, during, post"""
        now = datetime.now()
        if now < self.start_time - timedelta(minutes=30):
            return "pre_arrival"
        elif now < self.start_time:
            return "arriving"
        elif now < self.end_time:
            return "during"
        elif now < self.end_time + timedelta(minutes=30):
            return "departing"
        return "none"


@dataclass
class NavigationData:
    """Datos de navegación en tiempo real (estilo Waze/Google)"""
    segment_id: str
    speed_kmh: float                # Velocidad actual
    free_flow_speed_kmh: float      # Velocidad en flujo libre
    congestion_level: float         # 0=libre, 1=congestión total
    travel_time_s: float            # Tiempo de viaje actual
    free_flow_time_s: float         # Tiempo de viaje libre
    incidents: List[str] = field(default_factory=list)  # Incidentes activos
    timestamp: float = 0.0

    @property
    def delay_ratio(self) -> float:
        """Ratio de retraso vs flujo libre"""
        if self.free_flow_time_s > 0:
            return self.travel_time_s / self.free_flow_time_s
        return 1.0


@dataclass
class TransitVehicle:
    """Vehículo de transporte público"""
    vehicle_id: str
    route_id: str
    route_name: str
    vehicle_type: str               # bus, tram, metro
    lat: float
    lon: float
    speed_kmh: float
    heading_deg: float              # Dirección (0=N, 90=E)
    occupancy: float = 0.5          # Ocupación (0-1)
    delay_s: float = 0.0            # Retraso respecto horario
    next_stop: str = ""
    priority_active: bool = False   # Si tiene TSP activo
    timestamp: float = 0.0


# =============================================================================
# EVENTOS PÚBLICOS
# =============================================================================

class EventsManager:
    """
    Gestor de eventos públicos que afectan al tráfico.

    Fuentes soportadas:
    - Manual (JSON/API interna)
    - PredictHQ API
    - Google Calendar API
    - Eventbrite API
    - Datos municipales
    """

    # Impacto por categoría de evento
    IMPACT_FACTORS = {
        EventCategory.SPORTS: {
            'base_impact': 0.7,
            'per_1000_attendees': 0.05,
            'pre_event_hours': 2.0,
            'post_event_hours': 1.5
        },
        EventCategory.CONCERT: {
            'base_impact': 0.6,
            'per_1000_attendees': 0.04,
            'pre_event_hours': 1.5,
            'post_event_hours': 1.0
        },
        EventCategory.CONFERENCE: {
            'base_impact': 0.3,
            'per_1000_attendees': 0.02,
            'pre_event_hours': 1.0,
            'post_event_hours': 0.5
        },
        EventCategory.PROTEST: {
            'base_impact': 0.8,
            'per_1000_attendees': 0.06,
            'pre_event_hours': 0.5,
            'post_event_hours': 0.5
        },
        EventCategory.CONSTRUCTION: {
            'base_impact': 0.4,
            'per_1000_attendees': 0.0,
            'pre_event_hours': 0.0,
            'post_event_hours': 0.0
        },
        EventCategory.HOLIDAY: {
            'base_impact': 0.3,
            'per_1000_attendees': 0.0,
            'pre_event_hours': 3.0,
            'post_event_hours': 3.0
        }
    }

    def __init__(self, api_key: str = None, city_lat: float = 40.4168,
                 city_lon: float = -3.7038):
        """
        Args:
            api_key: API key para PredictHQ u otro servicio
            city_lat/lon: Centro de la ciudad para búsqueda
        """
        self.api_key = api_key or os.environ.get("PREDICTHQ_API_KEY", "")
        self.city_lat = city_lat
        self.city_lon = city_lon
        self.events: Dict[str, PublicEvent] = {}
        self.events_file = "config/events.json"

        self._load_local_events()

    def _load_local_events(self):
        """Carga eventos desde archivo local"""
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file) as f:
                    data = json.load(f)
                for ev in data.get('events', []):
                    event = PublicEvent(
                        event_id=ev['event_id'],
                        name=ev['name'],
                        category=EventCategory(ev.get('category', 'other')),
                        location_lat=ev.get('lat', self.city_lat),
                        location_lon=ev.get('lon', self.city_lon),
                        radius_m=ev.get('radius_m', 500),
                        start_time=datetime.fromisoformat(ev['start_time']),
                        end_time=datetime.fromisoformat(ev['end_time']),
                        expected_attendance=ev.get('attendance', 0),
                        traffic_impact=ev.get('impact', 0.5),
                        road_closures=ev.get('road_closures', []),
                        source="local"
                    )
                    self.events[event.event_id] = event
                logger.info(f"EventsManager: {len(self.events)} eventos cargados desde archivo")
            except Exception as e:
                logger.warning(f"Error cargando eventos: {e}")

    def add_event(self, event: PublicEvent):
        """Añade evento manualmente"""
        self.events[event.event_id] = event
        logger.info(f"Evento añadido: {event.name} ({event.category.value})")

    def remove_event(self, event_id: str):
        """Elimina evento"""
        if event_id in self.events:
            del self.events[event_id]

    def get_active_events(self) -> List[PublicEvent]:
        """Retorna eventos activos que están afectando al tráfico"""
        return [e for e in self.events.values() if e.is_active]

    def fetch_from_api(self) -> List[PublicEvent]:
        """Obtiene eventos de API externa (PredictHQ o similar)"""
        if not REQUESTS_DISPONIBLE or not self.api_key:
            return self._generate_simulated_events()

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }

            params = {
                'within': f'5km@{self.city_lat},{self.city_lon}',
                'active.gte': datetime.now().isoformat(),
                'active.lte': (datetime.now() + timedelta(days=7)).isoformat(),
                'category': 'sports,concerts,conferences,expos,festivals',
                'limit': 50
            }

            resp = requests.get(
                'https://api.predicthq.com/v1/events/',
                headers=headers,
                params=params,
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                events = []
                for item in data.get('results', []):
                    event = PublicEvent(
                        event_id=item['id'],
                        name=item['title'],
                        category=self._map_category(item.get('category', '')),
                        location_lat=item.get('location', [0, 0])[1],
                        location_lon=item.get('location', [0, 0])[0],
                        radius_m=500,
                        start_time=datetime.fromisoformat(item['start'].replace('Z', '+00:00')),
                        end_time=datetime.fromisoformat(item['end'].replace('Z', '+00:00')),
                        expected_attendance=item.get('phq_attendance', {}).get('forecast', 0),
                        source="predicthq"
                    )
                    event.traffic_impact = self._estimate_impact(event)
                    events.append(event)
                    self.events[event.event_id] = event

                logger.info(f"PredictHQ: {len(events)} eventos obtenidos")
                return events
            else:
                logger.warning(f"PredictHQ API error: {resp.status_code}")
                return self._generate_simulated_events()

        except Exception as e:
            logger.warning(f"Error fetching events: {e}")
            return self._generate_simulated_events()

    def _generate_simulated_events(self) -> List[PublicEvent]:
        """Genera eventos simulados para demo/testing"""
        now = datetime.now()
        simulated = [
            PublicEvent(
                event_id="sim_football_001",
                name="Partido Liga Local",
                category=EventCategory.SPORTS,
                location_lat=self.city_lat + 0.015,
                location_lon=self.city_lon - 0.008,
                radius_m=1000,
                start_time=now + timedelta(hours=2),
                end_time=now + timedelta(hours=4),
                expected_attendance=25000,
                traffic_impact=0.75,
                source="simulated"
            ),
            PublicEvent(
                event_id="sim_concert_001",
                name="Festival de Verano",
                category=EventCategory.CONCERT,
                location_lat=self.city_lat - 0.01,
                location_lon=self.city_lon + 0.005,
                radius_m=800,
                start_time=now + timedelta(hours=5),
                end_time=now + timedelta(hours=9),
                expected_attendance=15000,
                traffic_impact=0.6,
                source="simulated"
            ),
            PublicEvent(
                event_id="sim_construction_001",
                name="Obras Avenida Central",
                category=EventCategory.CONSTRUCTION,
                location_lat=self.city_lat,
                location_lon=self.city_lon,
                radius_m=300,
                start_time=now - timedelta(days=5),
                end_time=now + timedelta(days=10),
                expected_attendance=0,
                traffic_impact=0.4,
                road_closures=["road_av_central_2", "road_av_central_3"],
                source="simulated"
            )
        ]

        for ev in simulated:
            self.events[ev.event_id] = ev

        return simulated

    def _map_category(self, cat: str) -> EventCategory:
        """Mapea categoría de API a EventCategory"""
        mapping = {
            'sports': EventCategory.SPORTS,
            'concerts': EventCategory.CONCERT,
            'conferences': EventCategory.CONFERENCE,
            'expos': EventCategory.CONFERENCE,
            'festivals': EventCategory.CONCERT,
            'community': EventCategory.OTHER,
            'performing-arts': EventCategory.CONCERT
        }
        return mapping.get(cat, EventCategory.OTHER)

    def _estimate_impact(self, event: PublicEvent) -> float:
        """Estima impacto de tráfico basado en categoría y asistencia"""
        factors = self.IMPACT_FACTORS.get(event.category, {
            'base_impact': 0.3, 'per_1000_attendees': 0.02
        })

        impact = factors['base_impact']
        if event.expected_attendance > 0:
            impact += factors['per_1000_attendees'] * (event.expected_attendance / 1000)

        return min(1.0, impact)

    def get_traffic_adjustment(self, lat: float, lon: float) -> Dict:
        """
        Calcula ajuste de tráfico para una ubicación basado en eventos activos.

        Returns:
            {
                'flow_multiplier': float,    # Multiplicador de flujo (1.0=normal)
                'capacity_reduction': float,  # Reducción de capacidad (0=ninguna)
                'active_events': List[str],   # Nombres de eventos activos
                'phase': str,                 # Fase dominante
                'road_closures': List[str]    # Vías cerradas
            }
        """
        active = self.get_active_events()

        if not active:
            return {
                'flow_multiplier': 1.0,
                'capacity_reduction': 0.0,
                'active_events': [],
                'phase': 'none',
                'road_closures': []
            }

        flow_mult = 1.0
        cap_reduction = 0.0
        event_names = []
        closures = []
        dominant_phase = "none"

        for event in active:
            # Calcular distancia
            dist = self._haversine(lat, lon, event.location_lat, event.location_lon)

            if dist <= event.radius_m:
                # Dentro del radio de impacto
                proximity = 1.0 - (dist / event.radius_m)  # 1.0=centro, 0.0=borde
                phase = event.impact_phase

                if phase in ("arriving", "departing"):
                    flow_mult += event.traffic_impact * proximity * 0.8
                elif phase == "during":
                    flow_mult += event.traffic_impact * proximity * 0.3
                    cap_reduction += event.traffic_impact * proximity * 0.2

                if phase == "arriving":
                    dominant_phase = "arriving"
                elif phase == "departing" and dominant_phase != "arriving":
                    dominant_phase = "departing"

                event_names.append(event.name)
                closures.extend(event.road_closures)

        return {
            'flow_multiplier': round(flow_mult, 3),
            'capacity_reduction': round(min(0.5, cap_reduction), 3),
            'active_events': event_names,
            'phase': dominant_phase,
            'road_closures': list(set(closures))
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Distancia haversine en metros"""
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# =============================================================================
# DATOS DE NAVEGACIÓN
# =============================================================================

class NavigationDataProvider:
    """
    Proveedor de datos de navegación en tiempo real.

    Soporta:
    - TomTom Traffic API
    - Google Routes API
    - HERE Traffic API
    - Simulación para demo/testing
    """

    def __init__(self, provider: str = "simulation",
                 api_key: str = None):
        """
        Args:
            provider: "tomtom", "google", "here", "simulation"
            api_key: API key del proveedor
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get("TRAFFIC_API_KEY", "")
        self.cache: Dict[str, NavigationData] = {}
        self.cache_ttl = 120  # 2 minutos
        self.history: deque = deque(maxlen=288)  # 24h a 5min interval

        logger.info(f"NavigationDataProvider: provider={provider}")

    def get_segment_data(self, segments: List[Dict]) -> List[NavigationData]:
        """
        Obtiene datos de tráfico para segmentos de vía.

        Args:
            segments: [{'id': 'seg_01', 'start_lat': ..., 'start_lon': ...,
                        'end_lat': ..., 'end_lon': ..., 'length_m': ...}]
        """
        if self.provider == "tomtom" and self.api_key:
            return self._fetch_tomtom(segments)
        elif self.provider == "google" and self.api_key:
            return self._fetch_google(segments)
        else:
            return self._simulate_traffic(segments)

    def _fetch_tomtom(self, segments: List[Dict]) -> List[NavigationData]:
        """Obtiene datos de TomTom Traffic Flow API"""
        results = []

        for seg in segments:
            cache_key = seg['id']
            if self._cache_valid(cache_key):
                results.append(self.cache[cache_key])
                continue

            try:
                lat = (seg['start_lat'] + seg['end_lat']) / 2
                lon = (seg['start_lon'] + seg['end_lon']) / 2

                url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
                       f"absolute/10/json?point={lat},{lon}&key={self.api_key}")

                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get('flowSegmentData', {})
                    nav = NavigationData(
                        segment_id=seg['id'],
                        speed_kmh=data.get('currentSpeed', 50),
                        free_flow_speed_kmh=data.get('freeFlowSpeed', 50),
                        congestion_level=1 - data.get('currentSpeed', 50) / max(data.get('freeFlowSpeed', 50), 1),
                        travel_time_s=data.get('currentTravelTime', 60),
                        free_flow_time_s=data.get('freeFlowTravelTime', 60),
                        timestamp=time.time()
                    )
                    results.append(nav)
                    self.cache[cache_key] = nav
                else:
                    results.append(self._simulate_segment(seg))

            except Exception as e:
                logger.warning(f"TomTom error for {seg['id']}: {e}")
                results.append(self._simulate_segment(seg))

        return results

    def _fetch_google(self, segments: List[Dict]) -> List[NavigationData]:
        """Obtiene datos de Google Routes API"""
        results = []

        for seg in segments:
            cache_key = seg['id']
            if self._cache_valid(cache_key):
                results.append(self.cache[cache_key])
                continue

            try:
                url = "https://routes.googleapis.com/directions/v2:computeRoutes"
                headers = {
                    'X-Goog-Api-Key': self.api_key,
                    'X-Goog-FieldMask': 'routes.duration,routes.staticDuration,routes.distanceMeters'
                }
                body = {
                    'origin': {'location': {'latLng': {
                        'latitude': seg['start_lat'], 'longitude': seg['start_lon']
                    }}},
                    'destination': {'location': {'latLng': {
                        'latitude': seg['end_lat'], 'longitude': seg['end_lon']
                    }}},
                    'travelMode': 'DRIVE',
                    'routingPreference': 'TRAFFIC_AWARE'
                }

                resp = requests.post(url, json=body, headers=headers, timeout=10)
                if resp.status_code == 200:
                    route = resp.json().get('routes', [{}])[0]
                    duration = int(route.get('duration', '60s').replace('s', ''))
                    static_duration = int(route.get('staticDuration', '60s').replace('s', ''))
                    distance = route.get('distanceMeters', seg.get('length_m', 500))

                    speed = (distance / duration) * 3.6 if duration > 0 else 50
                    ff_speed = (distance / static_duration) * 3.6 if static_duration > 0 else 50

                    nav = NavigationData(
                        segment_id=seg['id'],
                        speed_kmh=speed,
                        free_flow_speed_kmh=ff_speed,
                        congestion_level=max(0, 1 - speed / max(ff_speed, 1)),
                        travel_time_s=duration,
                        free_flow_time_s=static_duration,
                        timestamp=time.time()
                    )
                    results.append(nav)
                    self.cache[cache_key] = nav
                else:
                    results.append(self._simulate_segment(seg))

            except Exception as e:
                logger.warning(f"Google Routes error for {seg['id']}: {e}")
                results.append(self._simulate_segment(seg))

        return results

    def _simulate_traffic(self, segments: List[Dict]) -> List[NavigationData]:
        """Genera datos de tráfico simulados realistas"""
        results = []
        hour = datetime.now().hour

        # Patrones de congestión por hora (0=libre, 1=máxima)
        congestion_pattern = {
            0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.1,
            6: 0.25, 7: 0.6, 8: 0.85, 9: 0.7, 10: 0.45, 11: 0.5,
            12: 0.55, 13: 0.6, 14: 0.5, 15: 0.55, 16: 0.65, 17: 0.8,
            18: 0.9, 19: 0.75, 20: 0.5, 21: 0.35, 22: 0.2, 23: 0.1
        }

        base_congestion = congestion_pattern.get(hour, 0.3)

        for seg in segments:
            # Añadir variación por segmento
            noise = np.random.normal(0, 0.1)
            congestion = max(0, min(1.0, base_congestion + noise))

            ff_speed = seg.get('speed_limit', 50)
            current_speed = ff_speed * (1 - congestion * 0.7)
            length_m = seg.get('length_m', 500)
            travel_time = length_m / (current_speed / 3.6) if current_speed > 0 else 999
            ff_time = length_m / (ff_speed / 3.6)

            nav = NavigationData(
                segment_id=seg['id'],
                speed_kmh=round(current_speed, 1),
                free_flow_speed_kmh=ff_speed,
                congestion_level=round(congestion, 3),
                travel_time_s=round(travel_time, 1),
                free_flow_time_s=round(ff_time, 1),
                timestamp=time.time()
            )
            results.append(nav)
            self.cache[seg['id']] = nav

        # Guardar en histórico
        self.history.append({
            'timestamp': time.time(),
            'segments': {r.segment_id: r.congestion_level for r in results}
        })

        return results

    def _simulate_segment(self, seg: Dict) -> NavigationData:
        """Simula un segmento individual"""
        return self._simulate_traffic([seg])[0]

    def _cache_valid(self, key: str) -> bool:
        """Verifica si cache es válida"""
        if key in self.cache:
            age = time.time() - self.cache[key].timestamp
            return age < self.cache_ttl
        return False

    def get_congestion_summary(self) -> Dict:
        """Resumen de congestión de todos los segmentos en cache"""
        if not self.cache:
            return {'avg_congestion': 0, 'max_congestion': 0, 'segments_monitored': 0}

        congestions = [v.congestion_level for v in self.cache.values()]
        return {
            'avg_congestion': round(np.mean(congestions), 3),
            'max_congestion': round(max(congestions), 3),
            'min_congestion': round(min(congestions), 3),
            'segments_monitored': len(congestions),
            'segments_congested': sum(1 for c in congestions if c > 0.6),
            'timestamp': time.time()
        }

    def get_congestion_prediction(self, hours_ahead: int = 1) -> Dict:
        """
        Predice congestión futura basado en histórico.

        Usa patrón típico del día + tendencia reciente.
        """
        future_hour = (datetime.now().hour + hours_ahead) % 24
        congestion_typical = {
            0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.1,
            6: 0.25, 7: 0.6, 8: 0.85, 9: 0.7, 10: 0.45, 11: 0.5,
            12: 0.55, 13: 0.6, 14: 0.5, 15: 0.55, 16: 0.65, 17: 0.8,
            18: 0.9, 19: 0.75, 20: 0.5, 21: 0.35, 22: 0.2, 23: 0.1
        }

        predicted = congestion_typical.get(future_hour, 0.3)

        # Ajustar con tendencia reciente
        if len(self.history) >= 3:
            recent = [np.mean(list(h['segments'].values())) for h in list(self.history)[-3:]]
            trend = recent[-1] - recent[0]
            predicted += trend * 0.3

        return {
            'hour': future_hour,
            'predicted_congestion': round(max(0, min(1, predicted)), 3),
            'confidence': 0.7 if len(self.history) >= 6 else 0.4
        }


# =============================================================================
# TRANSPORTE PÚBLICO
# =============================================================================

class TransitManager:
    """
    Gestor de transporte público para TSP (Transit Signal Priority).

    Soporta:
    - GTFS-RT (General Transit Feed Specification Realtime)
    - API municipal de buses/tranvías
    - Simulación para testing
    """

    def __init__(self, gtfs_rt_url: str = None,
                 api_key: str = None,
                 priority_threshold_delay_s: float = 120.0):
        """
        Args:
            gtfs_rt_url: URL del feed GTFS-RT
            api_key: API key para el servicio
            priority_threshold_delay_s: Retraso mínimo para activar TSP (s)
        """
        self.gtfs_rt_url = gtfs_rt_url
        self.api_key = api_key
        self.priority_threshold = priority_threshold_delay_s
        self.vehicles: Dict[str, TransitVehicle] = {}
        self.priority_requests: List[Dict] = []

        logger.info(f"TransitManager: threshold={priority_threshold_delay_s}s")

    def update_vehicles(self) -> List[TransitVehicle]:
        """Actualiza posiciones de vehículos de transporte público"""
        if self.gtfs_rt_url and REQUESTS_DISPONIBLE:
            return self._fetch_gtfs_rt()
        return self._simulate_vehicles()

    def _fetch_gtfs_rt(self) -> List[TransitVehicle]:
        """Obtiene datos de GTFS-RT feed"""
        try:
            # GTFS-RT puede ser protobuf o JSON
            resp = requests.get(self.gtfs_rt_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                vehicles = []
                for entity in data.get('entity', []):
                    vp = entity.get('vehicle', {})
                    pos = vp.get('position', {})
                    trip = vp.get('trip', {})

                    vehicle = TransitVehicle(
                        vehicle_id=vp.get('vehicle', {}).get('id', ''),
                        route_id=trip.get('routeId', ''),
                        route_name=trip.get('routeId', ''),
                        vehicle_type='bus',
                        lat=pos.get('latitude', 0),
                        lon=pos.get('longitude', 0),
                        speed_kmh=pos.get('speed', 0) * 3.6,
                        heading_deg=pos.get('bearing', 0),
                        delay_s=vp.get('timestamp', 0),
                        timestamp=time.time()
                    )
                    vehicles.append(vehicle)
                    self.vehicles[vehicle.vehicle_id] = vehicle

                return vehicles
        except Exception as e:
            logger.warning(f"GTFS-RT error: {e}")

        return self._simulate_vehicles()

    def _simulate_vehicles(self) -> List[TransitVehicle]:
        """Simula vehículos de transporte público"""
        vehicles = []
        routes = [
            ("bus_L1", "Línea 1 - Centro", "bus"),
            ("bus_L2", "Línea 2 - Norte-Sur", "bus"),
            ("bus_L5", "Línea 5 - Circular", "bus"),
            ("tram_T1", "Tranvía T1", "tram"),
        ]

        for route_id, route_name, vtype in routes:
            for i in range(3):  # 3 vehículos por línea
                vid = f"{route_id}_v{i}"
                vehicle = TransitVehicle(
                    vehicle_id=vid,
                    route_id=route_id,
                    route_name=route_name,
                    vehicle_type=vtype,
                    lat=40.4168 + np.random.uniform(-0.02, 0.02),
                    lon=-3.7038 + np.random.uniform(-0.02, 0.02),
                    speed_kmh=round(np.random.uniform(15, 40), 1),
                    heading_deg=round(np.random.uniform(0, 360), 1),
                    occupancy=round(np.random.uniform(0.2, 0.9), 2),
                    delay_s=round(np.random.uniform(-30, 300), 0),
                    next_stop=f"stop_{np.random.randint(1, 20):02d}",
                    timestamp=time.time()
                )
                vehicles.append(vehicle)
                self.vehicles[vid] = vehicle

        return vehicles

    def check_priority_requests(self,
                                intersections: List[Dict]) -> List[Dict]:
        """
        Verifica qué vehículos necesitan TSP en qué intersecciones.

        Args:
            intersections: [{'id': 'INT_01', 'lat': ..., 'lon': ..., 'radius_m': 200}]

        Returns:
            Lista de solicitudes de prioridad:
            [{'vehicle_id': ..., 'intersection_id': ..., 'delay_s': ...,
              'eta_s': ..., 'priority_level': 'high'|'medium'|'low'}]
        """
        self.priority_requests = []

        for vehicle in self.vehicles.values():
            if vehicle.delay_s < self.priority_threshold * 0.5:
                continue  # No necesita prioridad

            for intersection in intersections:
                dist = EventsManager._haversine(
                    vehicle.lat, vehicle.lon,
                    intersection['lat'], intersection['lon']
                )

                approach_radius = intersection.get('radius_m', 200)

                if dist <= approach_radius:
                    eta = (dist / (vehicle.speed_kmh / 3.6)) if vehicle.speed_kmh > 0 else 999

                    priority = "low"
                    if vehicle.delay_s > self.priority_threshold:
                        priority = "high"
                    elif vehicle.delay_s > self.priority_threshold * 0.7:
                        priority = "medium"

                    # Tranvías tienen prioridad mayor
                    if vehicle.vehicle_type == "tram":
                        priority = "high"

                    request = {
                        'vehicle_id': vehicle.vehicle_id,
                        'route_name': vehicle.route_name,
                        'vehicle_type': vehicle.vehicle_type,
                        'intersection_id': intersection['id'],
                        'distance_m': round(dist, 1),
                        'delay_s': vehicle.delay_s,
                        'eta_s': round(eta, 1),
                        'occupancy': vehicle.occupancy,
                        'priority_level': priority,
                        'timestamp': time.time()
                    }
                    self.priority_requests.append(request)

        # Ordenar por prioridad y ETA
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        self.priority_requests.sort(
            key=lambda r: (priority_order.get(r['priority_level'], 3), r['eta_s'])
        )

        return self.priority_requests

    def get_transit_summary(self) -> Dict:
        """Resumen del transporte público"""
        if not self.vehicles:
            return {'total_vehicles': 0}

        delays = [v.delay_s for v in self.vehicles.values()]
        occupancies = [v.occupancy for v in self.vehicles.values()]

        return {
            'total_vehicles': len(self.vehicles),
            'buses': sum(1 for v in self.vehicles.values() if v.vehicle_type == 'bus'),
            'trams': sum(1 for v in self.vehicles.values() if v.vehicle_type == 'tram'),
            'avg_delay_s': round(np.mean(delays), 1),
            'max_delay_s': round(max(delays), 1),
            'vehicles_delayed': sum(1 for d in delays if d > self.priority_threshold),
            'avg_occupancy': round(np.mean(occupancies), 2),
            'active_priority_requests': len(self.priority_requests)
        }


# =============================================================================
# INTEGRADOR CENTRAL
# =============================================================================

class ExternalDataIntegrator:
    """
    Integrador central de todos los datos externos.

    Combina: meteorología, eventos, navegación, transporte público
    para generar un vector de ajuste unificado para el agente RL.
    """

    def __init__(self, config: Dict = None):
        cfg = config or {}

        self.events_manager = EventsManager(
            api_key=cfg.get('events_api_key'),
            city_lat=cfg.get('city_lat', 40.4168),
            city_lon=cfg.get('city_lon', -3.7038)
        )

        self.nav_provider = NavigationDataProvider(
            provider=cfg.get('nav_provider', 'simulation'),
            api_key=cfg.get('nav_api_key')
        )

        self.transit_manager = TransitManager(
            gtfs_rt_url=cfg.get('gtfs_rt_url'),
            priority_threshold_delay_s=cfg.get('tsp_threshold', 120)
        )

        # Weather integration (importar si disponible)
        self.weather = None
        try:
            from weather_integration import WeatherIntegration
            self.weather = WeatherIntegration(
                api_key=cfg.get('weather_api_key')
            )
        except ImportError:
            logger.info("WeatherIntegration no disponible, continuando sin meteorología")

        logger.info("ExternalDataIntegrator inicializado")

    def get_full_context(self, lat: float = None, lon: float = None,
                        segments: List[Dict] = None,
                        intersections: List[Dict] = None) -> Dict:
        """
        Obtiene contexto completo de datos externos.

        Returns:
            {
                'weather': {...},
                'events': {...},
                'navigation': {...},
                'transit': {...},
                'combined_adjustment': {...},
                'rl_state_vector': [...]
            }
        """
        lat = lat or 40.4168
        lon = lon or -3.7038

        # Weather
        weather_data = {}
        weather_adj = {}
        if self.weather:
            weather_adj = self.weather.get_adjustment()
            weather_data = weather_adj

        # Events
        events_adj = self.events_manager.get_traffic_adjustment(lat, lon)

        # Navigation
        nav_data = []
        if segments:
            nav_data = self.nav_provider.get_segment_data(segments)
        nav_summary = self.nav_provider.get_congestion_summary()

        # Transit
        self.transit_manager.update_vehicles()
        transit_priority = []
        if intersections:
            transit_priority = self.transit_manager.check_priority_requests(intersections)
        transit_summary = self.transit_manager.get_transit_summary()

        # Calcular ajuste combinado
        combined = self._compute_combined_adjustment(
            weather_adj, events_adj, nav_summary, transit_summary
        )

        # Vector RL
        rl_vector = self._build_rl_state_vector(
            weather_adj, events_adj, nav_summary, transit_summary
        )

        return {
            'weather': weather_data,
            'events': {
                'adjustment': events_adj,
                'active_count': len(self.events_manager.get_active_events())
            },
            'navigation': {
                'summary': nav_summary,
                'prediction': self.nav_provider.get_congestion_prediction()
            },
            'transit': {
                'summary': transit_summary,
                'priority_requests': transit_priority[:5]  # Top 5
            },
            'combined_adjustment': combined,
            'rl_state_vector': rl_vector
        }

    def _compute_combined_adjustment(self, weather: Dict, events: Dict,
                                      nav: Dict, transit: Dict) -> Dict:
        """Calcula ajuste combinado de todas las fuentes"""
        # Flow multiplier
        flow_mult = events.get('flow_multiplier', 1.0)

        # Capacity reduction
        cap_reduction = events.get('capacity_reduction', 0.0)
        if weather.get('capacity_factor', 1.0) < 1.0:
            cap_reduction += (1.0 - weather.get('capacity_factor', 1.0))

        # Speed adjustment
        speed_factor = 1.0
        if weather.get('speed_factor', 1.0) < 1.0:
            speed_factor = weather['speed_factor']
        if nav.get('avg_congestion', 0) > 0.5:
            speed_factor *= (1.0 - nav['avg_congestion'] * 0.3)

        # Cycle time suggestion
        congestion = nav.get('avg_congestion', 0.3)
        if congestion > 0.7:
            suggested_cycle = 120  # Ciclos más largos en congestión
        elif congestion < 0.2:
            suggested_cycle = 60   # Ciclos más cortos en flujo libre
        else:
            suggested_cycle = 90

        # Priority level
        priority = "normal"
        if transit.get('active_priority_requests', 0) > 0:
            priority = "transit_priority"
        if len(events.get('road_closures', [])) > 0:
            priority = "event_management"
        if weather.get('severity', 0) > 0.7:
            priority = "weather_alert"

        return {
            'flow_multiplier': round(flow_mult, 3),
            'capacity_reduction': round(min(0.5, cap_reduction), 3),
            'speed_factor': round(speed_factor, 3),
            'suggested_cycle_s': suggested_cycle,
            'priority_mode': priority,
            'overall_severity': round(min(1.0, congestion + cap_reduction), 3)
        }

    def _build_rl_state_vector(self, weather: Dict, events: Dict,
                                nav: Dict, transit: Dict) -> List[float]:
        """
        Construye vector de estado para el agente RL.

        Tamaño fijo: 12 elementos
        """
        vector = [
            # Weather (3)
            weather.get('capacity_factor', 1.0),
            weather.get('speed_factor', 1.0),
            weather.get('severity', 0.0),
            # Events (3)
            events.get('flow_multiplier', 1.0) - 1.0,  # Delta sobre normal
            events.get('capacity_reduction', 0.0),
            1.0 if events.get('phase', 'none') in ('arriving', 'departing') else 0.0,
            # Navigation (3)
            nav.get('avg_congestion', 0.0),
            nav.get('max_congestion', 0.0),
            nav.get('segments_congested', 0) / max(nav.get('segments_monitored', 1), 1),
            # Transit (3)
            min(1.0, transit.get('avg_delay_s', 0) / 300.0),
            transit.get('avg_occupancy', 0.5),
            min(1.0, transit.get('active_priority_requests', 0) / 5.0)
        ]

        return [round(v, 4) for v in vector]


# =============================================================================
# DEMO
# =============================================================================

def demo_datos_externos():
    """Demo del sistema de datos externos"""

    print("=" * 70)
    print("  ATLAS Pro - Demo Datos Externos")
    print("=" * 70)

    integrator = ExternalDataIntegrator()

    # Simular segmentos de vía
    segments = [
        {'id': f'seg_{i:02d}', 'start_lat': 40.41 + i * 0.002,
         'start_lon': -3.70, 'end_lat': 40.41 + (i + 1) * 0.002,
         'end_lon': -3.70, 'length_m': 200, 'speed_limit': 50}
        for i in range(5)
    ]

    intersections = [
        {'id': f'INT_{i:02d}', 'lat': 40.41 + i * 0.002, 'lon': -3.70, 'radius_m': 200}
        for i in range(5)
    ]

    # Obtener contexto completo
    context = integrator.get_full_context(
        lat=40.4168, lon=-3.7038,
        segments=segments,
        intersections=intersections
    )

    print("\n--- Eventos ---")
    print(f"  Activos: {context['events']['active_count']}")
    adj = context['events']['adjustment']
    print(f"  Flow multiplier: {adj['flow_multiplier']}")
    print(f"  Eventos: {adj['active_events']}")

    print("\n--- Navegación ---")
    nav = context['navigation']['summary']
    print(f"  Segmentos monitorizados: {nav.get('segments_monitored', 0)}")
    print(f"  Congestión media: {nav.get('avg_congestion', 0):.1%}")
    print(f"  Congestión máxima: {nav.get('max_congestion', 0):.1%}")
    pred = context['navigation']['prediction']
    print(f"  Predicción +1h: {pred.get('predicted_congestion', 0):.1%} (conf: {pred.get('confidence', 0):.0%})")

    print("\n--- Transporte Público ---")
    tr = context['transit']['summary']
    print(f"  Vehículos: {tr.get('total_vehicles', 0)} ({tr.get('buses', 0)} buses, {tr.get('trams', 0)} tranvías)")
    print(f"  Retraso medio: {tr.get('avg_delay_s', 0):.0f}s")
    print(f"  Solicitudes TSP: {tr.get('active_priority_requests', 0)}")

    print("\n--- Ajuste Combinado ---")
    comb = context['combined_adjustment']
    print(f"  Flow multiplier:    {comb['flow_multiplier']}")
    print(f"  Capacity reduction: {comb['capacity_reduction']}")
    print(f"  Speed factor:       {comb['speed_factor']}")
    print(f"  Suggested cycle:    {comb['suggested_cycle_s']}s")
    print(f"  Priority mode:      {comb['priority_mode']}")

    print("\n--- Vector RL (12D) ---")
    print(f"  {context['rl_state_vector']}")

    print("\n" + "=" * 70)
    print("  Demo completada OK")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_datos_externos()
