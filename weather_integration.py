#!/usr/bin/env python3
"""
ATLAS Pro - Integración Meteorológica para Control de Tráfico
=============================================================
Ajusta el comportamiento del agente basándose en condiciones climáticas.

Funcionalidades:
- Consulta de API meteorológica (OpenWeatherMap / simulado)
- Factor de ajuste de recompensa por clima
- Detección de condiciones peligrosas
- Historial meteorológico y estadísticas
- Integración con el vector de estado 26D

Las condiciones climáticas afectan el tráfico:
- Lluvia: +15-40% tiempo de frenado, -10-25% velocidad
- Niebla: -20-40% visibilidad, velocidades reducidas
- Nieve/hielo: +30-60% tiempo de frenado, -20-40% velocidad
- Viento fuerte: Afecta vehículos altos, ciclistas
- Calor extremo: Más tráfico a ciertas horas (AC en autos)
"""

import os
import json
import time
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque

logger = logging.getLogger("ATLAS.Weather")

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WeatherCondition:
    """Condición meteorológica actual"""
    timestamp: str
    temperature: float          # °C
    humidity: float             # %
    wind_speed: float           # km/h
    wind_direction: float       # grados
    visibility: float           # km
    precipitation: float        # mm/h
    precipitation_type: str     # "none", "rain", "snow", "sleet"
    cloud_cover: float          # %
    condition: str              # "clear", "cloudy", "rain", "storm", "fog", "snow"
    condition_severity: float   # 0.0 (perfecto) a 1.0 (extremo)
    uv_index: float             # 0-11+
    pressure: float             # hPa
    source: str                 # "api", "simulated"

    def to_dict(self):
        return asdict(self)


@dataclass
class TrafficWeatherImpact:
    """Impacto del clima en el tráfico"""
    speed_factor: float         # Multiplicador de velocidad (1.0 = normal)
    braking_factor: float       # Multiplicador de distancia de frenado
    visibility_factor: float    # Factor de visibilidad (1.0 = perfecta)
    demand_factor: float        # Factor de demanda de tráfico
    risk_level: str             # "low", "moderate", "high", "extreme"
    recommendations: List[str]  # Recomendaciones para el controlador
    reward_modifier: float      # Modificador de recompensa (-0.3 a 0.3)


# =============================================================================
# PROVEEDOR METEOROLÓGICO
# =============================================================================

class WeatherProvider:
    """
    Obtiene datos meteorológicos de OpenWeatherMap o simula datos.
    API key gratuita: https://openweathermap.org/api
    """

    def __init__(self, api_key: str = None, lat: float = 40.4168,
                 lon: float = -3.7038, cache_minutes: int = 10):
        """
        Args:
            api_key: OpenWeatherMap API key (None = modo simulado)
            lat, lon: Coordenadas (default: Madrid)
            cache_minutes: Minutos de cache para la API
        """
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
        self.lat = lat
        self.lon = lon
        self.cache_minutes = cache_minutes
        self._cache: Optional[WeatherCondition] = None
        self._cache_time: float = 0
        self._simulated = not (self.api_key and REQUESTS_OK)

        if self._simulated:
            logger.info("WeatherProvider en modo SIMULADO")
        else:
            logger.info(f"WeatherProvider: API conectada, coords=({lat}, {lon})")

    def get_current(self) -> WeatherCondition:
        """Obtiene condiciones meteorológicas actuales"""
        # Check cache
        if self._cache and (time.time() - self._cache_time) < self.cache_minutes * 60:
            return self._cache

        if self._simulated:
            weather = self._simulate_weather()
        else:
            weather = self._fetch_api()

        self._cache = weather
        self._cache_time = time.time()
        return weather

    def _fetch_api(self) -> WeatherCondition:
        """Consulta OpenWeatherMap API"""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": self.lat,
                "lon": self.lon,
                "appid": self.api_key,
                "units": "metric",
                "lang": "es"
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Parsear respuesta
            main = data.get("main", {})
            wind = data.get("wind", {})
            clouds = data.get("clouds", {})
            weather_info = data.get("weather", [{}])[0]
            rain = data.get("rain", {})
            snow = data.get("snow", {})

            # Determinar tipo de precipitación
            precip = rain.get("1h", 0) + snow.get("1h", 0)
            if snow.get("1h", 0) > 0:
                precip_type = "snow"
            elif precip > 0:
                precip_type = "rain"
            else:
                precip_type = "none"

            # Mapear condición
            weather_id = weather_info.get("id", 800)
            condition = self._map_condition(weather_id)
            severity = self._calc_severity(weather_id, precip, wind.get("speed", 0) * 3.6)

            return WeatherCondition(
                timestamp=datetime.now().isoformat(),
                temperature=main.get("temp", 20),
                humidity=main.get("humidity", 50),
                wind_speed=wind.get("speed", 0) * 3.6,  # m/s -> km/h
                wind_direction=wind.get("deg", 0),
                visibility=data.get("visibility", 10000) / 1000,  # m -> km
                precipitation=precip,
                precipitation_type=precip_type,
                cloud_cover=clouds.get("all", 0),
                condition=condition,
                condition_severity=severity,
                uv_index=0,  # Necesita One Call API
                pressure=main.get("pressure", 1013),
                source="api"
            )

        except Exception as e:
            logger.error(f"Error API meteorológica: {e}")
            return self._simulate_weather()

    def _simulate_weather(self) -> WeatherCondition:
        """Genera datos meteorológicos simulados realistas"""
        import random

        hour = datetime.now().hour

        # Temperatura según hora
        temp_base = 18 + 8 * math.sin((hour - 6) * math.pi / 12)
        temp = temp_base + random.gauss(0, 2)

        # Condición aleatoria ponderada
        conditions = ["clear", "cloudy", "rain", "fog", "storm"]
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        condition = random.choices(conditions, weights=weights, k=1)[0]

        precip = 0.0
        precip_type = "none"
        visibility = 10.0

        if condition == "rain":
            precip = random.uniform(0.5, 15.0)
            precip_type = "rain"
            visibility = max(2.0, 10.0 - precip * 0.5)
        elif condition == "storm":
            precip = random.uniform(10.0, 40.0)
            precip_type = "rain"
            visibility = max(0.5, 5.0 - precip * 0.1)
        elif condition == "fog":
            visibility = random.uniform(0.1, 2.0)
        elif temp < 0 and condition in ["rain", "storm"]:
            precip_type = "snow"

        severity = self._calc_severity_simple(condition, precip, visibility)

        return WeatherCondition(
            timestamp=datetime.now().isoformat(),
            temperature=round(temp, 1),
            humidity=round(random.uniform(30, 95), 1),
            wind_speed=round(random.uniform(0, 40), 1),
            wind_direction=round(random.uniform(0, 360), 1),
            visibility=round(visibility, 1),
            precipitation=round(precip, 1),
            precipitation_type=precip_type,
            cloud_cover=round(random.uniform(0, 100), 0),
            condition=condition,
            condition_severity=round(severity, 2),
            uv_index=round(max(0, random.uniform(0, 8) * math.sin(max(0, (hour-6)*math.pi/12))), 1),
            pressure=round(random.uniform(1000, 1030), 1),
            source="simulated"
        )

    def _map_condition(self, weather_id: int) -> str:
        """Mapea OpenWeatherMap ID a condición simplificada"""
        if 200 <= weather_id < 300:
            return "storm"
        elif 300 <= weather_id < 500:
            return "rain"  # drizzle
        elif 500 <= weather_id < 600:
            return "rain"
        elif 600 <= weather_id < 700:
            return "snow"
        elif 700 <= weather_id < 800:
            return "fog"
        elif weather_id == 800:
            return "clear"
        else:
            return "cloudy"

    def _calc_severity(self, weather_id: int, precip: float, wind_kmh: float) -> float:
        """Calcula severidad 0-1 basada en condiciones"""
        severity = 0.0
        # Precipitación
        if precip > 0:
            severity += min(0.4, precip / 30.0 * 0.4)
        # Viento
        if wind_kmh > 20:
            severity += min(0.2, (wind_kmh - 20) / 60 * 0.2)
        # Tormenta
        if 200 <= weather_id < 300:
            severity += 0.3
        # Nieve
        if 600 <= weather_id < 700:
            severity += 0.2
        return min(1.0, severity)

    def _calc_severity_simple(self, condition: str, precip: float, visibility: float) -> float:
        sev = 0.0
        cond_map = {"clear": 0.0, "cloudy": 0.05, "rain": 0.2, "fog": 0.3, "storm": 0.5, "snow": 0.4}
        sev += cond_map.get(condition, 0.0)
        if precip > 5:
            sev += min(0.3, precip / 30.0 * 0.3)
        if visibility < 2:
            sev += min(0.3, (2 - visibility) / 2.0 * 0.3)
        return min(1.0, sev)


# =============================================================================
# MOTOR DE IMPACTO EN TRÁFICO
# =============================================================================

class WeatherTrafficEngine:
    """
    Calcula el impacto del clima en el tráfico y genera factores de ajuste
    para el agente de control de semáforos.
    """

    # Factores de impacto por condición
    IMPACT_TABLE = {
        "clear":  {"speed": 1.00, "braking": 1.00, "visibility": 1.00, "demand": 1.00},
        "cloudy": {"speed": 0.98, "braking": 1.00, "visibility": 0.95, "demand": 1.02},
        "rain":   {"speed": 0.82, "braking": 1.30, "visibility": 0.70, "demand": 0.90},
        "storm":  {"speed": 0.65, "braking": 1.50, "visibility": 0.40, "demand": 0.70},
        "fog":    {"speed": 0.70, "braking": 1.20, "visibility": 0.30, "demand": 0.85},
        "snow":   {"speed": 0.55, "braking": 1.60, "visibility": 0.50, "demand": 0.65},
    }

    def __init__(self):
        self.history: deque = deque(maxlen=288)  # 24h a 5min intervals

    def calculate_impact(self, weather: WeatherCondition) -> TrafficWeatherImpact:
        """Calcula el impacto del clima en el tráfico"""
        base = self.IMPACT_TABLE.get(weather.condition, self.IMPACT_TABLE["clear"])

        # Ajustar por severidad
        severity = weather.condition_severity
        speed_factor = base["speed"] - severity * 0.15
        braking_factor = base["braking"] + severity * 0.2
        visibility_factor = base["visibility"] - severity * 0.1
        demand_factor = base["demand"]

        # Ajustes adicionales por precipitación intensa
        if weather.precipitation > 20:
            speed_factor *= 0.85
            braking_factor *= 1.15

        # Ajuste por visibilidad muy baja
        if weather.visibility < 0.5:
            speed_factor *= 0.7
            visibility_factor = 0.1

        # Ajuste por viento fuerte
        if weather.wind_speed > 50:
            speed_factor *= 0.9
            demand_factor *= 0.85

        # Ajuste por temperatura extrema
        if weather.temperature > 38 or weather.temperature < -5:
            demand_factor *= 0.9

        # Clamp
        speed_factor = max(0.3, min(1.0, speed_factor))
        braking_factor = max(1.0, min(2.5, braking_factor))
        visibility_factor = max(0.05, min(1.0, visibility_factor))
        demand_factor = max(0.4, min(1.3, demand_factor))

        # Risk level
        if severity >= 0.7:
            risk_level = "extreme"
        elif severity >= 0.4:
            risk_level = "high"
        elif severity >= 0.15:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Recomendaciones
        recs = self._generate_recommendations(weather, speed_factor, risk_level)

        # Reward modifier
        reward_mod = self._calc_reward_modifier(weather, speed_factor)

        impact = TrafficWeatherImpact(
            speed_factor=round(speed_factor, 3),
            braking_factor=round(braking_factor, 3),
            visibility_factor=round(visibility_factor, 3),
            demand_factor=round(demand_factor, 3),
            risk_level=risk_level,
            recommendations=recs,
            reward_modifier=round(reward_mod, 3)
        )

        # Guardar en historial
        self.history.append({
            "timestamp": weather.timestamp,
            "condition": weather.condition,
            "severity": weather.condition_severity,
            "speed_factor": speed_factor,
            "risk_level": risk_level,
        })

        return impact

    def _generate_recommendations(self, weather: WeatherCondition,
                                  speed_factor: float, risk: str) -> List[str]:
        recs = []
        if speed_factor < 0.8:
            recs.append("Extender tiempos de verde para compensar velocidad reducida")
        if weather.visibility < 2:
            recs.append("Activar iluminacion extra en cruces")
        if weather.precipitation > 10:
            recs.append("Priorizar drenaje y ajustar ciclos para evitar acumulacion")
        if weather.wind_speed > 40:
            recs.append("Monitorear vehiculos altos y ciclistas")
        if risk == "extreme":
            recs.append("ALERTA: Considerar modo de emergencia")
        if weather.condition == "fog":
            recs.append("Reducir velocidad maxima recomendada en corredor")
        if weather.temperature < -2:
            recs.append("Posible hielo en calzada — extender fases peatonales")
        if not recs:
            recs.append("Condiciones normales — operacion estandar")
        return recs

    def _calc_reward_modifier(self, weather: WeatherCondition,
                              speed_factor: float) -> float:
        """
        Modificador de recompensa: en mal clima, el agente recibe bonificación
        por mantener flujo y penalización reducida por colas (es esperable).
        """
        if weather.condition_severity < 0.1:
            return 0.0  # Sin modificación en buen clima

        # En mal clima: bonificar throughput, reducir penalización de colas
        modifier = weather.condition_severity * 0.15
        return min(0.3, modifier)

    def get_state_augmentation(self, weather: WeatherCondition) -> List[float]:
        """
        Genera features adicionales para aumentar el vector de estado.
        Se pueden concatenar al estado 26D para tener 30D.
        """
        return [
            weather.condition_severity,
            weather.precipitation / 50.0,
            weather.visibility / 10.0,
            1.0 if weather.condition in ["rain", "storm", "snow"] else 0.0,
        ]


# =============================================================================
# INTEGRADOR CON ATLAS
# =============================================================================

class WeatherIntegration:
    """
    Clase de alto nivel que integra clima con el sistema ATLAS.
    Uso: instanciar y llamar a get_adjustment() cada ciclo.
    """

    def __init__(self, api_key: str = None, lat: float = 40.4168,
                 lon: float = -3.7038):
        self.provider = WeatherProvider(api_key, lat, lon)
        self.engine = WeatherTrafficEngine()
        self.last_weather: Optional[WeatherCondition] = None
        self.last_impact: Optional[TrafficWeatherImpact] = None

    def get_adjustment(self) -> Dict:
        """
        Obtiene ajustes de clima para el ciclo actual.
        Llamar cada N segundos (recomendado: 30-60s).
        """
        weather = self.provider.get_current()
        impact = self.engine.calculate_impact(weather)
        self.last_weather = weather
        self.last_impact = impact

        return {
            "weather": weather.to_dict(),
            "impact": {
                "speed_factor": impact.speed_factor,
                "braking_factor": impact.braking_factor,
                "visibility_factor": impact.visibility_factor,
                "demand_factor": impact.demand_factor,
                "risk_level": impact.risk_level,
                "recommendations": impact.recommendations,
                "reward_modifier": impact.reward_modifier,
            },
            "state_features": self.engine.get_state_augmentation(weather),
        }

    def modify_reward(self, base_reward: float) -> float:
        """Modifica la recompensa del agente basándose en el clima"""
        if self.last_impact is None:
            return base_reward
        return base_reward * (1.0 + self.last_impact.reward_modifier)

    def summary(self) -> str:
        if not self.last_weather:
            return "Sin datos meteorologicos"
        w = self.last_weather
        i = self.last_impact
        return (
            f"Clima: {w.condition} ({w.temperature}C, "
            f"precip={w.precipitation}mm/h, vis={w.visibility}km) | "
            f"Riesgo: {i.risk_level} | Speed factor: {i.speed_factor}"
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS Weather Integration")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenWeatherMap API key")
    parser.add_argument("--lat", type=float, default=40.4168,
                       help="Latitud (default: Madrid)")
    parser.add_argument("--lon", type=float, default=-3.7038,
                       help="Longitud (default: Madrid)")
    parser.add_argument("--test", action="store_true",
                       help="Modo test con datos simulados")
    parser.add_argument("--continuous", action="store_true",
                       help="Monitoreo continuo cada 30s")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  ATLAS Pro — Integracion Meteorologica")
    print("=" * 60)

    integration = WeatherIntegration(
        api_key=args.api_key, lat=args.lat, lon=args.lon
    )

    if args.continuous:
        print(f"\n  Monitoreo continuo (Ctrl+C para salir)\n")
        try:
            while True:
                adj = integration.get_adjustment()
                w = adj["weather"]
                imp = adj["impact"]
                print(
                    f"  [{w['timestamp'][-8:]}] "
                    f"{w['condition']:>7s} | "
                    f"{w['temperature']:>5.1f}C | "
                    f"Precip: {w['precipitation']:>5.1f}mm/h | "
                    f"Vis: {w['visibility']:>4.1f}km | "
                    f"Riesgo: {imp['risk_level']:>8s} | "
                    f"Speed: {imp['speed_factor']:.2f}x"
                )
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n  Monitoreo detenido.")
    else:
        # Single query
        adj = integration.get_adjustment()
        w = adj["weather"]
        imp = adj["impact"]

        print(f"\n  Fuente: {w['source']}")
        print(f"  Condicion: {w['condition']} (severidad: {w['condition_severity']:.2f})")
        print(f"  Temperatura: {w['temperature']}C")
        print(f"  Humedad: {w['humidity']}%")
        print(f"  Viento: {w['wind_speed']} km/h")
        print(f"  Precipitacion: {w['precipitation']} mm/h ({w['precipitation_type']})")
        print(f"  Visibilidad: {w['visibility']} km")
        print(f"  Presion: {w['pressure']} hPa")

        print(f"\n  --- Impacto en Trafico ---")
        print(f"  Factor velocidad: {imp['speed_factor']}x")
        print(f"  Factor frenado: {imp['braking_factor']}x")
        print(f"  Factor visibilidad: {imp['visibility_factor']}x")
        print(f"  Factor demanda: {imp['demand_factor']}x")
        print(f"  Nivel de riesgo: {imp['risk_level']}")
        print(f"  Mod. recompensa: {imp['reward_modifier']:+.3f}")

        print(f"\n  Recomendaciones:")
        for rec in imp["recommendations"]:
            print(f"    - {rec}")

        print(f"\n  Features para estado (4D extra):")
        print(f"    {adj['state_features']}")

        # Test reward modification
        base_reward = 25.0
        modified = integration.modify_reward(base_reward)
        print(f"\n  Ejemplo reward: {base_reward} -> {modified:.2f} "
              f"({'+' if modified > base_reward else ''}{modified-base_reward:.2f})")

        print(f"\n  Resumen: {integration.summary()}")
        print()


if __name__ == "__main__":
    main()
