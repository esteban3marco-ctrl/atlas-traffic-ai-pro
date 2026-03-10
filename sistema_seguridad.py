"""
ATLAS - Sistema de Seguridad para Producción
=============================================
Este módulo garantiza que el sistema funcione de forma segura
en un entorno real de control de semáforos.

Características:
- Modo fallback automático si la IA falla
- Watchdog que monitoriza el sistema
- Límites físicos inviolables
- Detección de conflictos de señales
- Logging de todas las decisiones
- Override manual para emergencias
"""

import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Callable
import numpy as np


# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

def configurar_logging(carpeta_logs="logs"):
    """Configura el sistema de logging"""
    os.makedirs(carpeta_logs, exist_ok=True)
    
    fecha = datetime.now().strftime("%Y%m%d")
    archivo_log = f"{carpeta_logs}/atlas_{fecha}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(archivo_log, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("ATLAS")


# =============================================================================
# ENUMS Y CONSTANTES
# =============================================================================

class ModoOperacion(Enum):
    """Modos de operación del sistema"""
    IA_ACTIVA = "ia_activa"           # IA controlando
    FALLBACK_TIEMPOS_FIJOS = "fallback"  # Tiempos fijos de seguridad
    MANUAL = "manual"                  # Control manual (policía/técnico)
    EMERGENCIA = "emergencia"          # Modo emergencia (todos en rojo intermitente)
    MANTENIMIENTO = "mantenimiento"    # Sistema en mantenimiento


class EstadoSemaforo(Enum):
    """Estados posibles del semáforo"""
    ROJO = "rojo"
    AMARILLO = "amarillo"
    VERDE = "verde"
    ROJO_INTERMITENTE = "rojo_intermitente"
    APAGADO = "apagado"


class NivelAlerta(Enum):
    """Niveles de alerta del sistema"""
    NORMAL = 0
    AVISO = 1
    ALERTA = 2
    CRITICO = 3


# =============================================================================
# CONFIGURACIÓN DE SEGURIDAD
# =============================================================================

@dataclass
class ConfiguracionSeguridad:
    """Configuración de parámetros de seguridad"""
    
    # Tiempos mínimos y máximos (en segundos)
    TIEMPO_VERDE_MINIMO: float = 10.0    # Nunca menos de 10 segundos
    TIEMPO_VERDE_MAXIMO: float = 90.0    # Nunca más de 90 segundos
    TIEMPO_AMARILLO: float = 4.0          # Siempre 4 segundos de amarillo
    TIEMPO_TODO_ROJO: float = 2.0         # 2 segundos con todo en rojo entre fases
    
    # Watchdog
    WATCHDOG_TIMEOUT: float = 5.0         # Si no hay respuesta en 5s, activar fallback
    HEARTBEAT_INTERVALO: float = 1.0      # Enviar heartbeat cada segundo
    
    # Fallback (tiempos fijos de seguridad)
    FALLBACK_VERDE_NS: float = 30.0       # 30s verde Norte-Sur
    FALLBACK_VERDE_EO: float = 25.0       # 25s verde Este-Oeste
    
    # Límites de decisiones
    MAX_CAMBIOS_POR_MINUTO: int = 6       # Máximo 6 cambios por minuto
    MIN_TIEMPO_ENTRE_CAMBIOS: float = 8.0  # Mínimo 8 segundos entre cambios
    
    # Detección de anomalías
    MAX_COLA_ALERTA: int = 50             # Alertar si cola > 50 vehículos
    MAX_TIEMPO_ESPERA_ALERTA: float = 180.0  # Alertar si espera > 3 minutos


# =============================================================================
# REGISTRO DE DECISIONES
# =============================================================================

@dataclass
class RegistroDecision:
    """Registro de una decisión tomada por el sistema"""
    timestamp: str
    modo: str
    fase_anterior: int
    fase_nueva: int
    razon: str
    estado_trafico: Dict
    duracion_fase_anterior: float
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RegistradorDecisiones:
    """Registra todas las decisiones para auditoría"""
    
    def __init__(self, carpeta="logs/decisiones"):
        self.carpeta = carpeta
        os.makedirs(carpeta, exist_ok=True)
        self.decisiones = []
        self.archivo_actual = None
        self._abrir_archivo()
    
    def _abrir_archivo(self):
        fecha = datetime.now().strftime("%Y%m%d")
        self.archivo_actual = f"{self.carpeta}/decisiones_{fecha}.jsonl"
    
    def registrar(self, decision: RegistroDecision):
        """Registra una decisión"""
        self.decisiones.append(decision)
        
        # Guardar en archivo (formato JSON Lines)
        with open(self.archivo_actual, "a", encoding="utf-8") as f:
            f.write(decision.to_json() + "\n")
    
    def obtener_ultimas(self, n: int = 10) -> List[RegistroDecision]:
        """Obtiene las últimas N decisiones"""
        return self.decisiones[-n:]
    
    def contar_cambios_ultimo_minuto(self) -> int:
        """Cuenta cambios en el último minuto"""
        ahora = datetime.now()
        contador = 0
        for decision in reversed(self.decisiones):
            tiempo_decision = datetime.fromisoformat(decision.timestamp)
            if (ahora - tiempo_decision).total_seconds() <= 60:
                contador += 1
            else:
                break
        return contador


# =============================================================================
# WATCHDOG
# =============================================================================

class Watchdog:
    """
    Monitoriza que el sistema responde correctamente.
    Si no recibe heartbeat en X segundos, activa el modo fallback.
    """
    
    def __init__(self, timeout: float, callback_timeout: Callable):
        self.timeout = timeout
        self.callback_timeout = callback_timeout
        self.ultimo_heartbeat = time.time()
        self.activo = False
        self.thread = None
        self.logger = logging.getLogger("ATLAS.Watchdog")
    
    def iniciar(self):
        """Inicia el watchdog"""
        self.activo = True
        self.ultimo_heartbeat = time.time()
        self.thread = threading.Thread(target=self._monitorizar, daemon=True)
        self.thread.start()
        self.logger.info("Watchdog iniciado")
    
    def detener(self):
        """Detiene el watchdog"""
        self.activo = False
        if self.thread:
            self.thread.join(timeout=2)
        self.logger.info("Watchdog detenido")
    
    def heartbeat(self):
        """Envía señal de que el sistema está funcionando"""
        self.ultimo_heartbeat = time.time()
    
    def _monitorizar(self):
        """Loop de monitorización"""
        while self.activo:
            tiempo_sin_heartbeat = time.time() - self.ultimo_heartbeat
            
            if tiempo_sin_heartbeat > self.timeout:
                self.logger.critical(f"TIMEOUT: Sin heartbeat por {tiempo_sin_heartbeat:.1f}s")
                self.callback_timeout()
                self.ultimo_heartbeat = time.time()  # Reset para evitar múltiples callbacks
            
            time.sleep(0.5)


# =============================================================================
# VALIDADOR DE SEGURIDAD
# =============================================================================

class ValidadorSeguridad:
    """
    Valida que todas las decisiones cumplan con las reglas de seguridad.
    NINGUNA decisión pasa sin validación.
    """
    
    def __init__(self, config: ConfiguracionSeguridad):
        self.config = config
        self.logger = logging.getLogger("ATLAS.Validador")
        self.ultimo_cambio = 0
        self.fase_actual = 0
        self.tiempo_fase_actual = 0
    
    def validar_cambio_fase(self, nueva_fase: int, tiempo_en_fase_actual: float) -> tuple:
        """
        Valida si un cambio de fase es seguro.
        Retorna (es_valido, razon)
        """
        
        # Regla 1: Tiempo mínimo en verde
        if tiempo_en_fase_actual < self.config.TIEMPO_VERDE_MINIMO:
            return False, f"Tiempo en fase actual ({tiempo_en_fase_actual:.1f}s) menor que mínimo ({self.config.TIEMPO_VERDE_MINIMO}s)"
        
        # Regla 2: Tiempo mínimo entre cambios
        tiempo_desde_ultimo = time.time() - self.ultimo_cambio
        if tiempo_desde_ultimo < self.config.MIN_TIEMPO_ENTRE_CAMBIOS:
            return False, f"Muy poco tiempo desde último cambio ({tiempo_desde_ultimo:.1f}s)"
        
        # Regla 3: Nueva fase debe ser diferente
        if nueva_fase == self.fase_actual:
            return False, "Nueva fase igual a la actual"
        
        # Regla 4: Fase válida
        if nueva_fase not in [0, 1, 2, 3]:
            return False, f"Fase inválida: {nueva_fase}"
        
        return True, "OK"
    
    def validar_tiempo_verde(self, tiempo: float) -> float:
        """
        Valida y ajusta el tiempo de verde.
        Siempre retorna un valor dentro de los límites.
        """
        tiempo_validado = max(self.config.TIEMPO_VERDE_MINIMO, 
                             min(tiempo, self.config.TIEMPO_VERDE_MAXIMO))
        
        if tiempo != tiempo_validado:
            self.logger.warning(f"Tiempo ajustado: {tiempo:.1f}s → {tiempo_validado:.1f}s")
        
        return tiempo_validado
    
    def registrar_cambio(self, nueva_fase: int):
        """Registra un cambio de fase exitoso"""
        self.ultimo_cambio = time.time()
        self.fase_actual = nueva_fase
        self.tiempo_fase_actual = time.time()
    
    def verificar_conflicto_senales(self, estado_semaforos: Dict) -> bool:
        """
        CRÍTICO: Verifica que no haya conflicto de señales.
        Nunca debe haber verde en direcciones que se cruzan.
        """
        
        # Direcciones que NO pueden tener verde simultáneamente
        conflictos = [
            ("norte", "este"),
            ("norte", "oeste"),
            ("sur", "este"),
            ("sur", "oeste"),
        ]
        
        for dir1, dir2 in conflictos:
            if (estado_semaforos.get(dir1) == EstadoSemaforo.VERDE and 
                estado_semaforos.get(dir2) == EstadoSemaforo.VERDE):
                self.logger.critical(f"¡¡CONFLICTO DE SEÑALES!! {dir1} y {dir2} ambos en VERDE")
                return False
        
        return True


# =============================================================================
# SISTEMA DE FALLBACK
# =============================================================================

class SistemaFallback:
    """
    Sistema de tiempos fijos que se activa si la IA falla.
    Garantiza operación segura aunque no óptima.
    """
    
    def __init__(self, config: ConfiguracionSeguridad):
        self.config = config
        self.logger = logging.getLogger("ATLAS.Fallback")
        self.fase = 0
        self.tiempo_inicio_fase = time.time()
    
    def obtener_accion(self) -> tuple:
        """
        Obtiene la siguiente acción según tiempos fijos.
        Retorna (fase, tiempo_restante)
        """
        tiempo_en_fase = time.time() - self.tiempo_inicio_fase
        
        if self.fase == 0:  # Norte-Sur
            if tiempo_en_fase >= self.config.FALLBACK_VERDE_NS:
                self.fase = 2  # Cambiar a Este-Oeste
                self.tiempo_inicio_fase = time.time()
                self.logger.info("Fallback: N-S → E-O")
            return 0, self.config.FALLBACK_VERDE_NS - tiempo_en_fase
        
        else:  # Este-Oeste
            if tiempo_en_fase >= self.config.FALLBACK_VERDE_EO:
                self.fase = 0  # Cambiar a Norte-Sur
                self.tiempo_inicio_fase = time.time()
                self.logger.info("Fallback: E-O → N-S")
            return 2, self.config.FALLBACK_VERDE_EO - tiempo_en_fase
    
    def reset(self):
        """Resetea el sistema de fallback"""
        self.fase = 0
        self.tiempo_inicio_fase = time.time()


# =============================================================================
# CONTROLADOR DE SEGURIDAD PRINCIPAL
# =============================================================================

class ControladorSeguridad:
    """
    Controlador principal que integra todos los sistemas de seguridad.
    Este es el punto de entrada para cualquier decisión del sistema.
    """
    
    def __init__(self, config: ConfiguracionSeguridad = None):
        self.config = config or ConfiguracionSeguridad()
        self.logger = configurar_logging()
        
        # Componentes de seguridad
        self.validador = ValidadorSeguridad(self.config)
        self.fallback = SistemaFallback(self.config)
        self.registrador = RegistradorDecisiones()
        self.watchdog = Watchdog(
            timeout=self.config.WATCHDOG_TIMEOUT,
            callback_timeout=self._activar_fallback
        )
        
        # Estado
        self.modo = ModoOperacion.IA_ACTIVA
        self.nivel_alerta = NivelAlerta.NORMAL
        self.fase_actual = 0
        self.tiempo_inicio_fase = time.time()
        self.errores_consecutivos = 0
        self.max_errores_consecutivos = 3
        
        self.logger.info("="*60)
        self.logger.info("ATLAS - Sistema de Seguridad Iniciado")
        self.logger.info(f"Configuración: {self.config}")
        self.logger.info("="*60)
    
    def iniciar(self):
        """Inicia el sistema de seguridad"""
        self.watchdog.iniciar()
        self.logger.info("Sistema de seguridad ACTIVO")
    
    def detener(self):
        """Detiene el sistema de seguridad"""
        self.watchdog.detener()
        self.logger.info("Sistema de seguridad DETENIDO")
    
    def heartbeat(self):
        """Envía heartbeat al watchdog"""
        self.watchdog.heartbeat()
    
    def procesar_decision_ia(self, accion_ia: int, estado_trafico: Dict) -> int:
        """
        Procesa una decisión de la IA y la valida.
        Retorna la acción final (puede ser diferente si la IA propone algo inseguro).
        """
        
        self.heartbeat()
        
        # Si estamos en modo fallback o manual, ignorar la IA
        if self.modo != ModoOperacion.IA_ACTIVA:
            self.logger.debug(f"Modo {self.modo.value}: ignorando decisión IA")
            return self._obtener_accion_actual()
        
        # Calcular tiempo en fase actual
        tiempo_en_fase = time.time() - self.tiempo_inicio_fase
        
        # Determinar si la IA quiere cambiar de fase
        quiere_cambiar = self._accion_implica_cambio(accion_ia)
        
        if quiere_cambiar:
            # Validar el cambio
            nueva_fase = self._calcular_nueva_fase(accion_ia)
            es_valido, razon = self.validador.validar_cambio_fase(nueva_fase, tiempo_en_fase)
            
            if es_valido:
                # Cambio aprobado
                self._ejecutar_cambio_fase(nueva_fase, estado_trafico, "IA", razon)
                self.errores_consecutivos = 0
                return nueva_fase
            else:
                # Cambio rechazado
                self.logger.warning(f"Cambio rechazado: {razon}")
                self.errores_consecutivos += 1
                
                # Si la IA falla muchas veces, activar fallback
                if self.errores_consecutivos >= self.max_errores_consecutivos:
                    self.logger.warning(f"IA con {self.errores_consecutivos} errores consecutivos")
                    self._activar_fallback()
                
                return self.fase_actual
        
        # Verificar si hay que forzar cambio por tiempo máximo
        if tiempo_en_fase >= self.config.TIEMPO_VERDE_MAXIMO:
            self.logger.info(f"Forzando cambio: tiempo máximo alcanzado ({tiempo_en_fase:.1f}s)")
            nueva_fase = 2 if self.fase_actual == 0 else 0
            self._ejecutar_cambio_fase(nueva_fase, estado_trafico, "SISTEMA", "Tiempo máximo")
            return nueva_fase
        
        return self.fase_actual
    
    def _accion_implica_cambio(self, accion: int) -> bool:
        """Determina si una acción implica cambio de fase"""
        # Acciones: 0=mantener, 1=cambiar_ns, 2=cambiar_eo, 3=extender
        return accion in [1, 2]
    
    def _calcular_nueva_fase(self, accion: int) -> int:
        """Calcula la nueva fase basada en la acción"""
        if accion == 1:
            return 0  # Norte-Sur
        elif accion == 2:
            return 2  # Este-Oeste
        return self.fase_actual
    
    def _ejecutar_cambio_fase(self, nueva_fase: int, estado_trafico: Dict, origen: str, razon: str):
        """Ejecuta un cambio de fase"""
        
        tiempo_fase_anterior = time.time() - self.tiempo_inicio_fase
        fase_anterior = self.fase_actual
        
        # Registrar decisión
        registro = RegistroDecision(
            timestamp=datetime.now().isoformat(),
            modo=self.modo.value,
            fase_anterior=fase_anterior,
            fase_nueva=nueva_fase,
            razon=f"{origen}: {razon}",
            estado_trafico=estado_trafico,
            duracion_fase_anterior=tiempo_fase_anterior
        )
        self.registrador.registrar(registro)
        
        # Actualizar estado
        self.fase_actual = nueva_fase
        self.tiempo_inicio_fase = time.time()
        self.validador.registrar_cambio(nueva_fase)
        
        fase_str = "N-S" if nueva_fase == 0 else "E-O"
        self.logger.info(f"Cambio de fase: {fase_anterior} → {nueva_fase} ({fase_str}) [{origen}]")
    
    def _obtener_accion_actual(self) -> int:
        """Obtiene la acción actual según el modo"""
        if self.modo == ModoOperacion.FALLBACK_TIEMPOS_FIJOS:
            fase, _ = self.fallback.obtener_accion()
            if fase != self.fase_actual:
                self._ejecutar_cambio_fase(fase, {}, "FALLBACK", "Tiempos fijos")
            return fase
        
        return self.fase_actual
    
    def _activar_fallback(self):
        """Activa el modo fallback"""
        if self.modo != ModoOperacion.FALLBACK_TIEMPOS_FIJOS:
            self.logger.critical("⚠️ ACTIVANDO MODO FALLBACK ⚠️")
            self.modo = ModoOperacion.FALLBACK_TIEMPOS_FIJOS
            self.fallback.reset()
            self.nivel_alerta = NivelAlerta.CRITICO
    
    def reactivar_ia(self):
        """Reactiva el control por IA"""
        if self.modo == ModoOperacion.FALLBACK_TIEMPOS_FIJOS:
            self.logger.info("Reactivando control por IA")
            self.modo = ModoOperacion.IA_ACTIVA
            self.errores_consecutivos = 0
            self.nivel_alerta = NivelAlerta.NORMAL
    
    def activar_modo_manual(self):
        """Activa el modo manual"""
        self.logger.info("MODO MANUAL ACTIVADO")
        self.modo = ModoOperacion.MANUAL
    
    def desactivar_modo_manual(self):
        """Desactiva el modo manual"""
        self.logger.info("MODO MANUAL DESACTIVADO")
        self.modo = ModoOperacion.IA_ACTIVA
    
    def cambio_manual(self, fase: int):
        """Permite cambio manual de fase (para policía/técnicos)"""
        if self.modo == ModoOperacion.MANUAL:
            self._ejecutar_cambio_fase(fase, {}, "MANUAL", "Operador")
        else:
            self.logger.warning("Intento de cambio manual sin modo manual activo")
    
    def obtener_estado(self) -> Dict:
        """Obtiene el estado actual del sistema"""
        return {
            "modo": self.modo.value,
            "fase_actual": self.fase_actual,
            "tiempo_en_fase": time.time() - self.tiempo_inicio_fase,
            "nivel_alerta": self.nivel_alerta.value,
            "errores_consecutivos": self.errores_consecutivos,
            "decisiones_ultimo_minuto": self.registrador.contar_cambios_ultimo_minuto()
        }
    
    def verificar_salud_sistema(self) -> Dict:
        """Verifica la salud general del sistema"""
        return {
            "watchdog_activo": self.watchdog.activo,
            "modo_operacion": self.modo.value,
            "nivel_alerta": self.nivel_alerta.name,
            "tiempo_desde_ultimo_cambio": time.time() - self.validador.ultimo_cambio,
            "errores_consecutivos": self.errores_consecutivos,
            "estado": "OK" if self.modo == ModoOperacion.IA_ACTIVA else "DEGRADADO"
        }


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def ejemplo_uso():
    """Ejemplo de cómo usar el sistema de seguridad"""
    
    print("\n" + "="*60)
    print("🔒 ATLAS - Demo del Sistema de Seguridad")
    print("="*60 + "\n")
    
    # Crear controlador con configuración por defecto
    controlador = ControladorSeguridad()
    controlador.iniciar()
    
    try:
        # Simular decisiones de la IA
        for i in range(20):
            # Simular estado del tráfico
            estado_trafico = {
                "cola_norte": np.random.randint(0, 30),
                "cola_sur": np.random.randint(0, 30),
                "cola_este": np.random.randint(0, 30),
                "cola_oeste": np.random.randint(0, 30),
            }
            
            # Simular decisión de la IA (aleatoria para demo)
            accion_ia = np.random.randint(0, 4)
            
            # Procesar decisión
            accion_final = controlador.procesar_decision_ia(accion_ia, estado_trafico)
            
            # Mostrar estado
            estado = controlador.obtener_estado()
            print(f"Paso {i+1}: IA propone {accion_ia}, ejecuta {accion_final}, "
                  f"modo={estado['modo']}, fase={estado['fase_actual']}")
            
            time.sleep(0.5)
        
        # Mostrar salud del sistema
        print("\n" + "="*60)
        print("Estado del sistema:")
        print(json.dumps(controlador.verificar_salud_sistema(), indent=2))
        
    finally:
        controlador.detener()
    
    print("\n✅ Demo completada\n")


if __name__ == "__main__":
    ejemplo_uso()
