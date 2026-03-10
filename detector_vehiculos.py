"""
ATLAS - Detector de Vehículos con YOLOv8
=========================================
Detecta vehículos en imágenes de cámaras de tráfico.

Clases detectadas:
- Coches
- Motos
- Buses
- Camiones
- Bicicletas
- Peatones (para cruces peatonales)

Uso:
    detector = DetectorVehiculos()
    resultados = detector.detectar(imagen)
"""

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ATLAS.Detector")


@dataclass
class VehiculoDetectado:
    """Información de un vehículo detectado"""
    clase: str                    # Tipo: coche, moto, bus, camion, bici, peaton
    confianza: float              # 0.0 a 1.0
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centro: Tuple[int, int]       # Centro del vehículo
    area: int                     # Área en píxeles
    carril: Optional[int] = None  # Carril detectado (si se configura)
    direccion: Optional[str] = None  # N, S, E, O (si se configura)
    velocidad_estimada: Optional[float] = None  # km/h (si hay tracking)


@dataclass 
class ResultadoDeteccion:
    """Resultado completo de una detección"""
    total_vehiculos: int
    por_clase: Dict[str, int]
    por_direccion: Dict[str, int]
    vehiculos: List[VehiculoDetectado]
    tiempo_inferencia_ms: float
    imagen_procesada: Optional[np.ndarray] = None


class DetectorVehiculos:
    """
    Detector de vehículos usando YOLOv8.
    Optimizado para cámaras de tráfico.
    """
    
    # Mapeo de clases COCO a nuestras clases
    CLASES_INTERES = {
        0: 'peaton',      # person
        1: 'bicicleta',   # bicycle
        2: 'coche',       # car
        3: 'moto',        # motorcycle
        5: 'bus',         # bus
        7: 'camion',      # truck
    }
    
    # Colores para visualización (BGR)
    COLORES = {
        'coche': (0, 255, 255),      # Amarillo
        'moto': (255, 0, 0),         # Azul
        'bus': (0, 255, 0),          # Verde
        'camion': (128, 128, 128),   # Gris
        'bicicleta': (255, 255, 0),  # Cyan
        'peaton': (0, 0, 255),       # Rojo
    }
    
    def __init__(self, modelo: str = "yolov8n.pt", confianza_minima: float = 0.5):
        """
        Inicializa el detector.
        
        Args:
            modelo: Modelo YOLO a usar (yolov8n.pt, yolov8s.pt, yolov8m.pt)
                   n=nano (rápido), s=small, m=medium (más preciso)
            confianza_minima: Umbral de confianza (0.0-1.0)
        """
        self.modelo_nombre = modelo
        self.confianza_minima = confianza_minima
        self.modelo = None
        self.usando_simulacion = False
        
        # Intentar cargar YOLO
        self._cargar_modelo()
        
        # Configuración de zonas de detección
        self.zonas_direccion = {}  # Se configura según la cámara
        
        logger.info(f"Detector inicializado - Modelo: {modelo}, Confianza: {confianza_minima}")
    
    def _cargar_modelo(self):
        """Carga el modelo YOLO"""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Cargando modelo YOLO: {self.modelo_nombre}")
            self.modelo = YOLO(self.modelo_nombre)
            logger.info("✅ Modelo YOLO cargado correctamente")
            
        except ImportError:
            logger.warning("⚠️ ultralytics no instalado. Usando modo simulación.")
            logger.warning("   Instalar con: pip install ultralytics")
            self.usando_simulacion = True
            
        except Exception as e:
            logger.warning(f"⚠️ Error cargando YOLO: {e}. Usando modo simulación.")
            self.usando_simulacion = True
    
    def detectar(self, imagen: np.ndarray) -> ResultadoDeteccion:
        """
        Detecta vehículos en una imagen.
        
        Args:
            imagen: Imagen en formato numpy (BGR o RGB)
            
        Returns:
            ResultadoDeteccion con todos los vehículos encontrados
        """
        import time
        inicio = time.time()
        
        if self.usando_simulacion:
            resultado = self._detectar_simulado(imagen)
        else:
            resultado = self._detectar_yolo(imagen)
        
        resultado.tiempo_inferencia_ms = (time.time() - inicio) * 1000
        
        return resultado
    
    def _detectar_yolo(self, imagen: np.ndarray) -> ResultadoDeteccion:
        """Detección real con YOLO"""
        
        # Ejecutar inferencia
        resultados = self.modelo(imagen, conf=self.confianza_minima, verbose=False)
        
        vehiculos = []
        por_clase = {clase: 0 for clase in self.COLORES.keys()}
        por_direccion = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        
        for r in resultados:
            boxes = r.boxes
            
            for box in boxes:
                clase_id = int(box.cls[0])
                
                # Solo clases de interés
                if clase_id not in self.CLASES_INTERES:
                    continue
                
                clase = self.CLASES_INTERES[clase_id]
                confianza = float(box.conf[0])
                
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                
                # Determinar dirección basada en posición
                direccion = self._determinar_direccion(centro, imagen.shape)
                
                vehiculo = VehiculoDetectado(
                    clase=clase,
                    confianza=confianza,
                    bbox=(x1, y1, x2, y2),
                    centro=centro,
                    area=area,
                    direccion=direccion
                )
                
                vehiculos.append(vehiculo)
                por_clase[clase] += 1
                if direccion:
                    por_direccion[direccion] += 1
        
        return ResultadoDeteccion(
            total_vehiculos=len(vehiculos),
            por_clase=por_clase,
            por_direccion=por_direccion,
            vehiculos=vehiculos,
            tiempo_inferencia_ms=0  # Se actualiza después
        )
    
    def _detectar_simulado(self, imagen: np.ndarray) -> ResultadoDeteccion:
        """Detección simulada para testing sin YOLO"""
        
        # Simular detecciones basadas en el tamaño de la imagen
        altura, anchura = imagen.shape[:2]
        
        # Generar vehículos aleatorios
        num_vehiculos = np.random.randint(5, 20)
        
        vehiculos = []
        por_clase = {clase: 0 for clase in self.COLORES.keys()}
        por_direccion = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        
        clases = ['coche', 'coche', 'coche', 'coche', 'moto', 'bus', 'camion']
        
        for _ in range(num_vehiculos):
            clase = np.random.choice(clases)
            
            # Posición aleatoria
            x1 = np.random.randint(0, anchura - 100)
            y1 = np.random.randint(0, altura - 80)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(40, 80)
            
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)
            direccion = self._determinar_direccion(centro, imagen.shape)
            
            vehiculo = VehiculoDetectado(
                clase=clase,
                confianza=np.random.uniform(0.7, 0.99),
                bbox=(x1, y1, x2, y2),
                centro=centro,
                area=(x2-x1) * (y2-y1),
                direccion=direccion
            )
            
            vehiculos.append(vehiculo)
            por_clase[clase] += 1
            if direccion:
                por_direccion[direccion] += 1
        
        return ResultadoDeteccion(
            total_vehiculos=len(vehiculos),
            por_clase=por_clase,
            por_direccion=por_direccion,
            vehiculos=vehiculos,
            tiempo_inferencia_ms=0
        )
    
    def _determinar_direccion(self, centro: Tuple[int, int], shape: tuple) -> str:
        """Determina la dirección basada en la posición en la imagen"""
        altura, anchura = shape[:2]
        x, y = centro
        
        # Dividir imagen en 4 cuadrantes
        mitad_x = anchura // 2
        mitad_y = altura // 2
        
        if y < mitad_y:
            if x < mitad_x:
                return 'N'  # Cuadrante superior izquierdo
            else:
                return 'E'  # Cuadrante superior derecho
        else:
            if x < mitad_x:
                return 'O'  # Cuadrante inferior izquierdo
            else:
                return 'S'  # Cuadrante inferior derecho
    
    def configurar_zonas(self, zonas: Dict[str, List[Tuple[int, int]]]):
        """
        Configura zonas de detección personalizadas.
        
        Args:
            zonas: Diccionario con polígonos para cada dirección
                   {'N': [(x1,y1), (x2,y2), ...], 'S': [...], ...}
        """
        self.zonas_direccion = zonas
        logger.info(f"Zonas configuradas: {list(zonas.keys())}")
    
    def dibujar_detecciones(self, imagen: np.ndarray, resultado: ResultadoDeteccion) -> np.ndarray:
        """
        Dibuja las detecciones sobre la imagen.
        
        Args:
            imagen: Imagen original
            resultado: Resultado de la detección
            
        Returns:
            Imagen con las detecciones dibujadas
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV no instalado. No se puede dibujar.")
            return imagen
        
        imagen_out = imagen.copy()
        
        for v in resultado.vehiculos:
            color = self.COLORES.get(v.clase, (255, 255, 255))
            x1, y1, x2, y2 = v.bbox
            
            # Dibujar rectángulo
            cv2.rectangle(imagen_out, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta
            etiqueta = f"{v.clase} {v.confianza:.2f}"
            cv2.putText(imagen_out, etiqueta, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info general
        info = f"Total: {resultado.total_vehiculos} | Tiempo: {resultado.tiempo_inferencia_ms:.1f}ms"
        cv2.putText(imagen_out, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return imagen_out
    
    def obtener_estado_trafico(self, resultado: ResultadoDeteccion) -> Dict:
        """
        Convierte el resultado de detección al formato esperado por el DQN.
        
        Returns:
            Diccionario con colas por dirección y otros datos
        """
        return {
            'cola_norte': resultado.por_direccion.get('N', 0),
            'cola_sur': resultado.por_direccion.get('S', 0),
            'cola_este': resultado.por_direccion.get('E', 0),
            'cola_oeste': resultado.por_direccion.get('O', 0),
            'total_vehiculos': resultado.total_vehiculos,
            'coches': resultado.por_clase.get('coche', 0),
            'motos': resultado.por_clase.get('moto', 0),
            'buses': resultado.por_clase.get('bus', 0),
            'camiones': resultado.por_clase.get('camion', 0),
            'tiempo_inferencia_ms': resultado.tiempo_inferencia_ms
        }


# =============================================================================
# TRACKER DE VEHÍCULOS (para contar y estimar velocidad)
# =============================================================================

class TrackerVehiculos:
    """
    Rastrea vehículos entre frames para:
    - Contar vehículos que pasan
    - Estimar velocidad
    - Evitar contar el mismo vehículo dos veces
    """
    
    def __init__(self, distancia_maxima: int = 50):
        self.distancia_maxima = distancia_maxima
        self.vehiculos_activos = {}  # id -> (centro, clase, frames_visto)
        self.siguiente_id = 0
        self.vehiculos_contados = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        self.lineas_conteo = {}  # Líneas virtuales para contar
    
    def actualizar(self, detecciones: List[VehiculoDetectado]) -> Dict[int, VehiculoDetectado]:
        """
        Actualiza el tracking con nuevas detecciones.
        
        Returns:
            Diccionario id -> vehiculo con IDs asignados
        """
        vehiculos_frame = {}
        centros_nuevos = [(v.centro, v) for v in detecciones]
        
        # Asociar detecciones con vehículos existentes
        usados = set()
        
        for vid, (centro_ant, clase, frames) in list(self.vehiculos_activos.items()):
            mejor_dist = float('inf')
            mejor_idx = -1
            
            for i, (centro_nuevo, v) in enumerate(centros_nuevos):
                if i in usados:
                    continue
                    
                dist = np.sqrt((centro_ant[0] - centro_nuevo[0])**2 + 
                              (centro_ant[1] - centro_nuevo[1])**2)
                
                if dist < mejor_dist and dist < self.distancia_maxima:
                    mejor_dist = dist
                    mejor_idx = i
            
            if mejor_idx >= 0:
                # Asociar
                usados.add(mejor_idx)
                centro_nuevo, v = centros_nuevos[mejor_idx]
                self.vehiculos_activos[vid] = (centro_nuevo, v.clase, frames + 1)
                vehiculos_frame[vid] = v
            else:
                # Vehículo perdido
                del self.vehiculos_activos[vid]
        
        # Nuevos vehículos
        for i, (centro, v) in enumerate(centros_nuevos):
            if i not in usados:
                self.vehiculos_activos[self.siguiente_id] = (centro, v.clase, 1)
                vehiculos_frame[self.siguiente_id] = v
                self.siguiente_id += 1
        
        return vehiculos_frame
    
    def obtener_conteo(self) -> Dict[str, int]:
        """Obtiene el conteo actual por dirección"""
        return self.vehiculos_contados.copy()
    
    def resetear_conteo(self):
        """Resetea los contadores"""
        self.vehiculos_contados = {'N': 0, 'S': 0, 'E': 0, 'O': 0}


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def ejemplo_uso():
    """Demuestra cómo usar el detector"""
    
    print("\n" + "="*60)
    print("🚗 ATLAS - Demo del Detector de Vehículos")
    print("="*60 + "\n")
    
    # Crear detector
    detector = DetectorVehiculos(modelo="yolov8n.pt", confianza_minima=0.5)
    
    # Crear imagen de prueba (en producción vendría de la cámara)
    imagen_prueba = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detectar
    resultado = detector.detectar(imagen_prueba)
    
    # Mostrar resultados
    print(f"📊 Resultados de detección:")
    print(f"   Total vehículos: {resultado.total_vehiculos}")
    print(f"   Por clase: {resultado.por_clase}")
    print(f"   Por dirección: {resultado.por_direccion}")
    print(f"   Tiempo: {resultado.tiempo_inferencia_ms:.1f}ms")
    
    # Obtener estado para DQN
    estado = detector.obtener_estado_trafico(resultado)
    print(f"\n📡 Estado para DQN:")
    for k, v in estado.items():
        print(f"   {k}: {v}")
    
    print("\n✅ Demo completada\n")


if __name__ == "__main__":
    ejemplo_uso()
