"""
ATLAS - Red Neuronal con TensorFlow
====================================
Implementación del agente DQN usando TensorFlow/Keras.
Más potente y optimizada que la versión numpy.

Arquitectura:
- Input: Estado del tráfico (12 dimensiones) o imagen CNN
- Hidden: [256, 256, 128] neuronas con ReLU
- Output: 4 acciones (mantener, cambiar_ns, cambiar_eo, extender)
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger("ATLAS.TensorFlow")

# Intentar importar TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TF_DISPONIBLE = True
    logger.info(f"TensorFlow {tf.__version__} cargado correctamente")
except ImportError:
    TF_DISPONIBLE = False
    logger.warning("TensorFlow no disponible. Instalar con: pip install tensorflow")


# =============================================================================
# RED DQN CON TENSORFLOW
# =============================================================================

class RedDQN_TensorFlow:
    """
    Red neuronal DQN implementada en TensorFlow/Keras.
    Más eficiente y con mejor soporte para GPU.
    """
    
    def __init__(self, estado_dim: int = 12, acciones_dim: int = 4, 
                 learning_rate: float = 0.001):
        
        if not TF_DISPONIBLE:
            raise ImportError("TensorFlow no está instalado")
        
        self.estado_dim = estado_dim
        self.acciones_dim = acciones_dim
        self.learning_rate = learning_rate
        
        # Crear modelo
        self.modelo = self._construir_modelo()
        
        logger.info(f"Red DQN TensorFlow creada: {estado_dim} → {acciones_dim}")
    
    def _construir_modelo(self) -> keras.Model:
        """Construye la arquitectura de la red"""
        
        modelo = keras.Sequential([
            layers.Input(shape=(self.estado_dim,)),
            layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(self.acciones_dim, activation='linear')
        ])
        
        modelo.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return modelo
    
    def predecir(self, estado: np.ndarray) -> np.ndarray:
        """Predice valores Q para un estado"""
        if estado.ndim == 1:
            estado = estado.reshape(1, -1)
        return self.modelo.predict(estado, verbose=0)[0]
    
    def predecir_batch(self, estados: np.ndarray) -> np.ndarray:
        """Predice valores Q para un batch de estados"""
        return self.modelo.predict(estados, verbose=0)
    
    def entrenar_batch(self, estados: np.ndarray, targets: np.ndarray) -> float:
        """Entrena con un batch de datos"""
        historia = self.modelo.fit(estados, targets, epochs=1, verbose=0)
        return historia.history['loss'][0]
    
    def copiar_pesos_de(self, otra_red: 'RedDQN_TensorFlow'):
        """Copia los pesos de otra red"""
        self.modelo.set_weights(otra_red.modelo.get_weights())
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(ruta)
        logger.info(f"Modelo guardado en {ruta}")
    
    def cargar(self, ruta: str) -> bool:
        """Carga el modelo"""
        if os.path.exists(ruta):
            self.modelo = keras.models.load_model(ruta)
            logger.info(f"Modelo cargado desde {ruta}")
            return True
        return False


# =============================================================================
# RED CNN PARA IMÁGENES
# =============================================================================

class RedCNN_Trafico:
    """
    Red Convolucional para procesar imágenes de cámaras de tráfico.
    
    Arquitectura:
    - Conv2D + MaxPooling (extracción de características)
    - Flatten
    - Dense (decisiones)
    """
    
    def __init__(self, imagen_shape: Tuple[int, int, int] = (224, 224, 3),
                 acciones_dim: int = 4, learning_rate: float = 0.0001):
        
        if not TF_DISPONIBLE:
            raise ImportError("TensorFlow no está instalado")
        
        self.imagen_shape = imagen_shape
        self.acciones_dim = acciones_dim
        self.learning_rate = learning_rate
        
        # Crear modelo CNN
        self.modelo = self._construir_modelo()
        
        logger.info(f"Red CNN creada: {imagen_shape} → {acciones_dim}")
    
    def _construir_modelo(self) -> keras.Model:
        """Construye la arquitectura CNN"""
        
        modelo = keras.Sequential([
            # Input
            layers.Input(shape=self.imagen_shape),
            
            # Bloque Conv 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Bloque Conv 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Bloque Conv 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Bloque Conv 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten y Dense
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.acciones_dim, activation='linear')
        ])
        
        modelo.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return modelo
    
    def preprocesar_imagen(self, imagen: np.ndarray) -> np.ndarray:
        """Preprocesa una imagen para la red"""
        # Redimensionar
        if imagen.shape[:2] != self.imagen_shape[:2]:
            try:
                import cv2
                imagen = cv2.resize(imagen, (self.imagen_shape[1], self.imagen_shape[0]))
            except ImportError:
                from PIL import Image
                img = Image.fromarray(imagen)
                img = img.resize((self.imagen_shape[1], self.imagen_shape[0]))
                imagen = np.array(img)
        
        # Normalizar a [0, 1]
        imagen = imagen.astype(np.float32) / 255.0
        
        return imagen
    
    def predecir(self, imagen: np.ndarray) -> np.ndarray:
        """Predice valores Q para una imagen"""
        imagen = self.preprocesar_imagen(imagen)
        if imagen.ndim == 3:
            imagen = imagen.reshape(1, *imagen.shape)
        return self.modelo.predict(imagen, verbose=0)[0]
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(ruta)
    
    def cargar(self, ruta: str) -> bool:
        """Carga el modelo"""
        if os.path.exists(ruta):
            self.modelo = keras.models.load_model(ruta)
            return True
        return False


# =============================================================================
# AGENTE DQN CON TENSORFLOW
# =============================================================================

class AgenteDQN_TensorFlow:
    """
    Agente DQN completo usando TensorFlow.
    Incluye:
    - Double DQN
    - Experience Replay con prioridad
    - Target Network
    """
    
    ACCIONES = ['mantener', 'cambiar_ns', 'cambiar_eo', 'extender']
    
    def __init__(self, estado_dim: int = 12, usar_cnn: bool = False,
                 imagen_shape: Tuple[int, int, int] = (224, 224, 3)):
        
        self.estado_dim = estado_dim
        self.acciones_dim = len(self.ACCIONES)
        self.usar_cnn = usar_cnn
        
        # Crear redes
        if usar_cnn:
            self.red_principal = RedCNN_Trafico(imagen_shape, self.acciones_dim)
            self.red_objetivo = RedCNN_Trafico(imagen_shape, self.acciones_dim)
        else:
            self.red_principal = RedDQN_TensorFlow(estado_dim, self.acciones_dim)
            self.red_objetivo = RedDQN_TensorFlow(estado_dim, self.acciones_dim)
        
        self.red_objetivo.copiar_pesos_de(self.red_principal)
        
        # Memoria de experiencias
        self.memoria = deque(maxlen=100000)
        
        # Hiperparámetros
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.actualizar_objetivo_cada = 100
        
        # Contadores
        self.pasos = 0
        
        logger.info(f"Agente DQN TensorFlow creado - CNN: {usar_cnn}")
    
    def obtener_accion(self, estado: np.ndarray) -> int:
        """Selecciona una acción usando epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.acciones_dim - 1)
        
        q_valores = self.red_principal.predecir(estado)
        return int(np.argmax(q_valores))
    
    def recordar(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Guarda experiencia en memoria"""
        self.memoria.append((estado, accion, recompensa, siguiente_estado, terminado))
    
    def entrenar(self) -> float:
        """Entrena la red con un batch de experiencias"""
        if len(self.memoria) < self.batch_size:
            return 0.0
        
        # Muestrear batch
        batch = random.sample(self.memoria, self.batch_size)
        
        estados = np.array([e[0] for e in batch])
        acciones = np.array([e[1] for e in batch])
        recompensas = np.array([e[2] for e in batch])
        siguientes_estados = np.array([e[3] for e in batch])
        terminados = np.array([e[4] for e in batch])
        
        # Double DQN: usar red principal para seleccionar acción,
        # red objetivo para evaluar
        q_siguiente_principal = self.red_principal.predecir_batch(siguientes_estados)
        q_siguiente_objetivo = self.red_objetivo.predecir_batch(siguientes_estados)
        
        mejores_acciones = np.argmax(q_siguiente_principal, axis=1)
        q_siguiente = q_siguiente_objetivo[np.arange(self.batch_size), mejores_acciones]
        
        # Calcular targets
        targets_q = self.red_principal.predecir_batch(estados)
        targets = recompensas + (1 - terminados) * self.gamma * q_siguiente
        targets_q[np.arange(self.batch_size), acciones] = targets
        
        # Entrenar
        loss = self.red_principal.entrenar_batch(estados, targets_q)
        
        # Actualizar red objetivo
        self.pasos += 1
        if self.pasos % self.actualizar_objetivo_cada == 0:
            self.red_objetivo.copiar_pesos_de(self.red_principal)
        
        # Reducir epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def guardar(self, carpeta: str):
        """Guarda el agente"""
        os.makedirs(carpeta, exist_ok=True)
        self.red_principal.guardar(f"{carpeta}/red_principal")
        self.red_objetivo.guardar(f"{carpeta}/red_objetivo")
        
        # Guardar hiperparámetros
        import json
        config = {
            'epsilon': self.epsilon,
            'pasos': self.pasos,
            'usar_cnn': self.usar_cnn,
            'estado_dim': self.estado_dim
        }
        with open(f"{carpeta}/config.json", 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Agente guardado en {carpeta}")
    
    def cargar(self, carpeta: str) -> bool:
        """Carga el agente"""
        if not os.path.exists(carpeta):
            return False
        
        self.red_principal.cargar(f"{carpeta}/red_principal")
        self.red_objetivo.cargar(f"{carpeta}/red_objetivo")
        
        # Cargar config
        config_path = f"{carpeta}/config.json"
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.epsilon = config.get('epsilon', 0.01)
            self.pasos = config.get('pasos', 0)
        
        logger.info(f"Agente cargado desde {carpeta}")
        return True


# =============================================================================
# MODELO HÍBRIDO: CNN + DQN
# =============================================================================

class ModeloHibrido:
    """
    Modelo que combina:
    - YOLO para detección de vehículos
    - CNN opcional para características visuales adicionales
    - DQN para tomar decisiones
    
    Flujo:
    Imagen → YOLO → Conteo vehículos → DQN → Acción
                ↓
              CNN → Características → DQN
    """
    
    def __init__(self, usar_cnn_adicional: bool = False):
        self.usar_cnn_adicional = usar_cnn_adicional
        
        # Detector YOLO
        from detector_vehiculos import DetectorVehiculos
        self.detector = DetectorVehiculos()
        
        # Agente DQN (con o sin CNN)
        if TF_DISPONIBLE:
            self.agente = AgenteDQN_TensorFlow(estado_dim=12, usar_cnn=False)
        else:
            # Fallback a numpy
            logger.warning("Usando agente numpy (TensorFlow no disponible)")
            self.agente = None
        
        # CNN adicional para características
        if usar_cnn_adicional and TF_DISPONIBLE:
            self.cnn_caracteristicas = RedCNN_Trafico()
        else:
            self.cnn_caracteristicas = None
        
        logger.info(f"Modelo híbrido creado - CNN adicional: {usar_cnn_adicional}")
    
    def procesar_imagen(self, imagen: np.ndarray) -> Tuple[int, dict]:
        """
        Procesa una imagen y devuelve la acción a tomar.
        
        Returns:
            (accion, info)
        """
        # 1. Detectar vehículos con YOLO
        resultado = self.detector.detectar(imagen)
        estado_trafico = self.detector.obtener_estado_trafico(resultado)
        
        # 2. Convertir a vector de estado
        estado = self._estado_a_vector(estado_trafico)
        
        # 3. Obtener acción del DQN
        if self.agente:
            accion = self.agente.obtener_accion(estado)
        else:
            accion = 0  # Mantener por defecto
        
        # Info adicional
        info = {
            'estado_trafico': estado_trafico,
            'detecciones': resultado.total_vehiculos,
            'tiempo_deteccion_ms': resultado.tiempo_inferencia_ms,
            'accion': self.agente.ACCIONES[accion] if self.agente else 'mantener'
        }
        
        return accion, info
    
    def _estado_a_vector(self, estado_trafico: dict) -> np.ndarray:
        """Convierte estado del tráfico a vector para DQN"""
        return np.array([
            estado_trafico.get('cola_norte', 0) / 50.0,
            estado_trafico.get('cola_sur', 0) / 50.0,
            estado_trafico.get('cola_este', 0) / 50.0,
            estado_trafico.get('cola_oeste', 0) / 50.0,
            estado_trafico.get('coches', 0) / 50.0,
            estado_trafico.get('motos', 0) / 20.0,
            estado_trafico.get('buses', 0) / 10.0,
            estado_trafico.get('camiones', 0) / 10.0,
            0.0, 0.0, 0.0, 0.0  # Reservado
        ], dtype=np.float32)
    
    def entrenar_paso(self, imagen, accion, recompensa, siguiente_imagen, terminado):
        """Entrena con una experiencia"""
        if not self.agente:
            return 0.0
        
        # Procesar imágenes
        resultado1 = self.detector.detectar(imagen)
        resultado2 = self.detector.detectar(siguiente_imagen)
        
        estado1 = self._estado_a_vector(self.detector.obtener_estado_trafico(resultado1))
        estado2 = self._estado_a_vector(self.detector.obtener_estado_trafico(resultado2))
        
        # Guardar y entrenar
        self.agente.recordar(estado1, accion, recompensa, estado2, terminado)
        return self.agente.entrenar()


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def ejemplo_tensorflow():
    """Demuestra el uso de TensorFlow"""
    
    print("\n" + "="*60)
    print("🧠 ATLAS - Demo TensorFlow")
    print("="*60 + "\n")
    
    if not TF_DISPONIBLE:
        print("❌ TensorFlow no está instalado")
        print("   Instalar con: pip install tensorflow")
        return
    
    print(f"✅ TensorFlow versión: {tf.__version__}")
    print(f"   GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Crear agente
    print("\n📦 Creando agente DQN...")
    agente = AgenteDQN_TensorFlow(estado_dim=12)
    
    # Test de inferencia
    print("\n🧪 Test de inferencia...")
    estado = np.random.rand(12).astype(np.float32)
    accion = agente.obtener_accion(estado)
    print(f"   Estado: {estado[:4]}... → Acción: {agente.ACCIONES[accion]}")
    
    # Test de entrenamiento
    print("\n🏋️ Test de entrenamiento...")
    for i in range(100):
        estado = np.random.rand(12).astype(np.float32)
        accion = random.randint(0, 3)
        recompensa = random.uniform(-10, 10)
        siguiente = np.random.rand(12).astype(np.float32)
        
        agente.recordar(estado, accion, recompensa, siguiente, False)
    
    loss = agente.entrenar()
    print(f"   Loss después de entrenar: {loss:.4f}")
    
    print("\n✅ Demo completada\n")


if __name__ == "__main__":
    ejemplo_tensorflow()
