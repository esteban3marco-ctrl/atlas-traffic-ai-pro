"""
ATLAS - Entrenamiento de Redes Neuronales Convolucionales
==========================================================
Sistema profesional de entrenamiento CNN para clasificación de tráfico.

Arquitectura CNN:
    Input (224x224x3)
        ↓
    Conv2D(32) → BatchNorm → ReLU → MaxPool
        ↓
    Conv2D(64) → BatchNorm → ReLU → MaxPool
        ↓
    Conv2D(128) → BatchNorm → ReLU → MaxPool
        ↓
    Conv2D(256) → BatchNorm → ReLU → MaxPool
        ↓
    GlobalAveragePooling
        ↓
    Dense(512) → Dropout(0.5)
        ↓
    Dense(256) → Dropout(0.3)
        ↓
    Output: 4 clases (bajo, medio, alto, muy_alto)

Uso:
    python entrenar_cnn.py --modo clasificador --epochs 50
    python entrenar_cnn.py --modo demo
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Verificar TensorFlow
print("Cargando TensorFlow...")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print(f"✅ TensorFlow {tf.__version__} cargado")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detectada: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("ℹ️  Usando CPU (sin GPU)")
        
except ImportError:
    print("❌ TensorFlow no instalado")
    print("   Ejecuta: pip install tensorflow")
    sys.exit(1)


# =============================================================================
# CONSTANTES
# =============================================================================

CLASES = ['bajo', 'medio', 'alto', 'muy_alto']
NUM_CLASES = len(CLASES)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32


# =============================================================================
# ARQUITECTURA CNN
# =============================================================================

def crear_modelo_cnn(input_shape=INPUT_SHAPE, num_clases=NUM_CLASES):
    """
    Crea el modelo CNN para clasificación de tráfico.
    
    Arquitectura inspirada en VGG pero más ligera.
    """
    
    modelo = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Bloque 1: 32 filtros
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 2: 64 filtros
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 3: 128 filtros
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 4: 256 filtros
        layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Fully Connected
        layers.Dense(512, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_clases, activation='softmax')
    ], name='ATLAS_CNN_Clasificador')
    
    return modelo


def crear_modelo_cnn_ligero(input_shape=INPUT_SHAPE, num_clases=NUM_CLASES):
    """
    Versión más ligera para entrenamiento rápido.
    """
    
    modelo = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ], name='ATLAS_CNN_Ligero')
    
    return modelo


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_dataset(carpeta_dataset='dataset', tamano=(224, 224)):
    """
    Carga el dataset de imágenes.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n📂 Cargando dataset...")
    
    imagenes = []
    etiquetas = []
    
    for i, clase in enumerate(CLASES):
        carpeta_clase = f"{carpeta_dataset}/{clase}"
        
        if not os.path.exists(carpeta_clase):
            print(f"   ⚠️  Carpeta no encontrada: {carpeta_clase}")
            continue
        
        archivos = [f for f in os.listdir(carpeta_clase) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"   📁 {clase}: {len(archivos)} imágenes")
        
        for archivo in archivos:
            ruta = os.path.join(carpeta_clase, archivo)
            
            try:
                # Cargar imagen
                try:
                    import cv2
                    img = cv2.imread(ruta)
                    img = cv2.resize(img, tamano)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    from PIL import Image
                    img = Image.open(ruta).convert('RGB')
                    img = img.resize(tamano)
                    img = np.array(img)
                
                # Normalizar
                img = img.astype(np.float32) / 255.0
                
                imagenes.append(img)
                etiquetas.append(i)
                
            except Exception as e:
                print(f"   ⚠️  Error cargando {archivo}: {e}")
    
    if len(imagenes) == 0:
        print("   ❌ No se encontraron imágenes")
        return None, None, None, None
    
    X = np.array(imagenes)
    y = np.array(etiquetas)
    
    print(f"\n   ✅ Total: {len(X)} imágenes cargadas")
    
    # Dividir en train/test
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"   📊 Train: {len(X_train)} | Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_modelo(modelo, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=32, usar_augmentacion=True):
    """
    Entrena el modelo CNN.
    """
    print("\n🏋️ Iniciando entrenamiento...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Data augmentation: {'Sí' if usar_augmentacion else 'No'}")
    
    # Crear carpeta para modelos
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Compilar modelo
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    mis_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'modelos/cnn_trafico_mejor.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entrenamiento
    if usar_augmentacion:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        historia = modelo.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=mis_callbacks,
            verbose=1
        )
    else:
        historia = modelo.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=mis_callbacks,
            verbose=1
        )
    
    # Guardar modelo final
    modelo.save('modelos/cnn_trafico_final.keras')
    print("\n✅ Modelo guardado en modelos/cnn_trafico_final.keras")
    
    return historia


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de test.
    """
    print("\n📊 Evaluando modelo...")
    
    # Evaluar
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    # Predicciones
    y_pred = modelo.predict(X_test, verbose=0)
    y_pred_clases = np.argmax(y_pred, axis=1)
    
    # Matriz de confusión
    print("\n   Matriz de confusión:")
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_test, y_pred_clases)
    print(f"   {' ':10} ", end='')
    for c in CLASES:
        print(f"{c[:6]:>8}", end='')
    print()
    
    for i, row in enumerate(cm):
        print(f"   {CLASES[i][:10]:10} ", end='')
        for val in row:
            print(f"{val:>8}", end='')
        print()
    
    # Reporte de clasificación
    print("\n   Reporte detallado:")
    print(classification_report(y_test, y_pred_clases, target_names=CLASES))
    
    return accuracy


# =============================================================================
# DQN VISUAL
# =============================================================================

def crear_modelo_dqn_visual(input_shape=INPUT_SHAPE, num_acciones=4):
    """
    Crea modelo DQN que toma decisiones desde imágenes.
    
    Arquitectura similar a DQN de Atari.
    """
    
    modelo = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv layers
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dense(num_acciones, activation='linear')  # Q-values
    ], name='ATLAS_DQN_Visual')
    
    return modelo


class AgenteDQNVisual:
    """
    Agente DQN que aprende directamente de imágenes.
    """
    
    ACCIONES = ['mantener', 'cambiar_ns', 'cambiar_eo', 'extender']
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.num_acciones = len(self.ACCIONES)
        
        # Redes
        self.red_principal = crear_modelo_dqn_visual(input_shape, self.num_acciones)
        self.red_objetivo = crear_modelo_dqn_visual(input_shape, self.num_acciones)
        
        self.red_principal.compile(
            optimizer=optimizers.Adam(learning_rate=0.00025),
            loss='mse'
        )
        
        self.red_objetivo.set_weights(self.red_principal.get_weights())
        
        # Memoria
        from collections import deque
        self.memoria = deque(maxlen=50000)
        
        # Hiperparámetros
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.batch_size = 32
        self.actualizar_cada = 1000
        self.pasos = 0
        
        print("✅ Agente DQN Visual creado")
    
    def preprocesar(self, imagen):
        """Preprocesa imagen"""
        if imagen.shape[:2] != self.input_shape[:2]:
            try:
                import cv2
                imagen = cv2.resize(imagen, (self.input_shape[1], self.input_shape[0]))
            except:
                from PIL import Image
                img = Image.fromarray(imagen)
                img = img.resize((self.input_shape[1], self.input_shape[0]))
                imagen = np.array(img)
        
        return imagen.astype(np.float32) / 255.0
    
    def obtener_accion(self, imagen):
        """Selecciona acción con epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acciones)
        
        imagen = self.preprocesar(imagen)
        imagen = np.expand_dims(imagen, 0)
        q_valores = self.red_principal.predict(imagen, verbose=0)[0]
        return int(np.argmax(q_valores))
    
    def recordar(self, imagen, accion, recompensa, siguiente, terminado):
        """Guarda experiencia"""
        imagen = self.preprocesar(imagen)
        siguiente = self.preprocesar(siguiente)
        self.memoria.append((imagen, accion, recompensa, siguiente, terminado))
    
    def entrenar(self):
        """Entrena con batch de experiencias"""
        if len(self.memoria) < self.batch_size:
            return 0.0
        
        import random
        batch = random.sample(self.memoria, self.batch_size)
        
        imgs = np.array([e[0] for e in batch])
        acciones = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        sigs = np.array([e[3] for e in batch])
        terms = np.array([e[4] for e in batch])
        
        q_sig = self.red_objetivo.predict(sigs, verbose=0)
        q_max = np.max(q_sig, axis=1)
        targets = rewards + (1 - terms) * self.gamma * q_max
        
        q_actual = self.red_principal.predict(imgs, verbose=0)
        q_actual[np.arange(self.batch_size), acciones] = targets
        
        loss = self.red_principal.train_on_batch(imgs, q_actual)
        
        self.pasos += 1
        if self.pasos % self.actualizar_cada == 0:
            self.red_objetivo.set_weights(self.red_principal.get_weights())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def guardar(self, carpeta='modelos/dqn_visual'):
        """Guarda el agente"""
        os.makedirs(carpeta, exist_ok=True)
        self.red_principal.save(f"{carpeta}/red_principal.keras")
        self.red_objetivo.save(f"{carpeta}/red_objetivo.keras")
        
        config = {'epsilon': self.epsilon, 'pasos': self.pasos}
        with open(f"{carpeta}/config.json", 'w') as f:
            json.dump(config, f)
        
        print(f"✅ Agente guardado en {carpeta}")
    
    def cargar(self, carpeta='modelos/dqn_visual'):
        """Carga el agente"""
        if os.path.exists(f"{carpeta}/red_principal.keras"):
            self.red_principal = keras.models.load_model(f"{carpeta}/red_principal.keras")
            self.red_objetivo = keras.models.load_model(f"{carpeta}/red_objetivo.keras")
            
            with open(f"{carpeta}/config.json", 'r') as f:
                config = json.load(f)
            self.epsilon = config.get('epsilon', 0.1)
            self.pasos = config.get('pasos', 0)
            
            print(f"✅ Agente cargado desde {carpeta}")
            return True
        return False


# =============================================================================
# MODO DEMO
# =============================================================================

def modo_demo():
    """Muestra las arquitecturas y verifica TensorFlow"""
    
    print("\n" + "="*60)
    print("🧠 ATLAS - Demo de Redes Neuronales CNN")
    print("="*60)
    
    print(f"\n✅ TensorFlow {tf.__version__}")
    
    # Modelo clasificador
    print("\n📦 Modelo CNN Clasificador:")
    modelo1 = crear_modelo_cnn()
    modelo1.summary()
    print(f"   Total parámetros: {modelo1.count_params():,}")
    
    # Modelo ligero
    print("\n📦 Modelo CNN Ligero:")
    modelo2 = crear_modelo_cnn_ligero()
    print(f"   Total parámetros: {modelo2.count_params():,}")
    
    # Modelo DQN
    print("\n📦 Modelo DQN Visual:")
    modelo3 = crear_modelo_dqn_visual()
    print(f"   Total parámetros: {modelo3.count_params():,}")
    
    # Test de inferencia
    print("\n🧪 Test de inferencia...")
    imagen_test = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    pred = modelo1.predict(imagen_test, verbose=0)
    print(f"   Clasificador: {pred[0]} → Clase: {CLASES[np.argmax(pred)]}")
    
    pred_dqn = modelo3.predict(imagen_test, verbose=0)
    print(f"   DQN Q-values: {pred_dqn[0]}")
    
    print("\n✅ Demo completada correctamente")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ATLAS - Entrenamiento CNN')
    parser.add_argument('--modo', choices=['demo', 'clasificador', 'dqn', 'evaluar'], 
                       default='demo', help='Modo de operación')
    parser.add_argument('--epochs', type=int, default=50, help='Número de epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', type=str, default='dataset', help='Carpeta del dataset')
    parser.add_argument('--ligero', action='store_true', help='Usar modelo ligero')
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*12 + "🚦 ATLAS - Entrenamiento CNN" + " "*12 + "║")
    print("╚" + "═"*58 + "╝")
    
    if args.modo == 'demo':
        modo_demo()
        
    elif args.modo == 'clasificador':
        # Cargar datos
        X_train, X_test, y_train, y_test = cargar_dataset(args.dataset)
        
        if X_train is None:
            print("\n❌ No se pudo cargar el dataset")
            print("   Ejecuta primero: python setup_atlas.py")
            return
        
        # Crear modelo
        if args.ligero:
            modelo = crear_modelo_cnn_ligero()
        else:
            modelo = crear_modelo_cnn()
        
        modelo.summary()
        
        # Entrenar
        historia = entrenar_modelo(
            modelo, X_train, y_train, X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch
        )
        
        # Evaluar
        evaluar_modelo(modelo, X_test, y_test)
        
    elif args.modo == 'evaluar':
        # Cargar modelo y evaluar
        if not os.path.exists('modelos/cnn_trafico_mejor.keras'):
            print("❌ No hay modelo entrenado")
            print("   Ejecuta: python entrenar_cnn.py --modo clasificador")
            return
        
        modelo = keras.models.load_model('modelos/cnn_trafico_mejor.keras')
        X_train, X_test, y_train, y_test = cargar_dataset(args.dataset)
        
        if X_test is not None:
            evaluar_modelo(modelo, X_test, y_test)
    
    elif args.modo == 'dqn':
        print("\n🎮 Entrenamiento DQN Visual")
        print("   Este modo requiere simulador SUMO activo")
        print("   Próximamente disponible")


if __name__ == "__main__":
    main()
