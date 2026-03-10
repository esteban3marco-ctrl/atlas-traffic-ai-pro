# 🚦 ATLAS - GUÍA COMPLETA DE INSTALACIÓN Y USO

## Sistema de Control Inteligente de Semáforos con IA y Visión por Computador

---

## 📋 ÍNDICE

1. [Requisitos Previos](#1-requisitos-previos)
2. [Instalación](#2-instalación)
3. [Configuración Inicial](#3-configuración-inicial)
4. [Generación del Dataset](#4-generación-del-dataset)
5. [Entrenamiento de la CNN](#5-entrenamiento-de-la-cnn)
6. [Entrenamiento del DQN](#6-entrenamiento-del-dqn)
7. [Tests y Validación](#7-tests-y-validación)
8. [Uso del Sistema](#8-uso-del-sistema)
9. [Subir a GitHub](#9-subir-a-github)
10. [Solución de Problemas](#10-solución-de-problemas)

---

## 1. REQUISITOS PREVIOS

### Software necesario:

| Software | Versión | Descarga |
|----------|---------|----------|
| Python | 3.10+ | https://python.org |
| SUMO | 1.19+ | https://sumo.dlr.de/docs/Downloads.php |
| Git | Cualquiera | https://git-scm.com |

### Verificar instalación:

```powershell
python --version
sumo --version
git --version
```

---

## 2. INSTALACIÓN

### Paso 2.1: Crear carpeta del proyecto

```powershell
mkdir C:\ATLAS\atlas-traffic-ai
cd C:\ATLAS\atlas-traffic-ai
```

### Paso 2.2: Extraer archivos

Extrae el ZIP `atlas-traffic-ai-pro.zip` en `C:\ATLAS\atlas-traffic-ai\`

### Paso 2.3: Crear entorno virtual (recomendado)

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Paso 2.4: Instalar dependencias

```powershell
pip install --upgrade pip
pip install numpy tensorflow opencv-python Pillow traci sumolib scikit-learn matplotlib
```

### Paso 2.5: Verificar instalación

```powershell
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

---

## 3. CONFIGURACIÓN INICIAL

### Paso 3.1: Ejecutar setup automático

```powershell
python setup_atlas.py
```

Este comando:
- ✅ Verifica dependencias
- ✅ Crea escenarios SUMO
- ✅ Genera dataset de imágenes (1000+)

**Tiempo estimado: 5-10 minutos**

### Paso 3.2: Verificar estructura

```powershell
dir
```

Deberías ver:
```
atlas-traffic-ai/
├── dataset/
│   ├── bajo/
│   ├── medio/
│   ├── alto/
│   └── muy_alto/
├── simulations/
│   ├── simple/
│   ├── hora_punta/
│   ├── noche/
│   └── emergencias/
├── modelos/
├── logs/
├── setup_atlas.py
├── entrenar_cnn.py
├── ...
```

---

## 4. GENERACIÓN DEL DATASET

Si el setup automático no generó suficientes imágenes:

### Paso 4.1: Verificar imágenes existentes

```powershell
dir dataset\bajo
dir dataset\medio
dir dataset\alto
dir dataset\muy_alto
```

### Paso 4.2: Generar más imágenes (si necesario)

```powershell
python setup_atlas.py
```

### Dataset ideal:
- **Mínimo**: 500 imágenes (para pruebas)
- **Recomendado**: 1000+ imágenes
- **Óptimo**: 5000+ imágenes

---

## 5. ENTRENAMIENTO DE LA CNN

### Paso 5.1: Ver demo de arquitecturas

```powershell
python entrenar_cnn.py --modo demo
```

### Paso 5.2: Entrenar el clasificador CNN

```powershell
python entrenar_cnn.py --modo clasificador --epochs 50
```

**Tiempo estimado:**
- CPU: 30-60 minutos
- GPU: 5-15 minutos

### Paso 5.3: Entrenar versión rápida (opcional)

```powershell
python entrenar_cnn.py --modo clasificador --epochs 30 --ligero
```

### Paso 5.4: Ver resultados

Al finalizar verás:
```
📊 Evaluando modelo...
   Loss: 0.2345
   Accuracy: 92.50%
   
   Matriz de confusión:
              bajo   medio    alto muy_alt
   bajo         45       2       0       0
   medio         1      78       3       0
   alto          0       2      65       1
   muy_alto      0       0       2      51
```

### Paso 5.5: Modelo guardado en:

```
modelos/cnn_trafico_mejor.keras   (mejor validación)
modelos/cnn_trafico_final.keras   (último epoch)
```

---

## 6. ENTRENAMIENTO DEL DQN

### Paso 6.1: Copiar modelos anteriores (si los tienes)

```powershell
Copy-Item -Path "C:\ATLAS\atlas-produccion\modelos\*.npz" -Destination "modelos\"
```

### Paso 6.2: Entrenar DQN con estados numéricos

```powershell
python entrenar_avanzado.py --episodios 200 --cruce simple
```

### Paso 6.3: Entrenar más escenarios

```powershell
python entrenar_avanzado.py --episodios 100 --cruce simple_hora_punta
python entrenar_avanzado.py --episodios 100 --cruce simple_noche
```

---

## 7. TESTS Y VALIDACIÓN

### Paso 7.1: Ejecutar tests automáticos

```powershell
python tests_automaticos.py
```

**Resultado esperado:**
```
📊 RESUMEN DE TESTS
==================
  Total tests:    50
  ✅ Pasados:     50
  ❌ Fallidos:    0
```

### Paso 7.2: Evaluar CNN entrenada

```powershell
python entrenar_cnn.py --modo evaluar
```

### Paso 7.3: Probar sistema completo

```powershell
python main.py --modo demo
```

---

## 8. USO DEL SISTEMA

### 8.1: Modo demo (simulación)

```powershell
python main.py --modo demo
```

### 8.2: Clasificar una imagen

```python
from entrenar_cnn import cargar_modelo
import cv2

modelo = keras.models.load_model('modelos/cnn_trafico_mejor.keras')
imagen = cv2.imread('mi_imagen.jpg')
imagen = cv2.resize(imagen, (224, 224)) / 255.0
prediccion = modelo.predict(imagen[np.newaxis, ...])
clase = ['bajo', 'medio', 'alto', 'muy_alto'][np.argmax(prediccion)]
print(f"Nivel de tráfico: {clase}")
```

### 8.3: Usar el detector YOLO

```python
from detector_vehiculos import DetectorVehiculos

detector = DetectorVehiculos()
resultado = detector.detectar(imagen)
print(f"Vehículos detectados: {resultado.total_vehiculos}")
```

---

## 9. SUBIR A GITHUB

### Paso 9.1: Inicializar repositorio

```powershell
git init
git add .
git commit -m "Initial commit - ATLAS Traffic AI System"
```

### Paso 9.2: Crear repositorio en GitHub

1. Ve a https://github.com/new
2. Nombre: `atlas-traffic-ai`
3. Descripción: "Sistema de Control Inteligente de Semáforos con IA"
4. **NO** marcar "Initialize with README"
5. Crear repositorio

### Paso 9.3: Conectar y subir

```powershell
git remote add origin https://github.com/TU_USUARIO/atlas-traffic-ai.git
git branch -M main
git push -u origin main
```

### Paso 9.4: Archivos que NO se suben (por .gitignore)

- `modelos/*.keras` (muy grandes)
- `dataset/` (muy grande)
- `venv/`
- `logs/`

---

## 10. SOLUCIÓN DE PROBLEMAS

### Error: TensorFlow no encontrado

```powershell
pip install tensorflow
```

### Error: CUDA/GPU no detectada

TensorFlow funcionará con CPU. Para GPU:
1. Instala CUDA Toolkit 11.8
2. Instala cuDNN 8.6
3. `pip install tensorflow[and-cuda]`

### Error: SUMO no encontrado

1. Descarga SUMO: https://sumo.dlr.de/docs/Downloads.php
2. Añade a PATH: `C:\Program Files (x86)\Eclipse\Sumo\bin`
3. Reinicia PowerShell

### Error: Lane 'X_2' not known

Ignora estos errores, el sistema funciona correctamente.

### Error: Memoria insuficiente

```powershell
python entrenar_cnn.py --modo clasificador --epochs 30 --batch 16 --ligero
```

### Error: Dataset vacío

```powershell
python setup_atlas.py
```

---

## 📊 RESUMEN DE COMANDOS

| Paso | Comando |
|------|---------|
| Instalar deps | `pip install numpy tensorflow opencv-python Pillow traci sumolib scikit-learn` |
| Setup inicial | `python setup_atlas.py` |
| Demo CNN | `python entrenar_cnn.py --modo demo` |
| Entrenar CNN | `python entrenar_cnn.py --modo clasificador --epochs 50` |
| Entrenar DQN | `python entrenar_avanzado.py --episodios 200` |
| Tests | `python tests_automaticos.py` |
| Demo sistema | `python main.py --modo demo` |

---

## 🎯 SIGUIENTE PASO

Una vez completado todo, tendrás:

1. ✅ **CNN entrenada** para clasificar niveles de tráfico
2. ✅ **DQN entrenado** para controlar semáforos
3. ✅ **Sistema de seguridad** para producción
4. ✅ **Código listo** para GitHub/Portfolio

---

## 📞 SOPORTE

Si tienes problemas:
1. Revisa la sección de Solución de Problemas
2. Verifica que todas las dependencias están instaladas
3. Ejecuta los tests automáticos para diagnosticar

---

**¡Buena suerte con tu proyecto ATLAS!** 🚦🤖
