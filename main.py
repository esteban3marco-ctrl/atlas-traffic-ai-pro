"""
ATLAS - Sistema Completo de Control de Semáforos con Visión por Computador
===========================================================================

Este es el archivo principal que integra todos los componentes:
- YOLOv8 para detección de vehículos
- TensorFlow/Keras para redes neuronales
- DQN para toma de decisiones
- Sistema de seguridad para producción
- Simulador SUMO para pruebas

Uso:
    python main.py --modo demo          # Demo rápida
    python main.py --modo entrenar      # Entrenar con imágenes
    python main.py --modo produccion    # Modo producción
    python main.py --modo generar_dataset  # Generar dataset de imágenes
"""

import os
import sys
import argparse
import time
import numpy as np
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("ATLAS")


def verificar_dependencias():
    """Verifica que todas las dependencias están instaladas"""
    
    print("\n" + "="*60)
    print("🔍 Verificando dependencias...")
    print("="*60)
    
    dependencias = {
        'numpy': 'numpy',
        'TensorFlow': 'tensorflow',
        'OpenCV': 'cv2',
        'YOLO (ultralytics)': 'ultralytics',
        'TraCI (SUMO)': 'traci',
        'PIL': 'PIL'
    }
    
    instaladas = {}
    
    for nombre, modulo in dependencias.items():
        try:
            mod = __import__(modulo)
            version = getattr(mod, '__version__', 'OK')
            instaladas[nombre] = True
            print(f"   ✅ {nombre}: {version}")
        except ImportError:
            instaladas[nombre] = False
            print(f"   ❌ {nombre}: No instalado")
    
    # Verificar SUMO
    import subprocess
    try:
        result = subprocess.run(['sumo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"   ✅ SUMO: {version}")
            instaladas['SUMO'] = True
        else:
            print(f"   ❌ SUMO: No encontrado")
            instaladas['SUMO'] = False
    except:
        print(f"   ❌ SUMO: No encontrado")
        instaladas['SUMO'] = False
    
    print()
    
    # Instrucciones de instalación
    faltantes = [k for k, v in instaladas.items() if not v]
    if faltantes:
        print("📦 Para instalar las dependencias faltantes:")
        print()
        if 'TensorFlow' in faltantes:
            print("   pip install tensorflow")
        if 'OpenCV' in faltantes:
            print("   pip install opencv-python")
        if 'YOLO (ultralytics)' in faltantes:
            print("   pip install ultralytics")
        if 'TraCI (SUMO)' in faltantes:
            print("   pip install traci sumolib")
        if 'SUMO' in faltantes:
            print("   Descargar SUMO de: https://sumo.dlr.de/docs/Downloads.php")
        print()
    
    return instaladas


def modo_demo():
    """Ejecuta una demo del sistema"""
    
    print("\n" + "="*70)
    print("🚦 ATLAS - Demo del Sistema de Control de Semáforos con IA")
    print("="*70 + "\n")
    
    # Verificar que existe la configuración
    config_paths = [
        "simulations/simple/simulation.sumocfg",
        "../atlas-produccion/simulations/simple/simulation.sumocfg"
    ]
    
    config_sumo = None
    for path in config_paths:
        if os.path.exists(path):
            config_sumo = path
            break
    
    if not config_sumo:
        print("❌ No se encontró configuración de SUMO")
        print("   Ejecuta primero: python crear_todo.py")
        return
    
    print(f"📁 Usando configuración: {config_sumo}")
    
    # Importar componentes
    try:
        from detector_vehiculos import DetectorVehiculos
        from simulador_camaras import SimuladorCamaras
        
        print("✅ Componentes cargados")
    except ImportError as e:
        print(f"❌ Error importando componentes: {e}")
        return
    
    # Crear detector
    detector = DetectorVehiculos()
    print(f"   Detector: {'YOLO' if not detector.usando_simulacion else 'Simulado'}")
    
    # Crear simulador
    print("\n🚗 Iniciando simulación...")
    simulador = SimuladorCamaras(config_sumo, usar_gui=True)
    simulador.conectar()
    
    # Cargar agente DQN si existe
    agente = None
    modelo_path = "modelos/mejor_agente_simple.npz"
    if not os.path.exists(modelo_path):
        modelo_path = "../atlas-produccion/modelos/mejor_agente_simple.npz"
    
    if os.path.exists(modelo_path):
        try:
            # Intentar cargar con TensorFlow primero
            from modelo_tensorflow import AgenteDQN_TensorFlow
            agente = AgenteDQN_TensorFlow(estado_dim=12)
            print(f"   Agente TensorFlow creado")
        except:
            # Fallback a versión numpy
            try:
                sys.path.insert(0, '../atlas-produccion')
                from entrenar_avanzado import AgenteDQN
                agente = AgenteDQN(estado_dim=12)
                if agente.cargar(modelo_path):
                    agente.epsilon = 0  # Sin exploración
                    print(f"   Agente numpy cargado desde {modelo_path}")
            except:
                print("   ⚠️ No se pudo cargar agente")
    
    # Loop de demo
    print("\n" + "-"*60)
    print("   Presiona Ctrl+C para detener")
    print("-"*60 + "\n")
    
    frames = 0
    acciones = {'mantener': 0, 'cambiar_ns': 0, 'cambiar_eo': 0, 'extender': 0}
    
    try:
        while simulador.simulacion_activa() and frames < 500:
            # Avanzar simulación
            for _ in range(10):
                simulador.paso_simulacion()
            
            # Capturar frame
            frame = simulador.capturar_frame()
            if not frame:
                continue
            
            # Detectar vehículos
            resultado = detector.detectar(frame.imagen)
            estado = detector.obtener_estado_trafico(resultado)
            
            # Obtener decisión del agente
            if agente:
                estado_vector = np.array([
                    estado['cola_norte'] / 50.0,
                    estado['cola_sur'] / 50.0,
                    estado['cola_este'] / 50.0,
                    estado['cola_oeste'] / 50.0,
                    estado.get('coches', 0) / 50.0,
                    estado.get('motos', 0) / 20.0,
                    estado.get('buses', 0) / 10.0,
                    estado.get('camiones', 0) / 10.0,
                    0.0, 0.0, 0.0, 0.0
                ], dtype=np.float32)
                
                accion = agente.obtener_accion(estado_vector)
                nombre_accion = ['mantener', 'cambiar_ns', 'cambiar_eo', 'extender'][accion]
            else:
                nombre_accion = 'mantener'
            
            acciones[nombre_accion] += 1
            frames += 1
            
            # Mostrar info cada 20 frames
            if frames % 20 == 0:
                fase = "N-S 🟢" if frame.fase_semaforo == 0 else "E-O 🟢"
                print(f"Frame {frames:4d} | "
                      f"Vehículos: {resultado.total_vehiculos:2d} | "
                      f"Fase: {fase} | "
                      f"Acción IA: {nombre_accion:12s} | "
                      f"Colas: N={estado['cola_norte']:2d} S={estado['cola_sur']:2d} "
                      f"E={estado['cola_este']:2d} O={estado['cola_oeste']:2d}")
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrumpida")
    
    finally:
        simulador.desconectar()
    
    # Resumen
    print("\n" + "="*60)
    print("📊 Resumen de la demo")
    print("="*60)
    print(f"   Frames procesados: {frames}")
    print(f"   Acciones tomadas:")
    for accion, conteo in acciones.items():
        print(f"      {accion}: {conteo} ({conteo/max(1,frames)*100:.1f}%)")
    print()


def modo_entrenar():
    """Entrena el sistema con imágenes"""
    
    print("\n" + "="*70)
    print("🏋️ ATLAS - Entrenamiento con Visión por Computador")
    print("="*70 + "\n")
    
    # Verificar TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        print(f"   GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError:
        print("❌ TensorFlow no instalado")
        print("   Instalar con: pip install tensorflow")
        return
    
    # Verificar dataset
    dataset_path = "dataset"
    if not os.path.exists(f"{dataset_path}/imagenes"):
        print("\n⚠️ No hay dataset de imágenes")
        print("   Primero genera uno con: python main.py --modo generar_dataset")
        return
    
    # TODO: Implementar entrenamiento completo
    print("\n📝 Entrenamiento con imágenes próximamente...")
    print("   Por ahora usa: python entrenar_avanzado.py --episodios 100")


def modo_generar_dataset():
    """Genera un dataset de imágenes"""
    
    print("\n" + "="*70)
    print("📸 ATLAS - Generación de Dataset")
    print("="*70 + "\n")
    
    # Verificar SUMO
    config_paths = [
        "simulations/simple/simulation.sumocfg",
        "../atlas-produccion/simulations/simple/simulation.sumocfg"
    ]
    
    config_sumo = None
    for path in config_paths:
        if os.path.exists(path):
            config_sumo = path
            break
    
    if not config_sumo:
        print("❌ No se encontró configuración de SUMO")
        return
    
    from simulador_camaras import SimuladorCamaras, GeneradorDataset
    
    simulador = SimuladorCamaras(config_sumo, usar_gui=False)
    generador = GeneradorDataset(simulador, carpeta_salida="dataset")
    
    print("📂 Generando dataset en: dataset/")
    print("   Esto puede tardar unos minutos...\n")
    
    num_frames = generador.generar(num_frames=500, intervalo_pasos=20)
    
    print(f"\n✅ Dataset generado: {num_frames} imágenes")


def modo_produccion():
    """Modo producción con todas las salvaguardas"""
    
    print("\n" + "="*70)
    print("🏭 ATLAS - Modo Producción")
    print("="*70 + "\n")
    
    print("⚠️ El modo producción requiere:")
    print("   - Hardware de control homologado")
    print("   - Cámaras conectadas")
    print("   - Permisos del ayuntamiento")
    print()
    print("   Para pruebas, usa: python main.py --modo demo")


def main():
    parser = argparse.ArgumentParser(
        description='ATLAS - Sistema de Control de Semáforos con IA y Visión por Computador',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --verificar              Verificar dependencias
  python main.py --modo demo              Demo del sistema
  python main.py --modo entrenar          Entrenar con imágenes
  python main.py --modo generar_dataset   Generar dataset
  python main.py --modo produccion        Modo producción (requiere hardware)
        """
    )
    
    parser.add_argument(
        '--modo',
        type=str,
        choices=['demo', 'entrenar', 'generar_dataset', 'produccion'],
        default='demo',
        help='Modo de operación'
    )
    
    parser.add_argument(
        '--verificar',
        action='store_true',
        help='Solo verificar dependencias'
    )
    
    args = parser.parse_args()
    
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🚦 ATLAS - Traffic Light AI System" + " "*16 + "║")
    print("║" + " "*10 + "Sistema de Control de Semáforos con IA" + " "*13 + "║")
    print("╚" + "="*68 + "╝")
    
    if args.verificar:
        verificar_dependencias()
        return
    
    # Verificar dependencias primero
    deps = verificar_dependencias()
    
    if args.modo == 'demo':
        modo_demo()
    elif args.modo == 'entrenar':
        modo_entrenar()
    elif args.modo == 'generar_dataset':
        modo_generar_dataset()
    elif args.modo == 'produccion':
        modo_produccion()


if __name__ == "__main__":
    main()
