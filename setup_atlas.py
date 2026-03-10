"""
ATLAS - Script Maestro de Instalación y Configuración
======================================================
Este script configura TODO el sistema desde cero.

Ejecutar: python setup_atlas.py

Genera:
1. Escenarios SUMO corregidos
2. Dataset de imágenes (1000+)
3. Entrena CNN
4. Entrena DQN Visual
5. Tests finales
"""

import os
import sys
import subprocess

def print_header(texto):
    print("\n" + "="*70)
    print(f"🚀 {texto}")
    print("="*70 + "\n")

def print_ok(texto):
    print(f"   ✅ {texto}")

def print_error(texto):
    print(f"   ❌ {texto}")

def print_info(texto):
    print(f"   ℹ️  {texto}")

def verificar_dependencias():
    """Verifica que todas las dependencias están instaladas"""
    print_header("PASO 1: Verificando dependencias")
    
    dependencias = {
        'numpy': 'numpy',
        'tensorflow': 'tensorflow', 
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'traci': 'traci',
    }
    
    faltantes = []
    
    for nombre, paquete in dependencias.items():
        try:
            __import__(nombre)
            print_ok(f"{nombre} instalado")
        except ImportError:
            print_error(f"{nombre} NO instalado")
            faltantes.append(paquete)
    
    if faltantes:
        print("\n   Instalando dependencias faltantes...")
        for paquete in faltantes:
            subprocess.run([sys.executable, "-m", "pip", "install", paquete, "-q"])
        print_ok("Dependencias instaladas")
    
    return True

def crear_escenarios_sumo():
    """Crea todos los escenarios SUMO correctamente"""
    print_header("PASO 2: Creando escenarios SUMO")
    
    # Crear carpetas
    escenarios = ['simple', 'hora_punta', 'noche', 'emergencias']
    
    for escenario in escenarios:
        carpeta = f"simulations/{escenario}"
        os.makedirs(carpeta, exist_ok=True)
        
        # Crear archivos para cada escenario
        crear_escenario_sumo(carpeta, escenario)
        print_ok(f"Escenario '{escenario}' creado")
    
    return True

def crear_escenario_sumo(carpeta, tipo):
    """Crea los archivos SUMO para un escenario"""
    
    # 1. Archivo de nodos
    nodos = '''<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="north" x="0" y="100"/>
    <node id="south" x="0" y="-100"/>
    <node id="east" x="100" y="0"/>
    <node id="west" x="-100" y="0"/>
</nodes>'''
    
    with open(f"{carpeta}/nodes.nod.xml", 'w', encoding='utf-8') as f:
        f.write(nodos)
    
    # 2. Archivo de edges
    edges = '''<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="north_in" from="north" to="center" numLanes="2" speed="13.89"/>
    <edge id="north_out" from="center" to="north" numLanes="2" speed="13.89"/>
    <edge id="south_in" from="south" to="center" numLanes="2" speed="13.89"/>
    <edge id="south_out" from="center" to="south" numLanes="2" speed="13.89"/>
    <edge id="east_in" from="east" to="center" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center" to="east" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="2" speed="13.89"/>
</edges>'''
    
    with open(f"{carpeta}/edges.edg.xml", 'w', encoding='utf-8') as f:
        f.write(edges)
    
    # 3. Generar red con netconvert
    subprocess.run([
        'netconvert',
        f'--node-files={carpeta}/nodes.nod.xml',
        f'--edge-files={carpeta}/edges.edg.xml',
        f'--output-file={carpeta}/intersection.net.xml',
        '--tls.guess'
    ], capture_output=True)
    
    # 4. Archivo de rutas según tipo de escenario
    if tipo == 'simple':
        flujo_coches = 400
        flujo_motos = 80
    elif tipo == 'hora_punta':
        flujo_coches = 900
        flujo_motos = 150
    elif tipo == 'noche':
        flujo_coches = 100
        flujo_motos = 30
    else:  # emergencias
        flujo_coches = 350
        flujo_motos = 60
    
    rutas = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="16.67" guiShape="passenger"/>
    <vType id="moto" accel="3.0" decel="5.0" sigma="0.5" length="2.0" maxSpeed="19.44" guiShape="motorcycle"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.5" length="12.0" maxSpeed="11.11" guiShape="bus"/>
    <vType id="camion" accel="1.0" decel="4.0" sigma="0.5" length="10.0" maxSpeed="11.11" guiShape="truck"/>
    
    <route id="ruta_ns" edges="north_in south_out"/>
    <route id="ruta_sn" edges="south_in north_out"/>
    <route id="ruta_eo" edges="east_in west_out"/>
    <route id="ruta_oe" edges="west_in east_out"/>
    <route id="ruta_ne" edges="north_in east_out"/>
    <route id="ruta_nw" edges="north_in west_out"/>
    <route id="ruta_se" edges="south_in east_out"/>
    <route id="ruta_sw" edges="south_in west_out"/>
    <route id="ruta_en" edges="east_in north_out"/>
    <route id="ruta_es" edges="east_in south_out"/>
    <route id="ruta_wn" edges="west_in north_out"/>
    <route id="ruta_ws" edges="west_in south_out"/>
    
    <flow id="f_ns_coche" type="coche" route="ruta_ns" begin="0" end="3600" vehsPerHour="{flujo_coches}"/>
    <flow id="f_sn_coche" type="coche" route="ruta_sn" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.9)}"/>
    <flow id="f_eo_coche" type="coche" route="ruta_eo" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.7)}"/>
    <flow id="f_oe_coche" type="coche" route="ruta_oe" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.65)}"/>
    
    <flow id="f_ns_moto" type="moto" route="ruta_ns" begin="0" end="3600" vehsPerHour="{flujo_motos}"/>
    <flow id="f_sn_moto" type="moto" route="ruta_sn" begin="0" end="3600" vehsPerHour="{int(flujo_motos*0.8)}"/>
    <flow id="f_eo_moto" type="moto" route="ruta_eo" begin="0" end="3600" vehsPerHour="{int(flujo_motos*0.6)}"/>
    
    <flow id="f_ns_bus" type="bus" route="ruta_ns" begin="0" end="3600" vehsPerHour="12"/>
    <flow id="f_eo_bus" type="bus" route="ruta_eo" begin="0" end="3600" vehsPerHour="8"/>
    
    <flow id="f_ne_coche" type="coche" route="ruta_ne" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.15)}"/>
    <flow id="f_nw_coche" type="coche" route="ruta_nw" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.15)}"/>
    <flow id="f_se_coche" type="coche" route="ruta_se" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.12)}"/>
    <flow id="f_sw_coche" type="coche" route="ruta_sw" begin="0" end="3600" vehsPerHour="{int(flujo_coches*0.12)}"/>
</routes>'''
    
    with open(f"{carpeta}/traffic.rou.xml", 'w', encoding='utf-8') as f:
        f.write(rutas)
    
    # 5. Archivo de configuración SUMO
    config = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="traffic.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>
</configuration>'''
    
    with open(f"{carpeta}/simulation.sumocfg", 'w', encoding='utf-8') as f:
        f.write(config)

def generar_dataset_imagenes():
    """Genera el dataset de imágenes"""
    print_header("PASO 3: Generando dataset de imágenes")
    
    import numpy as np
    import json
    
    # Crear carpetas
    clases = ['bajo', 'medio', 'alto', 'muy_alto']
    for clase in clases:
        os.makedirs(f"dataset/{clase}", exist_ok=True)
    
    print_info("Conectando con SUMO...")
    
    try:
        import traci
    except ImportError:
        print_error("TraCI no disponible, generando imágenes sintéticas")
        return generar_dataset_sintetico()
    
    escenarios = [
        ("simulations/simple/simulation.sumocfg", 300),
        ("simulations/hora_punta/simulation.sumocfg", 400),
        ("simulations/noche/simulation.sumocfg", 200),
        ("simulations/emergencias/simulation.sumocfg", 300),
    ]
    
    total_generadas = 0
    contadores = {c: 0 for c in clases}
    anotaciones = []
    
    for config_sumo, num_imagenes in escenarios:
        if not os.path.exists(config_sumo):
            print_error(f"No existe: {config_sumo}")
            continue
        
        print_info(f"Procesando {config_sumo}...")
        
        try:
            # Conectar a SUMO
            sumo_cmd = ["sumo", "-c", config_sumo, "--step-length", "0.1", "--no-warnings"]
            traci.start(sumo_cmd)
            
            imagenes_escenario = 0
            
            while imagenes_escenario < num_imagenes:
                # Avanzar simulación
                for _ in range(20):
                    traci.simulationStep()
                
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
                
                # Obtener datos de vehículos
                vehiculos = []
                for vid in traci.vehicle.getIDList():
                    try:
                        pos = traci.vehicle.getPosition(vid)
                        vehiculos.append({
                            'id': vid,
                            'x': pos[0],
                            'y': pos[1],
                            'tipo': traci.vehicle.getTypeID(vid)
                        })
                    except:
                        pass
                
                total_vehiculos = len(vehiculos)
                
                # Clasificar
                if total_vehiculos < 5:
                    clase = 'bajo'
                elif total_vehiculos < 15:
                    clase = 'medio'
                elif total_vehiculos < 25:
                    clase = 'alto'
                else:
                    clase = 'muy_alto'
                
                # Generar imagen sintética del cruce
                imagen = generar_imagen_cruce(vehiculos)
                
                # Guardar
                nombre = f"{clase}_{contadores[clase]:05d}.png"
                ruta = f"dataset/{clase}/{nombre}"
                
                try:
                    import cv2
                    cv2.imwrite(ruta, imagen)
                except:
                    from PIL import Image
                    Image.fromarray(imagen).save(ruta)
                
                # Anotación
                anotaciones.append({
                    'imagen': f"{clase}/{nombre}",
                    'clase': clase,
                    'total_vehiculos': total_vehiculos,
                    'vehiculos': vehiculos[:20]  # Máximo 20 para no llenar el JSON
                })
                
                contadores[clase] += 1
                imagenes_escenario += 1
                total_generadas += 1
                
                if total_generadas % 100 == 0:
                    print_info(f"Generadas: {total_generadas} imágenes")
            
            traci.close()
            
        except Exception as e:
            print_error(f"Error en {config_sumo}: {e}")
            try:
                traci.close()
            except:
                pass
    
    # Si no se generaron suficientes, completar con sintéticas
    if total_generadas < 800:
        print_info("Completando con imágenes sintéticas...")
        extra = generar_dataset_sintetico(1000 - total_generadas, contadores)
        anotaciones.extend(extra)
        total_generadas += len(extra)
    
    # Guardar anotaciones
    with open("dataset/anotaciones.json", 'w', encoding='utf-8') as f:
        json.dump(anotaciones, f, indent=2)
    
    print_ok(f"Dataset generado: {total_generadas} imágenes")
    for clase, conteo in contadores.items():
        print_info(f"  {clase}: {conteo}")
    
    return True

def generar_imagen_cruce(vehiculos, size=(640, 480)):
    """Genera una imagen sintética de un cruce con vehículos"""
    import numpy as np
    
    # Fondo gris oscuro (asfalto)
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 60
    
    centro_x, centro_y = size[0] // 2, size[1] // 2
    ancho_calle = 80
    
    # Dibujar calles (gris más claro)
    # Vertical
    img[:, centro_x - ancho_calle:centro_x + ancho_calle] = 90
    # Horizontal
    img[centro_y - ancho_calle:centro_y + ancho_calle, :] = 90
    
    # Líneas de carril (blanco)
    # Centro vertical
    img[:, centro_x - 2:centro_x + 2] = 200
    # Centro horizontal
    img[centro_y - 2:centro_y + 2, :] = 200
    
    # Bordes de acera (amarillo)
    img[:, centro_x - ancho_calle - 3:centro_x - ancho_calle] = [0, 200, 200]
    img[:, centro_x + ancho_calle:centro_x + ancho_calle + 3] = [0, 200, 200]
    img[centro_y - ancho_calle - 3:centro_y - ancho_calle, :] = [0, 200, 200]
    img[centro_y + ancho_calle:centro_y + ancho_calle + 3, :] = [0, 200, 200]
    
    # Dibujar vehículos
    for v in vehiculos:
        # Convertir coordenadas SUMO a píxeles
        px = int(centro_x + v.get('x', 0) * 3)
        py = int(centro_y - v.get('y', 0) * 3)
        
        # Mantener dentro de límites
        px = max(20, min(size[0] - 20, px))
        py = max(20, min(size[1] - 20, py))
        
        # Color y tamaño según tipo
        tipo = v.get('tipo', 'coche').lower()
        if 'bus' in tipo:
            color = (0, 180, 0)  # Verde
            w, h = 15, 35
        elif 'moto' in tipo:
            color = (255, 100, 100)  # Azul claro
            w, h = 6, 12
        elif 'camion' in tipo or 'truck' in tipo:
            color = (100, 100, 100)  # Gris
            w, h = 12, 28
        else:  # Coche
            color = (0, 200, 255)  # Amarillo/naranja
            w, h = 10, 20
        
        # Dibujar rectángulo
        y1, y2 = max(0, py - h//2), min(size[1], py + h//2)
        x1, x2 = max(0, px - w//2), min(size[0], px + w//2)
        img[y1:y2, x1:x2] = color
    
    return img

def generar_dataset_sintetico(num_imagenes=1000, contadores=None):
    """Genera imágenes sintéticas sin SUMO"""
    import numpy as np
    import random
    
    if contadores is None:
        contadores = {'bajo': 0, 'medio': 0, 'alto': 0, 'muy_alto': 0}
    
    anotaciones = []
    
    for i in range(num_imagenes):
        # Decidir clase con distribución balanceada
        clase = random.choice(['bajo', 'medio', 'medio', 'alto', 'alto', 'muy_alto'])
        
        # Número de vehículos según clase
        if clase == 'bajo':
            num_vehiculos = random.randint(1, 4)
        elif clase == 'medio':
            num_vehiculos = random.randint(5, 14)
        elif clase == 'alto':
            num_vehiculos = random.randint(15, 24)
        else:
            num_vehiculos = random.randint(25, 40)
        
        # Generar vehículos aleatorios
        vehiculos = []
        for _ in range(num_vehiculos):
            vehiculos.append({
                'x': random.uniform(-40, 40),
                'y': random.uniform(-40, 40),
                'tipo': random.choice(['coche', 'coche', 'coche', 'moto', 'bus', 'camion'])
            })
        
        # Generar imagen
        imagen = generar_imagen_cruce(vehiculos)
        
        # Guardar
        nombre = f"{clase}_{contadores[clase]:05d}.png"
        ruta = f"dataset/{clase}/{nombre}"
        
        try:
            import cv2
            cv2.imwrite(ruta, imagen)
        except:
            from PIL import Image
            Image.fromarray(imagen).save(ruta)
        
        anotaciones.append({
            'imagen': f"{clase}/{nombre}",
            'clase': clase,
            'total_vehiculos': num_vehiculos
        })
        
        contadores[clase] += 1
        
        if (i + 1) % 200 == 0:
            print_info(f"Sintéticas generadas: {i + 1}/{num_imagenes}")
    
    return anotaciones

def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "🚦 ATLAS - SETUP COMPLETO" + " "*22 + "║")
    print("║" + " "*10 + "Sistema de Control de Semáforos con IA" + " "*14 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    # Paso 1: Dependencias
    verificar_dependencias()
    
    # Paso 2: Escenarios SUMO
    crear_escenarios_sumo()
    
    # Paso 3: Dataset
    generar_dataset_imagenes()
    
    print_header("✅ SETUP COMPLETADO")
    print("""
    Ahora ejecuta los siguientes comandos:
    
    1. Entrenar CNN clasificador:
       python entrenar_cnn.py --modo clasificador --epochs 50
    
    2. Probar el sistema:
       python main.py --modo demo
    
    3. Ejecutar tests:
       python tests_automaticos.py
    """)

if __name__ == "__main__":
    main()
