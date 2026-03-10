"""
ATLAS - Creador de TODOS los escenarios
========================================
Ejecutar: python crear_todo.py
"""

import os
import subprocess

def crear_carpeta(carpeta):
    os.makedirs(carpeta, exist_ok=True)

def ejecutar_netconvert(carpeta):
    subprocess.run([
        "netconvert",
        f"--node-files={carpeta}/nodes.nod.xml",
        f"--edge-files={carpeta}/edges.edg.xml", 
        f"--output-file={carpeta}/intersection.net.xml",
        "--tls.guess",
        "--no-warnings", "true"
    ], capture_output=True)

def crear_config(carpeta):
    config = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="traffic.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""
    with open(f"{carpeta}/simulation.sumocfg", "w", encoding="utf-8") as f:
        f.write(config)


# =============================================================================
# CRUCE SIMPLE (4 vías, 2 carriles)
# =============================================================================

def crear_simple():
    print("📍 Creando cruce SIMPLE...")
    carpeta = "simulations/simple"
    crear_carpeta(carpeta)
    
    nodos = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="north" x="0" y="100"/>
    <node id="south" x="0" y="-100"/>
    <node id="east" x="100" y="0"/>
    <node id="west" x="-100" y="0"/>
</nodes>"""
    
    edges = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="north_in" from="north" to="center" numLanes="2" speed="13.89"/>
    <edge id="north_out" from="center" to="north" numLanes="2" speed="13.89"/>
    <edge id="south_in" from="south" to="center" numLanes="2" speed="13.89"/>
    <edge id="south_out" from="center" to="south" numLanes="2" speed="13.89"/>
    <edge id="east_in" from="east" to="center" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center" to="east" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="2" speed="13.89"/>
</edges>"""
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="camion" accel="1.0" decel="3.5" sigma="0.5" length="10" maxSpeed="25" color="gray"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    <route id="r_ne" edges="north_in east_out"/>
    <route id="r_nw" edges="north_in west_out"/>
    <route id="r_se" edges="south_in east_out"/>
    <route id="r_sw" edges="south_in west_out"/>
    
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="400" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="350" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="300" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="280" departLane="best"/>
    <flow id="c_ne" type="coche" route="r_ne" begin="0" end="3600" vehsPerHour="60" departLane="best"/>
    <flow id="c_nw" type="coche" route="r_nw" begin="0" end="3600" vehsPerHour="60" departLane="best"/>
    <flow id="m_ns" type="moto" route="r_ns" begin="0" end="3600" vehsPerHour="80"/>
    <flow id="m_ew" type="moto" route="r_ew" begin="0" end="3600" vehsPerHour="60"/>
    <flow id="b_ns" type="bus" route="r_ns" begin="0" end="3600" period="300"/>
    <flow id="b_ew" type="bus" route="r_ew" begin="150" end="3600" period="360"/>
    <flow id="t_ns" type="camion" route="r_ns" begin="0" end="3600" vehsPerHour="30"/>
    <flow id="e_ns" type="emergencia" route="r_ns" begin="600" end="3600" period="900"/>
</routes>"""
    
    with open(f"{carpeta}/nodes.nod.xml", "w", encoding="utf-8") as f: f.write(nodos)
    with open(f"{carpeta}/edges.edg.xml", "w", encoding="utf-8") as f: f.write(edges)
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    ejecutar_netconvert(carpeta)
    crear_config(carpeta)
    print("   ✅ Cruce simple creado")


# =============================================================================
# CRUCE SIMPLE - HORA PUNTA MAÑANA
# =============================================================================

def crear_simple_hora_punta():
    print("📍 Creando cruce SIMPLE - HORA PUNTA MAÑANA...")
    carpeta = "simulations/simple_hora_punta"
    crear_carpeta(carpeta)
    
    # Copiar red del simple
    import shutil
    shutil.copy("simulations/simple/nodes.nod.xml", carpeta)
    shutil.copy("simulations/simple/edges.edg.xml", carpeta)
    ejecutar_netconvert(carpeta)
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="camion" accel="1.0" decel="3.5" sigma="0.5" length="10" maxSpeed="25" color="gray"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    
    <!-- MUCHO MÁS TRÁFICO -->
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="800" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="700" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="600" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="550" departLane="best"/>
    <flow id="m_ns" type="moto" route="r_ns" begin="0" end="3600" vehsPerHour="150"/>
    <flow id="m_ew" type="moto" route="r_ew" begin="0" end="3600" vehsPerHour="120"/>
    <flow id="b_ns" type="bus" route="r_ns" begin="0" end="3600" period="180"/>
    <flow id="b_ew" type="bus" route="r_ew" begin="90" end="3600" period="240"/>
    <flow id="t_ns" type="camion" route="r_ns" begin="0" end="3600" vehsPerHour="60"/>
    <flow id="e_ns" type="emergencia" route="r_ns" begin="300" end="3600" period="600"/>
</routes>"""
    
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    crear_config(carpeta)
    print("   ✅ Cruce simple hora punta creado")


# =============================================================================
# CRUCE SIMPLE - NOCHE
# =============================================================================

def crear_simple_noche():
    print("📍 Creando cruce SIMPLE - NOCHE...")
    carpeta = "simulations/simple_noche"
    crear_carpeta(carpeta)
    
    import shutil
    shutil.copy("simulations/simple/nodes.nod.xml", carpeta)
    shutil.copy("simulations/simple/edges.edg.xml", carpeta)
    ejecutar_netconvert(carpeta)
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="taxi" accel="2.8" decel="4.5" sigma="0.4" length="5" maxSpeed="50" color="white"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    
    <!-- POCO TRÁFICO -->
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="80" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="70" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="60" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="50" departLane="best"/>
    <flow id="t_ns" type="taxi" route="r_ns" begin="0" end="3600" vehsPerHour="40"/>
    <flow id="t_ew" type="taxi" route="r_ew" begin="0" end="3600" vehsPerHour="30"/>
    <flow id="m_ns" type="moto" route="r_ns" begin="0" end="3600" vehsPerHour="20"/>
    <flow id="e_ns" type="emergencia" route="r_ns" begin="600" end="3600" period="1200"/>
</routes>"""
    
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    crear_config(carpeta)
    print("   ✅ Cruce simple noche creado")


# =============================================================================
# CRUCE SIMPLE - EMERGENCIAS
# =============================================================================

def crear_simple_emergencias():
    print("📍 Creando cruce SIMPLE - EMERGENCIAS...")
    carpeta = "simulations/simple_emergencias"
    crear_carpeta(carpeta)
    
    import shutil
    shutil.copy("simulations/simple/nodes.nod.xml", carpeta)
    shutil.copy("simulations/simple/edges.edg.xml", carpeta)
    ejecutar_netconvert(carpeta)
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="ambulancia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="1,1,1"/>
    <vType id="policia" accel="4.0" decel="5.5" sigma="0.2" length="5" maxSpeed="80" color="0,0,1"/>
    <vType id="bomberos" accel="2.5" decel="4.5" sigma="0.2" length="10" maxSpeed="60" color="1,0,0"/>
    
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="300" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="280" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="250" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="250" departLane="best"/>
    
    <!-- MUCHAS EMERGENCIAS -->
    <flow id="amb_ns" type="ambulancia" route="r_ns" begin="60" end="3600" period="180"/>
    <flow id="amb_ew" type="ambulancia" route="r_ew" begin="120" end="3600" period="240"/>
    <flow id="pol_ns" type="policia" route="r_ns" begin="30" end="3600" period="200"/>
    <flow id="pol_sn" type="policia" route="r_sn" begin="90" end="3600" period="220"/>
    <flow id="bomb_ew" type="bomberos" route="r_ew" begin="300" end="3600" period="400"/>
</routes>"""
    
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    crear_config(carpeta)
    print("   ✅ Cruce simple emergencias creado")


# =============================================================================
# AVENIDA (4 carriles principales, 2 secundarios)
# =============================================================================

def crear_avenida():
    print("📍 Creando AVENIDA...")
    carpeta = "simulations/avenida"
    crear_carpeta(carpeta)
    
    nodos = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="north" x="0" y="150"/>
    <node id="south" x="0" y="-150"/>
    <node id="east" x="150" y="0"/>
    <node id="west" x="-150" y="0"/>
</nodes>"""
    
    edges = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <!-- Avenida principal N-S: 4 carriles, 60 km/h -->
    <edge id="north_in" from="north" to="center" numLanes="4" speed="16.67"/>
    <edge id="north_out" from="center" to="north" numLanes="4" speed="16.67"/>
    <edge id="south_in" from="south" to="center" numLanes="4" speed="16.67"/>
    <edge id="south_out" from="center" to="south" numLanes="4" speed="16.67"/>
    <!-- Calles secundarias E-O: 2 carriles, 50 km/h -->
    <edge id="east_in" from="east" to="center" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center" to="east" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="2" speed="13.89"/>
</edges>"""
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="coche_rapido" accel="3.0" decel="5.0" sigma="0.4" length="4.5" maxSpeed="70" color="orange"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="camion" accel="1.0" decel="3.5" sigma="0.5" length="12" maxSpeed="25" color="gray"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="80" color="red"/>
    
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    <route id="r_ne" edges="north_in east_out"/>
    <route id="r_nw" edges="north_in west_out"/>
    
    <!-- Avenida: MUCHO tráfico N-S -->
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="700" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="650" departLane="best"/>
    <flow id="cr_ns" type="coche_rapido" route="r_ns" begin="0" end="3600" vehsPerHour="200" departLane="best"/>
    <flow id="cr_sn" type="coche_rapido" route="r_sn" begin="0" end="3600" vehsPerHour="180" departLane="best"/>
    <!-- Calles secundarias: menos tráfico -->
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="200" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="180" departLane="best"/>
    <!-- Giros -->
    <flow id="c_ne" type="coche" route="r_ne" begin="0" end="3600" vehsPerHour="80" departLane="best"/>
    <flow id="c_nw" type="coche" route="r_nw" begin="0" end="3600" vehsPerHour="70" departLane="best"/>
    <!-- Motos -->
    <flow id="m_ns" type="moto" route="r_ns" begin="0" end="3600" vehsPerHour="120"/>
    <flow id="m_sn" type="moto" route="r_sn" begin="0" end="3600" vehsPerHour="100"/>
    <!-- Buses (línea de avenida) -->
    <flow id="b_ns" type="bus" route="r_ns" begin="0" end="3600" period="180"/>
    <flow id="b_sn" type="bus" route="r_sn" begin="90" end="3600" period="180"/>
    <!-- Camiones -->
    <flow id="t_ns" type="camion" route="r_ns" begin="0" end="3600" vehsPerHour="40"/>
    <!-- Emergencias -->
    <flow id="e_ns" type="emergencia" route="r_ns" begin="300" end="3600" period="600"/>
</routes>"""
    
    with open(f"{carpeta}/nodes.nod.xml", "w", encoding="utf-8") as f: f.write(nodos)
    with open(f"{carpeta}/edges.edg.xml", "w", encoding="utf-8") as f: f.write(edges)
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    ejecutar_netconvert(carpeta)
    crear_config(carpeta)
    print("   ✅ Avenida creada")


# =============================================================================
# CRUCE EN T
# =============================================================================

def crear_cruce_t():
    print("📍 Creando CRUCE EN T...")
    carpeta = "simulations/cruce_t"
    crear_carpeta(carpeta)
    
    nodos = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="north" x="0" y="100"/>
    <node id="east" x="100" y="0"/>
    <node id="west" x="-100" y="0"/>
</nodes>"""
    
    edges = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="north_in" from="north" to="center" numLanes="2" speed="13.89"/>
    <edge id="north_out" from="center" to="north" numLanes="2" speed="13.89"/>
    <edge id="east_in" from="east" to="center" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center" to="east" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="2" speed="13.89"/>
</edges>"""
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <route id="r_ne" edges="north_in east_out"/>
    <route id="r_nw" edges="north_in west_out"/>
    <route id="r_en" edges="east_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_wn" edges="west_in north_out"/>
    <route id="r_we" edges="west_in east_out"/>
    
    <flow id="c_ne" type="coche" route="r_ne" begin="0" end="3600" vehsPerHour="250" departLane="best"/>
    <flow id="c_nw" type="coche" route="r_nw" begin="0" end="3600" vehsPerHour="200" departLane="best"/>
    <flow id="c_en" type="coche" route="r_en" begin="0" end="3600" vehsPerHour="180" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="350" departLane="best"/>
    <flow id="c_wn" type="coche" route="r_wn" begin="0" end="3600" vehsPerHour="150" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="320" departLane="best"/>
    <flow id="m_ew" type="moto" route="r_ew" begin="0" end="3600" vehsPerHour="60"/>
    <flow id="b_ew" type="bus" route="r_ew" begin="0" end="3600" period="300"/>
    <flow id="e_ne" type="emergencia" route="r_ne" begin="600" end="3600" period="800"/>
</routes>"""
    
    with open(f"{carpeta}/nodes.nod.xml", "w", encoding="utf-8") as f: f.write(nodos)
    with open(f"{carpeta}/edges.edg.xml", "w", encoding="utf-8") as f: f.write(edges)
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    ejecutar_netconvert(carpeta)
    crear_config(carpeta)
    print("   ✅ Cruce en T creado")


# =============================================================================
# CRUCE DOBLE (2 semáforos conectados)
# =============================================================================

def crear_doble():
    print("📍 Creando CRUCE DOBLE...")
    carpeta = "simulations/doble"
    crear_carpeta(carpeta)
    
    nodos = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="center2" x="200" y="0" type="traffic_light"/>
    <node id="north" x="0" y="100"/>
    <node id="south" x="0" y="-100"/>
    <node id="north2" x="200" y="100"/>
    <node id="south2" x="200" y="-100"/>
    <node id="west" x="-100" y="0"/>
    <node id="east" x="300" y="0"/>
</nodes>"""
    
    edges = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <!-- Cruce 1 -->
    <edge id="north_in" from="north" to="center" numLanes="2" speed="13.89"/>
    <edge id="north_out" from="center" to="north" numLanes="2" speed="13.89"/>
    <edge id="south_in" from="south" to="center" numLanes="2" speed="13.89"/>
    <edge id="south_out" from="center" to="south" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="2" speed="13.89"/>
    <!-- Conexión entre cruces -->
    <edge id="link_12" from="center" to="center2" numLanes="2" speed="13.89"/>
    <edge id="link_21" from="center2" to="center" numLanes="2" speed="13.89"/>
    <!-- Cruce 2 -->
    <edge id="north2_in" from="north2" to="center2" numLanes="2" speed="13.89"/>
    <edge id="north2_out" from="center2" to="north2" numLanes="2" speed="13.89"/>
    <edge id="south2_in" from="south2" to="center2" numLanes="2" speed="13.89"/>
    <edge id="south2_out" from="center2" to="south2" numLanes="2" speed="13.89"/>
    <edge id="east_in" from="east" to="center2" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center2" to="east" numLanes="2" speed="13.89"/>
</edges>"""
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <!-- Rutas que atraviesan ambos cruces -->
    <route id="r_we" edges="west_in link_12 east_out"/>
    <route id="r_ew" edges="east_in link_21 west_out"/>
    <!-- Rutas locales cruce 1 -->
    <route id="r_ns1" edges="north_in south_out"/>
    <route id="r_sn1" edges="south_in north_out"/>
    <!-- Rutas locales cruce 2 -->
    <route id="r_ns2" edges="north2_in south2_out"/>
    <route id="r_sn2" edges="south2_in north2_out"/>
    
    <!-- Tráfico principal E-O -->
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="500" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="450" departLane="best"/>
    <!-- Tráfico local -->
    <flow id="c_ns1" type="coche" route="r_ns1" begin="0" end="3600" vehsPerHour="200" departLane="best"/>
    <flow id="c_sn1" type="coche" route="r_sn1" begin="0" end="3600" vehsPerHour="180" departLane="best"/>
    <flow id="c_ns2" type="coche" route="r_ns2" begin="0" end="3600" vehsPerHour="200" departLane="best"/>
    <flow id="c_sn2" type="coche" route="r_sn2" begin="0" end="3600" vehsPerHour="180" departLane="best"/>
    <!-- Motos -->
    <flow id="m_we" type="moto" route="r_we" begin="0" end="3600" vehsPerHour="80"/>
    <!-- Buses -->
    <flow id="b_we" type="bus" route="r_we" begin="0" end="3600" period="300"/>
    <!-- Emergencias -->
    <flow id="e_we" type="emergencia" route="r_we" begin="600" end="3600" period="900"/>
</routes>"""
    
    with open(f"{carpeta}/nodes.nod.xml", "w", encoding="utf-8") as f: f.write(nodos)
    with open(f"{carpeta}/edges.edg.xml", "w", encoding="utf-8") as f: f.write(edges)
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    ejecutar_netconvert(carpeta)
    crear_config(carpeta)
    print("   ✅ Cruce doble creado")


# =============================================================================
# CRUCE COMPLEJO (con giros dedicados)
# =============================================================================

def crear_complejo():
    print("📍 Creando CRUCE COMPLEJO...")
    carpeta = "simulations/complejo"
    crear_carpeta(carpeta)
    
    nodos = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="north" x="0" y="120"/>
    <node id="south" x="0" y="-120"/>
    <node id="east" x="120" y="0"/>
    <node id="west" x="-120" y="0"/>
</nodes>"""
    
    # 3 carriles por dirección
    edges = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="north_in" from="north" to="center" numLanes="3" speed="13.89"/>
    <edge id="north_out" from="center" to="north" numLanes="3" speed="13.89"/>
    <edge id="south_in" from="south" to="center" numLanes="3" speed="13.89"/>
    <edge id="south_out" from="center" to="south" numLanes="3" speed="13.89"/>
    <edge id="east_in" from="east" to="center" numLanes="3" speed="13.89"/>
    <edge id="east_out" from="center" to="east" numLanes="3" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="3" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="3" speed="13.89"/>
</edges>"""
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="camion" accel="1.0" decel="3.5" sigma="0.5" length="12" maxSpeed="25" color="gray"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <!-- Rutas rectas -->
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    <!-- Giros derechas -->
    <route id="r_ne" edges="north_in east_out"/>
    <route id="r_es" edges="east_in south_out"/>
    <route id="r_sw" edges="south_in west_out"/>
    <route id="r_wn" edges="west_in north_out"/>
    <!-- Giros izquierdas -->
    <route id="r_nw" edges="north_in west_out"/>
    <route id="r_en" edges="east_in north_out"/>
    <route id="r_se" edges="south_in east_out"/>
    <route id="r_ws" edges="west_in south_out"/>
    
    <!-- Tráfico recto (principal) -->
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="500" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="480" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="450" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="420" departLane="best"/>
    <!-- Giros derechas -->
    <flow id="c_ne" type="coche" route="r_ne" begin="0" end="3600" vehsPerHour="100" departLane="best"/>
    <flow id="c_es" type="coche" route="r_es" begin="0" end="3600" vehsPerHour="90" departLane="best"/>
    <flow id="c_sw" type="coche" route="r_sw" begin="0" end="3600" vehsPerHour="85" departLane="best"/>
    <flow id="c_wn" type="coche" route="r_wn" begin="0" end="3600" vehsPerHour="95" departLane="best"/>
    <!-- Giros izquierdas -->
    <flow id="c_nw" type="coche" route="r_nw" begin="0" end="3600" vehsPerHour="70" departLane="best"/>
    <flow id="c_en" type="coche" route="r_en" begin="0" end="3600" vehsPerHour="65" departLane="best"/>
    <flow id="c_se" type="coche" route="r_se" begin="0" end="3600" vehsPerHour="60" departLane="best"/>
    <flow id="c_ws" type="coche" route="r_ws" begin="0" end="3600" vehsPerHour="75" departLane="best"/>
    <!-- Motos -->
    <flow id="m_ns" type="moto" route="r_ns" begin="0" end="3600" vehsPerHour="100"/>
    <flow id="m_ew" type="moto" route="r_ew" begin="0" end="3600" vehsPerHour="80"/>
    <!-- Buses -->
    <flow id="b_ns" type="bus" route="r_ns" begin="0" end="3600" period="240"/>
    <flow id="b_ew" type="bus" route="r_ew" begin="120" end="3600" period="300"/>
    <!-- Camiones -->
    <flow id="t_ns" type="camion" route="r_ns" begin="0" end="3600" vehsPerHour="40"/>
    <flow id="t_ew" type="camion" route="r_ew" begin="0" end="3600" vehsPerHour="35"/>
    <!-- Emergencias -->
    <flow id="e_ns" type="emergencia" route="r_ns" begin="300" end="3600" period="600"/>
    <flow id="e_ew" type="emergencia" route="r_ew" begin="450" end="3600" period="700"/>
</routes>"""
    
    with open(f"{carpeta}/nodes.nod.xml", "w", encoding="utf-8") as f: f.write(nodos)
    with open(f"{carpeta}/edges.edg.xml", "w", encoding="utf-8") as f: f.write(edges)
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    ejecutar_netconvert(carpeta)
    crear_config(carpeta)
    print("   ✅ Cruce complejo creado")


# =============================================================================
# EVENTO ESPECIAL (concierto, partido...)
# =============================================================================

def crear_evento():
    print("📍 Creando EVENTO ESPECIAL...")
    carpeta = "simulations/evento"
    crear_carpeta(carpeta)
    
    import shutil
    shutil.copy("simulations/simple/nodes.nod.xml", carpeta)
    shutil.copy("simulations/simple/edges.edg.xml", carpeta)
    ejecutar_netconvert(carpeta)
    
    rutas = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="coche" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="yellow"/>
    <vType id="moto" accel="4.0" decel="6.0" sigma="0.5" length="2.2" maxSpeed="60" color="blue"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="30" color="green"/>
    <vType id="emergencia" accel="3.5" decel="5.0" sigma="0.2" length="6" maxSpeed="70" color="red"/>
    
    <route id="r_ns" edges="north_in south_out"/>
    <route id="r_sn" edges="south_in north_out"/>
    <route id="r_ew" edges="east_in west_out"/>
    <route id="r_we" edges="west_in east_out"/>
    
    <!-- TRÁFICO MUY ALTO (evento) -->
    <flow id="c_ns" type="coche" route="r_ns" begin="0" end="3600" vehsPerHour="900" departLane="best"/>
    <flow id="c_sn" type="coche" route="r_sn" begin="0" end="3600" vehsPerHour="850" departLane="best"/>
    <flow id="c_ew" type="coche" route="r_ew" begin="0" end="3600" vehsPerHour="800" departLane="best"/>
    <flow id="c_we" type="coche" route="r_we" begin="0" end="3600" vehsPerHour="750" departLane="best"/>
    <flow id="m_ns" type="moto" route="r_ns" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="m_ew" type="moto" route="r_ew" begin="0" end="3600" vehsPerHour="180"/>
    <flow id="b_ns" type="bus" route="r_ns" begin="0" end="3600" period="120"/>
    <flow id="b_ew" type="bus" route="r_ew" begin="60" end="3600" period="150"/>
    <flow id="e_ns" type="emergencia" route="r_ns" begin="300" end="3600" period="500"/>
    <flow id="e_ew" type="emergencia" route="r_ew" begin="400" end="3600" period="600"/>
</routes>"""
    
    with open(f"{carpeta}/traffic.rou.xml", "w", encoding="utf-8") as f: f.write(rutas)
    crear_config(carpeta)
    print("   ✅ Evento especial creado")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 70)
    print("🏗️  ATLAS - CREANDO TODOS LOS ESCENARIOS")
    print("=" * 70)
    print()
    
    # Crear carpeta base
    crear_carpeta("simulations")
    crear_carpeta("modelos")
    
    # Crear todos los cruces
    crear_simple()
    crear_simple_hora_punta()
    crear_simple_noche()
    crear_simple_emergencias()
    crear_avenida()
    crear_cruce_t()
    crear_doble()
    crear_complejo()
    crear_evento()
    
    print()
    print("=" * 70)
    print("✅ ¡TODOS LOS ESCENARIOS CREADOS!")
    print("=" * 70)
    print()
    print("📁 Escenarios disponibles:")
    print()
    print("   CRUCES BÁSICOS:")
    print("   ├── simple              - Cruce 4 vías, 2 carriles")
    print("   ├── avenida             - Avenida 4 carriles + calles 2 carriles")
    print("   ├── cruce_t             - Cruce en T (3 vías)")
    print("   ├── doble               - 2 semáforos conectados")
    print("   └── complejo            - Cruce 4 vías, 3 carriles, muchos giros")
    print()
    print("   ESCENARIOS DE TRÁFICO:")
    print("   ├── simple_hora_punta   - Mucho tráfico (mañana)")
    print("   ├── simple_noche        - Poco tráfico (noche)")
    print("   ├── simple_emergencias  - Muchas ambulancias/policía")
    print("   └── evento              - Tráfico extremo (concierto/partido)")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
