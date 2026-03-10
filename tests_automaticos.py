"""
ATLAS - Tests Automáticos
==========================
Ejecuta todos los tests: python tests_automaticos.py
Tests individuales: python tests_automaticos.py --test seguridad

Este módulo verifica que el sistema funciona correctamente antes de
implementarlo en un entorno real.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Colores para la terminal
class Colores:
    OK = '\033[92m'      # Verde
    FAIL = '\033[91m'    # Rojo
    WARN = '\033[93m'    # Amarillo
    INFO = '\033[94m'    # Azul
    END = '\033[0m'      # Reset

def ok(msg):
    print(f"  {Colores.OK}✅ PASS:{Colores.END} {msg}")

def fail(msg):
    print(f"  {Colores.FAIL}❌ FAIL:{Colores.END} {msg}")

def warn(msg):
    print(f"  {Colores.WARN}⚠️  WARN:{Colores.END} {msg}")

def info(msg):
    print(f"  {Colores.INFO}ℹ️  INFO:{Colores.END} {msg}")


class ResultadosTest:
    """Almacena resultados de los tests"""
    def __init__(self):
        self.total = 0
        self.pasados = 0
        self.fallidos = 0
        self.warnings = 0
        self.detalles = []
    
    def agregar(self, nombre, pasado, mensaje=""):
        self.total += 1
        if pasado:
            self.pasados += 1
            self.detalles.append(("PASS", nombre, mensaje))
        else:
            self.fallidos += 1
            self.detalles.append(("FAIL", nombre, mensaje))
    
    def agregar_warning(self, nombre, mensaje):
        self.warnings += 1
        self.detalles.append(("WARN", nombre, mensaje))


# =============================================================================
# TEST 1: VERIFICAR ARCHIVOS Y ESTRUCTURA
# =============================================================================

def test_estructura_archivos(resultados):
    """Verifica que todos los archivos necesarios existen"""
    
    print("\n" + "="*60)
    print("📁 TEST 1: Estructura de archivos")
    print("="*60)
    
    archivos_requeridos = [
        "entrenar_avanzado.py",
        "crear_todo.py",
        "simulations/simple/simulation.sumocfg",
        "simulations/simple/intersection.net.xml",
        "simulations/simple/traffic.rou.xml",
    ]
    
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            ok(f"Existe: {archivo}")
            resultados.agregar(f"Archivo {archivo}", True)
        else:
            fail(f"No existe: {archivo}")
            resultados.agregar(f"Archivo {archivo}", False, "Archivo no encontrado")
    
    # Verificar carpeta de modelos
    if os.path.exists("modelos"):
        modelos = [f for f in os.listdir("modelos") if f.endswith('.npz')]
        if len(modelos) > 0:
            ok(f"Modelos entrenados: {len(modelos)} encontrados")
            resultados.agregar("Modelos entrenados", True)
        else:
            warn("No hay modelos entrenados")
            resultados.agregar_warning("Modelos entrenados", "Ningún modelo encontrado")
    else:
        fail("Carpeta 'modelos' no existe")
        resultados.agregar("Carpeta modelos", False)


# =============================================================================
# TEST 2: VERIFICAR RED NEURONAL
# =============================================================================

def test_red_neuronal(resultados):
    """Verifica que la red neuronal funciona correctamente"""
    
    print("\n" + "="*60)
    print("🧠 TEST 2: Red Neuronal")
    print("="*60)
    
    try:
        # Importar la red neuronal del script principal
        sys.path.insert(0, '.')
        from entrenar_avanzado import RedNeuronal, AgenteDQN
        
        ok("Importación de módulos correcta")
        resultados.agregar("Importar RedNeuronal", True)
        
        # Test: Crear red neuronal
        red = RedNeuronal(estado_dim=12, acciones_dim=4)
        ok("Red neuronal creada correctamente")
        resultados.agregar("Crear red neuronal", True)
        
        # Test: Forward pass
        estado = np.random.rand(12).astype(np.float32)
        salida = red.forward(estado)
        
        if salida.shape == (4,):
            ok(f"Forward pass: entrada(12) → salida(4)")
            resultados.agregar("Forward pass", True)
        else:
            fail(f"Forward pass incorrecto: salida shape = {salida.shape}")
            resultados.agregar("Forward pass", False)
        
        # Test: Salida no tiene NaN
        if not np.any(np.isnan(salida)):
            ok("Salida sin valores NaN")
            resultados.agregar("Sin NaN en salida", True)
        else:
            fail("Salida contiene NaN")
            resultados.agregar("Sin NaN en salida", False, "Valores NaN detectados")
        
        # Test: Salida no tiene infinitos
        if not np.any(np.isinf(salida)):
            ok("Salida sin valores infinitos")
            resultados.agregar("Sin Inf en salida", True)
        else:
            fail("Salida contiene infinitos")
            resultados.agregar("Sin Inf en salida", False, "Valores infinitos detectados")
        
        # Test: Crear agente DQN
        agente = AgenteDQN(estado_dim=12)
        ok("Agente DQN creado correctamente")
        resultados.agregar("Crear AgenteDQN", True)
        
        # Test: Obtener acción
        accion = agente.obtener_accion(estado)
        if 0 <= accion <= 3:
            ok(f"Acción válida: {accion} (rango 0-3)")
            resultados.agregar("Obtener acción", True)
        else:
            fail(f"Acción inválida: {accion}")
            resultados.agregar("Obtener acción", False)
        
        # Test: Entrenar sin errores
        agente.recordar(estado, accion, -1.0, estado, False)
        for _ in range(50):
            agente.recordar(estado, accion, -1.0, estado, False)
        error = agente.entrenar()
        ok(f"Entrenamiento ejecutado (error: {error:.4f})")
        resultados.agregar("Entrenar agente", True)
        
    except Exception as e:
        fail(f"Error en test de red neuronal: {e}")
        resultados.agregar("Red neuronal", False, str(e))


# =============================================================================
# TEST 3: VERIFICAR MODELOS GUARDADOS
# =============================================================================

def test_modelos_guardados(resultados):
    """Verifica que los modelos guardados se pueden cargar"""
    
    print("\n" + "="*60)
    print("💾 TEST 3: Modelos Guardados")
    print("="*60)
    
    if not os.path.exists("modelos"):
        warn("Carpeta 'modelos' no existe")
        return
    
    modelos = [f for f in os.listdir("modelos") if f.endswith('.npz')]
    
    if len(modelos) == 0:
        warn("No hay modelos para verificar")
        return
    
    try:
        from entrenar_avanzado import AgenteDQN
        
        for modelo in modelos:
            ruta = f"modelos/{modelo}"
            try:
                agente = AgenteDQN(estado_dim=12)
                if agente.cargar(ruta):
                    # Verificar que funciona
                    estado = np.random.rand(12).astype(np.float32)
                    accion = agente.obtener_accion(estado)
                    
                    if 0 <= accion <= 3:
                        ok(f"Modelo {modelo}: carga OK, acción válida")
                        resultados.agregar(f"Modelo {modelo}", True)
                    else:
                        fail(f"Modelo {modelo}: acción inválida")
                        resultados.agregar(f"Modelo {modelo}", False)
                else:
                    fail(f"Modelo {modelo}: no se pudo cargar")
                    resultados.agregar(f"Modelo {modelo}", False)
            except Exception as e:
                fail(f"Modelo {modelo}: error - {e}")
                resultados.agregar(f"Modelo {modelo}", False, str(e))
                
    except Exception as e:
        fail(f"Error importando AgenteDQN: {e}")


# =============================================================================
# TEST 4: VERIFICAR LÍMITES DE SEGURIDAD
# =============================================================================

def test_limites_seguridad(resultados):
    """Verifica que los límites de seguridad están configurados"""
    
    print("\n" + "="*60)
    print("🔒 TEST 4: Límites de Seguridad")
    print("="*60)
    
    try:
        from entrenar_avanzado import EntornoSUMO
        
        # Crear entorno (sin conectar a SUMO)
        entorno = EntornoSUMO("simulations/simple/simulation.sumocfg", gui=False)
        
        # Verificar límites
        if hasattr(entorno, 'MIN_VERDE'):
            if entorno.MIN_VERDE >= 50:  # 5 segundos mínimo
                ok(f"Tiempo mínimo verde: {entorno.MIN_VERDE/10}s (≥5s)")
                resultados.agregar("Mínimo verde", True)
            else:
                fail(f"Tiempo mínimo verde muy bajo: {entorno.MIN_VERDE/10}s")
                resultados.agregar("Mínimo verde", False, "Menos de 5 segundos")
        else:
            fail("MIN_VERDE no definido")
            resultados.agregar("MIN_VERDE definido", False)
        
        if hasattr(entorno, 'MAX_VERDE'):
            if entorno.MAX_VERDE <= 1200:  # 120 segundos máximo
                ok(f"Tiempo máximo verde: {entorno.MAX_VERDE/10}s (≤120s)")
                resultados.agregar("Máximo verde", True)
            else:
                warn(f"Tiempo máximo verde muy alto: {entorno.MAX_VERDE/10}s")
                resultados.agregar_warning("Máximo verde", "Más de 120 segundos")
        else:
            fail("MAX_VERDE no definido")
            resultados.agregar("MAX_VERDE definido", False)
            
    except Exception as e:
        fail(f"Error verificando límites: {e}")
        resultados.agregar("Límites de seguridad", False, str(e))


# =============================================================================
# TEST 5: TEST DE ESTRÉS DE LA RED NEURONAL
# =============================================================================

def test_estres_red_neuronal(resultados):
    """Prueba la red neuronal con valores extremos"""
    
    print("\n" + "="*60)
    print("💪 TEST 5: Test de Estrés")
    print("="*60)
    
    try:
        from entrenar_avanzado import RedNeuronal
        
        red = RedNeuronal(estado_dim=12, acciones_dim=4)
        
        # Test con valores normales
        estado_normal = np.random.rand(12).astype(np.float32)
        salida = red.forward(estado_normal)
        if not np.any(np.isnan(salida)) and not np.any(np.isinf(salida)):
            ok("Valores normales: OK")
            resultados.agregar("Estrés: valores normales", True)
        else:
            fail("Valores normales: produce NaN/Inf")
            resultados.agregar("Estrés: valores normales", False)
        
        # Test con ceros
        estado_cero = np.zeros(12, dtype=np.float32)
        salida = red.forward(estado_cero)
        if not np.any(np.isnan(salida)) and not np.any(np.isinf(salida)):
            ok("Valores cero: OK")
            resultados.agregar("Estrés: valores cero", True)
        else:
            fail("Valores cero: produce NaN/Inf")
            resultados.agregar("Estrés: valores cero", False)
        
        # Test con valores altos
        estado_alto = np.ones(12, dtype=np.float32) * 10
        salida = red.forward(estado_alto)
        if not np.any(np.isnan(salida)) and not np.any(np.isinf(salida)):
            ok("Valores altos (10): OK")
            resultados.agregar("Estrés: valores altos", True)
        else:
            warn("Valores altos (10): produce NaN/Inf")
            resultados.agregar_warning("Estrés: valores altos", "Produce NaN/Inf con valores altos")
        
        # Test con valores negativos
        estado_negativo = np.ones(12, dtype=np.float32) * -1
        salida = red.forward(estado_negativo)
        if not np.any(np.isnan(salida)) and not np.any(np.isinf(salida)):
            ok("Valores negativos: OK")
            resultados.agregar("Estrés: valores negativos", True)
        else:
            fail("Valores negativos: produce NaN/Inf")
            resultados.agregar("Estrés: valores negativos", False)
        
        # Test de velocidad (1000 inferencias)
        inicio = time.time()
        for _ in range(1000):
            red.forward(estado_normal)
        tiempo = time.time() - inicio
        
        if tiempo < 1.0:
            ok(f"Velocidad: 1000 inferencias en {tiempo*1000:.1f}ms")
            resultados.agregar("Estrés: velocidad", True)
        else:
            warn(f"Velocidad lenta: 1000 inferencias en {tiempo:.2f}s")
            resultados.agregar_warning("Estrés: velocidad", f"{tiempo:.2f}s para 1000 inferencias")
            
    except Exception as e:
        fail(f"Error en test de estrés: {e}")
        resultados.agregar("Test de estrés", False, str(e))


# =============================================================================
# TEST 6: VERIFICAR SIMULACIONES SUMO
# =============================================================================

def test_simulaciones_sumo(resultados):
    """Verifica que las simulaciones SUMO están bien configuradas"""
    
    print("\n" + "="*60)
    print("🚗 TEST 6: Simulaciones SUMO")
    print("="*60)
    
    escenarios = [
        "simple",
        "simple_hora_punta",
        "simple_noche",
        "simple_emergencias",
        "avenida",
        "cruce_t",
        "doble",
        "complejo",
        "evento"
    ]
    
    for escenario in escenarios:
        carpeta = f"simulations/{escenario}"
        
        if not os.path.exists(carpeta):
            warn(f"{escenario}: carpeta no existe")
            resultados.agregar_warning(f"SUMO {escenario}", "Carpeta no existe")
            continue
        
        archivos_necesarios = [
            "simulation.sumocfg",
            "intersection.net.xml",
            "traffic.rou.xml"
        ]
        
        todos_existen = True
        for archivo in archivos_necesarios:
            if not os.path.exists(f"{carpeta}/{archivo}"):
                todos_existen = False
                break
        
        if todos_existen:
            ok(f"{escenario}: todos los archivos presentes")
            resultados.agregar(f"SUMO {escenario}", True)
        else:
            fail(f"{escenario}: faltan archivos")
            resultados.agregar(f"SUMO {escenario}", False, "Archivos incompletos")


# =============================================================================
# TEST 7: TEST DE CONSISTENCIA DE ACCIONES
# =============================================================================

def test_consistencia_acciones(resultados):
    """Verifica que las acciones son consistentes"""
    
    print("\n" + "="*60)
    print("🎯 TEST 7: Consistencia de Acciones")
    print("="*60)
    
    try:
        from entrenar_avanzado import AgenteDQN
        
        agente = AgenteDQN(estado_dim=12)
        agente.epsilon = 0.0  # Sin exploración
        
        # Mismo estado debe dar misma acción
        estado = np.array([0.5] * 12, dtype=np.float32)
        
        acciones = []
        for _ in range(10):
            accion = agente.obtener_accion(estado)
            acciones.append(accion)
        
        if len(set(acciones)) == 1:
            ok(f"Acciones consistentes: siempre {acciones[0]}")
            resultados.agregar("Consistencia acciones", True)
        else:
            fail(f"Acciones inconsistentes: {acciones}")
            resultados.agregar("Consistencia acciones", False, "Diferentes acciones para mismo estado")
        
        # Verificar rango de acciones
        estados_random = [np.random.rand(12).astype(np.float32) for _ in range(100)]
        acciones = [agente.obtener_accion(e) for e in estados_random]
        
        if all(0 <= a <= 3 for a in acciones):
            ok("Todas las acciones en rango válido (0-3)")
            resultados.agregar("Rango acciones", True)
        else:
            fail(f"Acciones fuera de rango: {set(acciones)}")
            resultados.agregar("Rango acciones", False)
            
    except Exception as e:
        fail(f"Error en test de consistencia: {e}")
        resultados.agregar("Consistencia acciones", False, str(e))


# =============================================================================
# MAIN
# =============================================================================

def ejecutar_todos_los_tests():
    """Ejecuta todos los tests"""
    
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "🚦 ATLAS - TESTS AUTOMÁTICOS" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    print()
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio: {os.getcwd()}")
    
    resultados = ResultadosTest()
    
    # Ejecutar todos los tests
    test_estructura_archivos(resultados)
    test_red_neuronal(resultados)
    test_modelos_guardados(resultados)
    test_limites_seguridad(resultados)
    test_estres_red_neuronal(resultados)
    test_simulaciones_sumo(resultados)
    test_consistencia_acciones(resultados)
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE TESTS")
    print("="*60)
    print()
    print(f"  Total tests:    {resultados.total}")
    print(f"  {Colores.OK}Pasados:{Colores.END}        {resultados.pasados}")
    print(f"  {Colores.FAIL}Fallidos:{Colores.END}       {resultados.fallidos}")
    print(f"  {Colores.WARN}Warnings:{Colores.END}       {resultados.warnings}")
    print()
    
    if resultados.fallidos == 0:
        print(f"  {Colores.OK}✅ TODOS LOS TESTS PASARON{Colores.END}")
        print()
        print("  El sistema está listo para producción.")
    else:
        print(f"  {Colores.FAIL}❌ HAY {resultados.fallidos} TESTS FALLIDOS{Colores.END}")
        print()
        print("  Corrige los errores antes de implementar.")
    
    print()
    print("="*60)
    
    return resultados


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ATLAS - Tests Automáticos')
    parser.add_argument('--test', type=str, help='Ejecutar test específico')
    
    args = parser.parse_args()
    
    if args.test:
        # Ejecutar test específico
        resultados = ResultadosTest()
        tests = {
            'archivos': test_estructura_archivos,
            'red': test_red_neuronal,
            'modelos': test_modelos_guardados,
            'seguridad': test_limites_seguridad,
            'estres': test_estres_red_neuronal,
            'sumo': test_simulaciones_sumo,
            'consistencia': test_consistencia_acciones
        }
        
        if args.test in tests:
            tests[args.test](resultados)
        else:
            print(f"Test '{args.test}' no encontrado.")
            print(f"Tests disponibles: {', '.join(tests.keys())}")
    else:
        ejecutar_todos_los_tests()
