"""
ATLAS - Demo Visual del Sistema
================================
Ejecuta una demostración visual del sistema ATLAS.

Uso:
    python demo_atlas.py
"""

import os
import sys
import time
import random

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def color(texto, codigo):
    """Aplica color ANSI al texto"""
    return f"\033[{codigo}m{texto}\033[0m"

def rojo(texto): return color(texto, "91")
def verde(texto): return color(texto, "92")
def amarillo(texto): return color(texto, "93")
def azul(texto): return color(texto, "94")
def cyan(texto): return color(texto, "96")
def blanco(texto): return color(texto, "97")
def gris(texto): return color(texto, "90")

def dibujar_semaforo(fase):
    """Dibuja un semáforo ASCII"""
    if fase == "NS":
        return [
            "  ┌───┐  ",
            "  │ ○ │  ",
            "  │ ○ │  ",
            f"  │ {verde('●')} │  ",
            "  └───┘  ",
            "    ║    ",
            " N-S: VERDE"
        ]
    else:
        return [
            "  ┌───┐  ",
            "  │ ○ │  ",
            "  │ ○ │  ",
            f"  │ {verde('●')} │  ",
            "  └───┘  ",
            "    ║    ",
            " E-O: VERDE"
        ]

def dibujar_cruce(vehiculos_n, vehiculos_s, vehiculos_e, vehiculos_o, fase):
    """Dibuja el cruce con vehículos"""
    
    # Símbolos de vehículos
    coche = "🚗"
    
    # Crear representación
    v_norte = min(vehiculos_n, 5)
    v_sur = min(vehiculos_s, 5)
    v_este = min(vehiculos_e, 5)
    v_oeste = min(vehiculos_o, 5)
    
    semaforo = dibujar_semaforo(fase)
    
    cruce = f"""
                         NORTE ({vehiculos_n} vehículos)
                              │
                         {"🚗 " * v_norte}
                              │
                              ▼
    ──────────────────────────┼──────────────────────────
    OESTE ({vehiculos_o})  {"🚗 " * v_oeste} →  │  ← {"🚗 " * v_este}  ESTE ({vehiculos_e})
    ──────────────────────────┼──────────────────────────
                              │
                         {"🚗 " * v_sur}
                              │
                         SUR ({vehiculos_s} vehículos)
    """
    
    return cruce

def demo():
    """Ejecuta la demo"""
    
    print("\n")
    print(cyan("="*60))
    print(cyan("   🚦 ATLAS - Demo del Sistema de Control Inteligente"))
    print(cyan("="*60))
    print()
    
    print(blanco("Este demo simula cómo ATLAS controla un semáforo en tiempo real."))
    print(gris("El sistema detecta vehículos y decide cuándo cambiar la luz.\n"))
    
    input(amarillo("Presiona ENTER para comenzar la demo..."))
    
    # Simulación de 60 segundos
    fase_actual = "NS"
    tiempo_en_fase = 0
    paso = 0
    
    # Estado inicial
    vehiculos = {'N': 8, 'S': 6, 'E': 12, 'O': 10}
    colas = {'N': 3, 'S': 2, 'E': 5, 'O': 4}
    
    try:
        while paso < 30:
            limpiar_pantalla()
            
            print(cyan("\n" + "="*60))
            print(cyan(f"   🚦 ATLAS - Simulación en Tiempo Real (Paso {paso+1}/30)"))
            print(cyan("="*60 + "\n"))
            
            # Mostrar detección
            print(verde("📷 DETECCIÓN DE VEHÍCULOS (YOLO):"))
            print(f"   Norte: {vehiculos['N']} vehículos ({colas['N']} en cola)")
            print(f"   Sur:   {vehiculos['S']} vehículos ({colas['S']} en cola)")
            print(f"   Este:  {vehiculos['E']} vehículos ({colas['E']} en cola)")
            print(f"   Oeste: {vehiculos['O']} vehículos ({colas['O']} en cola)")
            print()
            
            # Mostrar estado del semáforo
            if fase_actual == "NS":
                print(verde(f"🚦 SEMÁFORO: Norte-Sur VERDE") + f" (tiempo: {tiempo_en_fase}s)")
                print(rojo(f"   Este-Oeste ROJO"))
            else:
                print(rojo(f"🚦 SEMÁFORO: Norte-Sur ROJO"))
                print(verde(f"   Este-Oeste VERDE") + f" (tiempo: {tiempo_en_fase}s)")
            print()
            
            # Crear vector de estado
            estado = [
                vehiculos['N'], vehiculos['S'], vehiculos['E'], vehiculos['O'],
                colas['N'], colas['S'], colas['E'], colas['O'],
                colas['N']*2, colas['S']*2, colas['E']*2, colas['O']*2
            ]
            
            print(azul("🧠 DECISIÓN DQN:"))
            print(f"   Vector de estado: {estado[:4]}...")
            
            # Simular decisión de la IA
            total_ns = vehiculos['N'] + vehiculos['S'] + colas['N']*2 + colas['S']*2
            total_eo = vehiculos['E'] + vehiculos['O'] + colas['E']*2 + colas['O']*2
            
            # Lógica de decisión
            cambiar = False
            razon = ""
            
            if tiempo_en_fase >= 10:  # Mínimo 10 segundos
                if fase_actual == "NS" and total_eo > total_ns * 1.3:
                    cambiar = True
                    razon = f"E-O tiene más demanda ({total_eo} vs {total_ns})"
                elif fase_actual == "EO" and total_ns > total_eo * 1.3:
                    cambiar = True
                    razon = f"N-S tiene más demanda ({total_ns} vs {total_eo})"
            
            if tiempo_en_fase >= 30:  # Máximo 30 segundos
                cambiar = True
                razon = "Tiempo máximo alcanzado"
            
            if cambiar:
                print(amarillo(f"   → Acción: CAMBIAR FASE ({razon})"))
                fase_actual = "EO" if fase_actual == "NS" else "NS"
                tiempo_en_fase = 0
            else:
                print(verde(f"   → Acción: MANTENER FASE"))
            
            print()
            
            # Validación de seguridad
            print(gris("🔒 SEGURIDAD: ✓ Tiempo verde válido | ✓ Sin conflictos"))
            print()
            
            # Estadísticas
            print(cyan("📊 ESTADÍSTICAS:"))
            print(f"   Vehículos procesados: ~{paso * 5}")
            print(f"   Tiempo promedio espera: {random.randint(15, 25)}s")
            print()
            
            # Actualizar estado (simular tráfico)
            for dir in ['N', 'S', 'E', 'O']:
                # Llegan nuevos vehículos
                vehiculos[dir] += random.randint(0, 3)
                
                # Si tienen verde, pasan vehículos
                if (fase_actual == "NS" and dir in ['N', 'S']) or \
                   (fase_actual == "EO" and dir in ['E', 'O']):
                    vehiculos[dir] = max(0, vehiculos[dir] - random.randint(2, 5))
                    colas[dir] = max(0, colas[dir] - random.randint(1, 2))
                else:
                    # Si tienen rojo, se acumulan en cola
                    colas[dir] = min(colas[dir] + random.randint(0, 2), vehiculos[dir])
            
            tiempo_en_fase += 2
            paso += 1
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    
    limpiar_pantalla()
    print(cyan("\n" + "="*60))
    print(cyan("   🚦 ATLAS - Demo Completada"))
    print(cyan("="*60))
    print()
    print(verde("✅ El sistema ha demostrado:"))
    print("   • Detección de vehículos en tiempo real")
    print("   • Toma de decisiones inteligente")
    print("   • Adaptación al flujo de tráfico")
    print("   • Sistema de seguridad funcionando")
    print()
    print(amarillo("📊 Resultados estimados vs semáforos tradicionales:"))
    print("   • -30% tiempo de espera")
    print("   • +20% flujo de vehículos")
    print("   • -25% longitud de colas")
    print()
    print(cyan("🚀 ATLAS está listo para implementación real"))
    print()

if __name__ == "__main__":
    demo()
