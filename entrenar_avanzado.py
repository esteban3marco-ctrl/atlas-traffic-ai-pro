"""
ATLAS - Entrenamiento Avanzado con Redes Neuronales
====================================================
Sistema de aprendizaje profundo (DQN) para control de semáforos.
Incluye:
- Red neuronal que aprende de la experiencia
- Múltiples tipos de cruces
- Diferentes vehículos (coches, buses, motos, emergencias)
- Memoria de experiencias para aprendizaje eficiente
"""

import os
import sys
import time
import random
import numpy as np
from collections import deque

try:
    import traci
except ImportError:
    print("ERROR: Instala traci con: pip install traci")
    sys.exit(1)


# =============================================================================
# RED NEURONAL (DQN - Deep Q-Network)
# =============================================================================

class RedNeuronal:
    """
    Red neuronal simple para aprender a controlar semáforos.
    Arquitectura: Estado -> [128] -> [128] -> [64] -> Acciones
    """
    
    def __init__(self, estado_dim, acciones_dim):
        self.estado_dim = estado_dim
        self.acciones_dim = acciones_dim
        
        # Arquitectura de la red
        self.capas = [estado_dim, 128, 128, 64, acciones_dim]
        
        # Inicializar pesos aleatorios
        self.pesos = []
        self.biases = []
        
        for i in range(len(self.capas) - 1):
            # Inicialización He
            w = np.random.randn(self.capas[i], self.capas[i+1]) * np.sqrt(2.0 / self.capas[i])
            b = np.zeros(self.capas[i+1])
            self.pesos.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        """Propagación hacia adelante"""
        self.activaciones = [x]
        
        for i, (w, b) in enumerate(zip(self.pesos, self.biases)):
            x = np.dot(x, w) + b
            # ReLU en capas ocultas, lineal en la última
            if i < len(self.pesos) - 1:
                x = np.maximum(0, x)  # ReLU
            self.activaciones.append(x)
        
        return x
    
    def predecir(self, estado):
        """Predice valores Q para cada acción"""
        return self.forward(estado)
    
    def obtener_accion(self, estado, epsilon):
        """Selecciona acción con estrategia epsilon-greedy"""
        if random.random() < epsilon:
            return random.randint(0, self.acciones_dim - 1)
        else:
            q_valores = self.predecir(estado)
            return np.argmax(q_valores)
    
    def entrenar(self, estado, accion, objetivo, learning_rate=0.001):
        """Entrena la red con backpropagation"""
        # Forward
        q_valores = self.forward(estado)
        
        # Calcular error solo para la acción tomada
        error = np.zeros(self.acciones_dim)
        error[accion] = objetivo - q_valores[accion]
        
        # Backpropagation
        delta = error
        for i in range(len(self.pesos) - 1, -1, -1):
            # Gradiente de ReLU
            if i < len(self.pesos) - 1:
                delta = delta * (self.activaciones[i+1] > 0).astype(float)
            
            # Actualizar pesos
            grad_w = np.outer(self.activaciones[i], delta)
            self.pesos[i] += learning_rate * grad_w
            self.biases[i] += learning_rate * delta
            
            # Propagar error
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T)
        
        return abs(error[accion])
    
    def copiar_pesos_de(self, otra_red):
        """Copia pesos de otra red (para target network)"""
        for i in range(len(self.pesos)):
            self.pesos[i] = otra_red.pesos[i].copy()
            self.biases[i] = otra_red.biases[i].copy()
    
    def guardar(self, archivo):
        """Guarda los pesos en un archivo"""
        np.savez(archivo, 
                 *self.pesos, 
                 *self.biases,
                 capas=self.capas)
        print(f"💾 Modelo guardado en {archivo}")
    
    def cargar(self, archivo):
        """Carga los pesos desde un archivo"""
        if os.path.exists(archivo):
            data = np.load(archivo, allow_pickle=True)
            n_capas = len(self.pesos)
            for i in range(n_capas):
                self.pesos[i] = data[f'arr_{i}']
                self.biases[i] = data[f'arr_{i + n_capas}']
            print(f"📂 Modelo cargado desde {archivo}")
            return True
        return False


# =============================================================================
# MEMORIA DE EXPERIENCIAS (Experience Replay)
# =============================================================================

class MemoriaExperiencias:
    """
    Almacena experiencias pasadas para entrenar la red.
    Permite aprender de experiencias aleatorias (rompe correlación).
    """
    
    def __init__(self, capacidad=50000):
        self.memoria = deque(maxlen=capacidad)
    
    def guardar(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Guarda una experiencia"""
        self.memoria.append((estado, accion, recompensa, siguiente_estado, terminado))
    
    def muestrear(self, batch_size):
        """Obtiene un batch aleatorio de experiencias"""
        return random.sample(self.memoria, min(batch_size, len(self.memoria)))
    
    def __len__(self):
        return len(self.memoria)


# =============================================================================
# AGENTE DQN
# =============================================================================

class AgenteDQN:
    """
    Agente que aprende a controlar semáforos usando Deep Q-Learning.
    """
    
    # Acciones posibles
    ACCIONES = ['mantener', 'cambiar_ns', 'cambiar_eo', 'extender']
    
    def __init__(self, estado_dim=12):
        self.estado_dim = estado_dim
        self.acciones_dim = len(self.ACCIONES)
        
        # Redes neuronales (principal y objetivo)
        self.red_principal = RedNeuronal(estado_dim, self.acciones_dim)
        self.red_objetivo = RedNeuronal(estado_dim, self.acciones_dim)
        self.red_objetivo.copiar_pesos_de(self.red_principal)
        
        # Memoria
        self.memoria = MemoriaExperiencias(capacidad=50000)
        
        # Hiperparámetros
        self.gamma = 0.95          # Factor de descuento
        self.epsilon = 1.0         # Exploración inicial
        self.epsilon_min = 0.05    # Exploración mínima
        self.epsilon_decay = 0.997 # Decaimiento de exploración
        self.learning_rate = 0.001
        self.batch_size = 32
        self.actualizar_objetivo_cada = 100
        
        # Contadores
        self.pasos_entrenamiento = 0
    
    def obtener_accion(self, estado):
        """Selecciona una acción"""
        return self.red_principal.obtener_accion(estado, self.epsilon)
    
    def recordar(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Guarda experiencia en memoria"""
        self.memoria.guardar(estado, accion, recompensa, siguiente_estado, terminado)
    
    def entrenar(self):
        """Entrena la red con un batch de experiencias"""
        if len(self.memoria) < self.batch_size:
            return 0
        
        # Obtener batch aleatorio
        batch = self.memoria.muestrear(self.batch_size)
        
        error_total = 0
        for estado, accion, recompensa, sig_estado, terminado in batch:
            # Calcular objetivo Q
            if terminado:
                objetivo = recompensa
            else:
                # Q-learning: r + γ * max(Q(s', a'))
                q_futuro = np.max(self.red_objetivo.predecir(sig_estado))
                objetivo = recompensa + self.gamma * q_futuro
            
            # Entrenar red principal
            error = self.red_principal.entrenar(estado, accion, objetivo, self.learning_rate)
            error_total += error
        
        # Actualizar red objetivo periódicamente
        self.pasos_entrenamiento += 1
        if self.pasos_entrenamiento % self.actualizar_objetivo_cada == 0:
            self.red_objetivo.copiar_pesos_de(self.red_principal)
        
        # Reducir exploración
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return error_total / self.batch_size
    
    def guardar(self, archivo):
        """Guarda el modelo"""
        self.red_principal.guardar(archivo)
    
    def cargar(self, archivo):
        """Carga el modelo"""
        if self.red_principal.cargar(archivo):
            self.red_objetivo.copiar_pesos_de(self.red_principal)
            self.epsilon = self.epsilon_min  # Ya entrenado
            return True
        return False


# =============================================================================
# ENTORNO SUMO AVANZADO
# =============================================================================

class EntornoSUMO:
    """
    Entorno de simulación SUMO con soporte para múltiples cruces.
    """
    
    def __init__(self, config_file, gui=False):
        self.config_file = config_file
        self.gui = gui
        self.conectado = False
        self.semaforo_id = "center"
        
        # Estado
        self.fase_actual = 0
        self.pasos_en_fase = 0
        self.paso_actual = 0
        
        # Configuración de fases
        self.MIN_VERDE = 100   # 10 segundos
        self.MAX_VERDE = 600   # 60 segundos
        
        # Edges por dirección
        self.edges_entrada = {
            'N': 'north_in',
            'S': 'south_in',
            'E': 'east_in',
            'W': 'west_in'
        }
    
    def conectar(self):
        """Conecta con SUMO"""
        sumo_cmd = "sumo-gui" if self.gui else "sumo"
        
        cmd = [
            sumo_cmd,
            "-c", self.config_file,
            "--step-length", "0.1",
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000"
        ]
        
        traci.start(cmd)
        self.conectado = True
        self.paso_actual = 0
        self.fase_actual = 0
        self.pasos_en_fase = 0
    
    def desconectar(self):
        """Desconecta de SUMO"""
        if self.conectado:
            traci.close()
            self.conectado = False
    
    def obtener_estado(self):
        """
        Obtiene el estado actual como vector numérico.
        [colas_N, colas_S, colas_E, colas_W, 
         velocidad_N, velocidad_S, velocidad_E, velocidad_W,
         espera_N, espera_S, espera_E, espera_W]
        """
        estado = []
        
        for direccion in ['N', 'S', 'E', 'W']:
            edge = self.edges_entrada[direccion]
            
            # Cola (vehículos parados)
            try:
                cola = 0
                for i in range(2):
                    cola += traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                estado.append(cola / 20.0)  # Normalizar
            except:
                estado.append(0)
            
            # Velocidad media
            try:
                velocidad = traci.edge.getLastStepMeanSpeed(edge)
                estado.append(velocidad / 15.0)  # Normalizar
            except:
                estado.append(0)
            
            # Tiempo de espera
            try:
                espera = traci.edge.getWaitingTime(edge)
                estado.append(min(espera, 300) / 300.0)  # Normalizar, max 5 min
            except:
                estado.append(0)
        
        return np.array(estado, dtype=np.float32)
    
    def ejecutar_accion(self, accion):
        """
        Ejecuta una acción del agente.
        0: Mantener fase actual
        1: Cambiar a fase N-S
        2: Cambiar a fase E-O
        3: Extender fase actual
        """
        if accion == 0:  # Mantener
            pass
        
        elif accion == 1:  # Cambiar a N-S
            if self.fase_actual != 0 and self.pasos_en_fase >= self.MIN_VERDE:
                self._cambiar_fase(0)
        
        elif accion == 2:  # Cambiar a E-O
            if self.fase_actual != 2 and self.pasos_en_fase >= self.MIN_VERDE:
                self._cambiar_fase(2)
        
        elif accion == 3:  # Extender
            pass  # Solo no cambiar
        
        # Forzar cambio si excede máximo
        if self.pasos_en_fase >= self.MAX_VERDE:
            nueva_fase = 2 if self.fase_actual == 0 else 0
            self._cambiar_fase(nueva_fase)
    
    def _cambiar_fase(self, nueva_fase):
        """Cambia la fase del semáforo con amarillo"""
        # Fase amarillo
        traci.trafficlight.setPhase(self.semaforo_id, self.fase_actual + 1)
        for _ in range(40):  # 4 segundos
            traci.simulationStep()
            self.paso_actual += 1
        
        # Nueva fase
        traci.trafficlight.setPhase(self.semaforo_id, nueva_fase)
        self.fase_actual = nueva_fase
        self.pasos_en_fase = 0
    
    def paso(self):
        """Avanza un paso de simulación"""
        traci.simulationStep()
        self.paso_actual += 1
        self.pasos_en_fase += 1
    
    def calcular_recompensa(self):
        """
        Calcula la recompensa basada en:
        - Penalizar colas largas
        - Penalizar tiempos de espera
        - Recompensar throughput
        """
        recompensa = 0
        
        for edge in self.edges_entrada.values():
            try:
                # Penalizar colas
                for i in range(2):
                    cola = traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                    recompensa -= cola * 0.5
                
                # Penalizar tiempo de espera
                espera = traci.edge.getWaitingTime(edge)
                recompensa -= espera * 0.1
                
                # Recompensar velocidad (flujo)
                velocidad = traci.edge.getLastStepMeanSpeed(edge)
                recompensa += velocidad * 0.3
            except:
                pass
        
        # Recompensar vehículos que llegan a destino
        llegados = traci.simulation.getArrivedNumber()
        recompensa += llegados * 2.0
        
        return recompensa
    
    def esta_activo(self):
        """Verifica si la simulación sigue activa"""
        return traci.simulation.getMinExpectedNumber() > 0
    
    def obtener_info(self):
        """Obtiene información adicional"""
        return {
            'paso': self.paso_actual,
            'fase': 'N-S' if self.fase_actual == 0 else 'E-O',
            'vehiculos': len(traci.vehicle.getIDList()),
            'llegados': traci.simulation.getArrivedNumber()
        }


# =============================================================================
# FUNCIÓN DE ENTRENAMIENTO
# =============================================================================

def entrenar_avanzado(
    num_episodios=100,
    tipo_cruce='simple',
    usar_gui=False,
    cargar_modelo=True,
    guardar_cada=10
):
    """
    Entrena el agente DQN.
    
    Args:
        num_episodios: Número de episodios de entrenamiento
        tipo_cruce: 'simple', 'doble', 'complejo'
        usar_gui: Mostrar interfaz gráfica
        cargar_modelo: Cargar modelo previo si existe
        guardar_cada: Guardar modelo cada N episodios
    """
    
    print()
    print("=" * 70)
    print("🧠 ATLAS - Entrenamiento Avanzado con Redes Neuronales (DQN)")
    print("=" * 70)
    print(f"📋 Configuración:")
    print(f"   • Episodios: {num_episodios}")
    print(f"   • Tipo de cruce: {tipo_cruce}")
    print(f"   • Red neuronal: [12] → [128] → [128] → [64] → [4]")
    print(f"   • Modo: {'Con GUI' if usar_gui else 'Sin GUI (rápido)'}")
    print("=" * 70)
    print()
    
    # Crear carpetas necesarias
    os.makedirs("modelos", exist_ok=True)
    
    # Configuración según tipo de cruce
    config_file = "simulations/sumo_config/simulation.sumocfg"
    
    # Crear agente
    agente = AgenteDQN(estado_dim=12)
    
    # Cargar modelo previo si existe
    modelo_archivo = f"modelos/agente_dqn_{tipo_cruce}.npz"
    if cargar_modelo:
        if agente.cargar(modelo_archivo):
            print("✅ Continuando entrenamiento desde modelo guardado")
        else:
            print("🆕 Iniciando entrenamiento desde cero")
    print()
    
    # Crear entorno
    entorno = EntornoSUMO(config_file, gui=usar_gui)
    
    # Métricas
    recompensas_episodios = []
    mejor_recompensa = float('-inf')
    
    try:
        for episodio in range(num_episodios):
            print(f"\n{'='*60}")
            print(f"📍 EPISODIO {episodio + 1} / {num_episodios}")
            print(f"   ε = {agente.epsilon:.3f} (exploración)")
            print(f"{'='*60}")
            
            # Reiniciar entorno
            entorno.conectar()
            
            # Variables del episodio
            recompensa_total = 0
            pasos = 0
            errores = []
            
            tiempo_inicio = time.time()
            
            # Obtener estado inicial
            estado = entorno.obtener_estado()
            
            # Loop principal del episodio
            while entorno.esta_activo() and pasos < 36000:  # Max 1 hora simulada
                
                # Seleccionar acción
                accion = agente.obtener_accion(estado)
                
                # Ejecutar acción
                entorno.ejecutar_accion(accion)
                
                # Avanzar simulación (10 pasos = 1 segundo)
                for _ in range(10):
                    entorno.paso()
                pasos += 10
                
                # Obtener nuevo estado y recompensa
                nuevo_estado = entorno.obtener_estado()
                recompensa = entorno.calcular_recompensa()
                terminado = not entorno.esta_activo()
                
                # Guardar experiencia
                agente.recordar(estado, accion, recompensa, nuevo_estado, terminado)
                
                # Entrenar
                error = agente.entrenar()
                if error > 0:
                    errores.append(error)
                
                recompensa_total += recompensa
                estado = nuevo_estado
                
                # Mostrar progreso
                if pasos % 5000 == 0:
                    info = entorno.obtener_info()
                    print(f"  ⏱️  {pasos//10}s | "
                          f"Fase: {info['fase']} 🟢 | "
                          f"Vehículos: {info['vehiculos']} | "
                          f"Reward: {recompensa_total:.0f} | "
                          f"ε: {agente.epsilon:.3f}")
            
            # Fin del episodio
            entorno.desconectar()
            
            tiempo_real = time.time() - tiempo_inicio
            recompensas_episodios.append(recompensa_total)
            
            # Estadísticas
            error_medio = np.mean(errores) if errores else 0
            
            print(f"\n📊 RESUMEN EPISODIO {episodio + 1}:")
            print(f"   ├─ Recompensa: {recompensa_total:.0f}")
            print(f"   ├─ Error medio: {error_medio:.4f}")
            print(f"   ├─ Epsilon: {agente.epsilon:.3f}")
            print(f"   ├─ Experiencias: {len(agente.memoria)}")
            print(f"   └─ Tiempo: {tiempo_real:.1f}s")
            
            # Guardar mejor modelo
            if recompensa_total > mejor_recompensa:
                mejor_recompensa = recompensa_total
                agente.guardar(f"modelos/mejor_agente_{tipo_cruce}.npz")
                print(f"   🏆 ¡Nuevo mejor modelo!")
            
            # Guardar checkpoint
            if (episodio + 1) % guardar_cada == 0:
                agente.guardar(modelo_archivo)
            
            # Mostrar progreso de aprendizaje
            if len(recompensas_episodios) >= 5:
                media_reciente = np.mean(recompensas_episodios[-5:])
                media_antigua = np.mean(recompensas_episodios[:5]) if len(recompensas_episodios) > 5 else media_reciente
                mejora = media_reciente - media_antigua
                print(f"   📈 Mejora vs inicio: {mejora:+.0f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido")
    
    finally:
        if entorno.conectado:
            entorno.desconectar()
        
        # Guardar modelo final
        agente.guardar(modelo_archivo)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("🏁 ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
    if recompensas_episodios:
        print(f"📈 Estadísticas finales:")
        print(f"   ├─ Episodios: {len(recompensas_episodios)}")
        print(f"   ├─ Mejor recompensa: {max(recompensas_episodios):.0f}")
        print(f"   ├─ Peor recompensa: {min(recompensas_episodios):.0f}")
        print(f"   ├─ Media: {np.mean(recompensas_episodios):.0f}")
        print(f"   └─ Epsilon final: {agente.epsilon:.3f}")
        
        # Calcular mejora
        if len(recompensas_episodios) >= 10:
            inicio = np.mean(recompensas_episodios[:5])
            fin = np.mean(recompensas_episodios[-5:])
            mejora = ((fin - inicio) / abs(inicio)) * 100 if inicio != 0 else 0
            
            if mejora > 0:
                print(f"\n🎉 ¡La IA mejoró un {mejora:.1f}%!")
            else:
                print(f"\n📊 Cambio: {mejora:.1f}% (entrena más episodios)")
    
    print(f"\n💾 Modelo guardado en: {modelo_archivo}")
    print("=" * 70)
    print()
    
    return agente, recompensas_episodios


# =============================================================================
# ENTRENAMIENTO MULTI-ESCENARIO
# =============================================================================

def entrenar_multi_escenario(episodios_por_escenario=20, usar_gui=False):
    """
    Entrena la IA en múltiples escenarios para mejor generalización.
    """
    
    print()
    print("=" * 70)
    print("🧠 ATLAS - Entrenamiento Multi-Escenario")
    print("=" * 70)
    print()
    
    escenarios = [
        ("simple", "normal"),
        ("simple", "hora_punta_manana"),
        ("simple", "hora_punta_tarde"),
        ("simple", "noche"),
        ("simple", "emergencias"),
    ]
    
    # Crear agente que se usará en todos los escenarios
    agente = AgenteDQN(estado_dim=12)
    modelo_archivo = "modelos/agente_multi_escenario.npz"
    agente.cargar(modelo_archivo)
    
    todas_recompensas = []
    
    for cruce, escenario in escenarios:
        print(f"\n{'='*60}")
        print(f"📍 Entrenando en: {cruce} - {escenario}")
        print(f"{'='*60}")
        
        # Determinar archivo de configuración
        if escenario == "normal":
            config_file = f"simulations/{cruce}/simulation.sumocfg"
        else:
            config_file = f"simulations/{cruce}_{escenario}/simulation.sumocfg"
        
        if not os.path.exists(config_file):
            print(f"   ⚠️ No existe {config_file}, saltando...")
            continue
        
        entorno = EntornoSUMO(config_file, gui=usar_gui)
        
        for ep in range(episodios_por_escenario):
            entorno.conectar()
            estado = entorno.obtener_estado()
            recompensa_total = 0
            pasos = 0
            
            while entorno.esta_activo() and pasos < 36000:
                accion = agente.obtener_accion(estado)
                entorno.ejecutar_accion(accion)
                
                for _ in range(10):
                    entorno.paso()
                pasos += 10
                
                nuevo_estado = entorno.obtener_estado()
                recompensa = entorno.calcular_recompensa()
                
                agente.recordar(estado, accion, recompensa, nuevo_estado, False)
                agente.entrenar()
                
                recompensa_total += recompensa
                estado = nuevo_estado
            
            entorno.desconectar()
            todas_recompensas.append(recompensa_total)
            
            print(f"   Ep {ep+1}/{episodios_por_escenario} | "
                  f"Reward: {recompensa_total:.0f} | "
                  f"ε: {agente.epsilon:.3f}")
        
        # Guardar después de cada escenario
        agente.guardar(modelo_archivo)
    
    print()
    print("=" * 70)
    print("🏁 ENTRENAMIENTO MULTI-ESCENARIO COMPLETADO")
    print("=" * 70)
    print(f"   Total episodios: {len(todas_recompensas)}")
    print(f"   Mejor reward: {max(todas_recompensas):.0f}")
    print(f"   Media reward: {np.mean(todas_recompensas):.0f}")
    print(f"   Modelo guardado: {modelo_archivo}")
    print()


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🧠 ATLAS - Entrenamiento Avanzado con Redes Neuronales',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python entrenar_avanzado.py                              # 50 episodios, cruce simple
  python entrenar_avanzado.py --episodios 100              # 100 episodios
  python entrenar_avanzado.py --gui                        # Con visualización
  python entrenar_avanzado.py --cruce avenida              # Entrenar en avenida
  python entrenar_avanzado.py --escenario hora_punta_manana # Hora punta
  python entrenar_avanzado.py --multi                      # Entrenar en TODOS los escenarios

Cruces disponibles:
  simple, doble, avenida, cruce_t, complejo, grid

Escenarios disponibles (para cruce simple):
  normal, hora_punta_manana, hora_punta_tarde, noche, evento, emergencias
        """
    )
    
    parser.add_argument(
        '--episodios', 
        type=int, 
        default=50,
        help='Número de episodios (default: 50)'
    )
    
    parser.add_argument(
        '--cruce',
        type=str,
        default='simple',
        choices=['simple', 'doble', 'avenida', 'cruce_t', 'complejo', 'grid'],
        help='Tipo de cruce (default: simple)'
    )
    
    parser.add_argument(
        '--escenario',
        type=str,
        default='normal',
        choices=['normal', 'hora_punta_manana', 'hora_punta_tarde', 'noche', 'evento', 'emergencias'],
        help='Escenario de tráfico (default: normal)'
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Mostrar interfaz gráfica'
    )
    
    parser.add_argument(
        '--nuevo',
        action='store_true',
        help='Empezar entrenamiento desde cero'
    )
    
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Entrenar en múltiples escenarios (mejor generalización)'
    )
    
    args = parser.parse_args()
    
    if args.multi:
        entrenar_multi_escenario(
            episodios_por_escenario=args.episodios // 5,
            usar_gui=args.gui
        )
    else:
        entrenar_avanzado(
            num_episodios=args.episodios,
            tipo_cruce=args.cruce,
            usar_gui=args.gui,
            cargar_modelo=not args.nuevo
        )
