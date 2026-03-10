
# 🚦 ATLAS Pro: Smart City Traffic & Coordination (TRL 7)

Este documento sirve como material de apoyo para la presentación del proyecto ATLAS Pro, detallando las capacidades de producción y la arquitectura del sistema.

## 1. El Centro de Control Geoespacial (Dashboard)
El dashboard implementado no es solo visual; es el nexo de unión entre el Hardware Real y el Motor de Inferencia.
- **Mapa en Tiempo Real**: Integración con Leaflet.js para posicionar semáforos en la ciudad (Madrid, Gran Vía en la demo). Los marcadores cambian de color (Rojo/Verde) dinámicamente según la decisión de la IA.
- **Diagnóstico Conversacional (XAI)**: Una IA secundaria traduce los tensores de activación en explicaciones textuales: *"Priorizando eje Norte porque se ha detectado un vehículo de emergencia detenido"*.
- **Mapa de Atención (Saliency)**: Visualización técnica de qué sensores (cámaras) están influyendo más en la decisión actual (Transformer Attention).

## 2. Motor de Inteligencia MARL (QMIX)
A diferencia de los sistemas tradicionales que controlan un solo cruce, ATLAS Pro utiliza **Multi-Agent Reinforcement Learning**:
- **Coordinación QMIX**: Los semáforos no compiten entre sí. Existe una "red mezcladora" (Mixer Network) que garantiza que la ganancia de flujo en una intersección contribuye a la fluidez óptima de toda la red.
- **Olas Verdes Dinámicas**: La IA detecta el desplazamiento de grupos de vehículos y prepara los semáforos adyacentes antes de que lleguen, reduciendo las paradas innecesarias.

## 3. Preparado para Despliegue Real (Product Readiness)
- **Modo Sombra (Shadow Mode)**: El sistema puede instalarse en una ciudad real y "practicar" durante semanas sin tomar control físico, registrando sus decisiones y comparándolas con el sistema actual del ayuntamiento.
- **Protocolos Industriales**: Soporte nativo para **Modbus TCP** (Siemens, Swarco) y **REST APIs** modernas.
- **Seguridad Triple Capa**:
    1. **Física**: Redundancia de hardware para evitar verdes conflictivos.
    2. **Monitorización**: El `SafetyMonitor` de software bloquea cualquier acción ilógica de la IA (ej. un verde de solo 1 segundo).
    3. **Robustez S2R**: Entrenamiento con ruido en sensores y latencia de red para evitar que la IA falle por imperfecciones del mundo real.

## 4. Métricas de Impacto (Estimadas)
- **-30%** en el tiempo de viaje total en zonas densas.
- **-25%** de emisiones de CO2 al reducir el "Stop-and-Go".
- **100%** de transparencia en las decisiones gracias al motor XAI.

---
**Desarrollado por**: Equipo ATLAS Pro AI
**Estado**: Advanced Production Prototype (TRL 7)
