# ATLAS Pro - System Architecture

## Visión General de Arquitectura | Architecture Overview

ATLAS Pro es un sistema distribuido de control de tráfico inteligente que combina inteligencia artificial de última generación con estándares de tráfico ampliamente adoptados. La arquitectura está diseñada para escalabilidad, privacidad y despliegue flexible tanto en edge como en cloud.

ATLAS Pro is a distributed intelligent traffic control system combining cutting-edge AI with widely-adopted traffic standards. The architecture is designed for scalability, privacy, and flexible deployment across edge and cloud environments.

---

## Componentes del Sistema | System Components

### 1. **Capa de Sensores y Captura de Datos | Sensor & Data Collection Layer**

Integra múltiples fuentes de datos de tráfico:

- **Bucles Inductivos | Inductive Loops:** Detección de presencia de vehículos, velocidad, conteo
- **Cámaras de Tráfico | Traffic Cameras:** Visión por computadora para detección avanzada (opcional)
- **V2I (Vehicle-to-Infrastructure):** Comunicación directa de vehículos conectados
- **Sensores Ambientales | Environmental Sensors:** Temperatura, humedad, condiciones de tráfico
- **APIs Externas | External APIs:** Datos de Google Maps, Waze, transporte público

**Frecuencia de Actualización:** 100-500ms (configurable)

### 2. **Gateway de Comunicación | Communication Gateway**

Interfaz central que normaliza datos y comunica con todos los dispositivos:

- **NTCIP Protocol Handler:** Comunicación con controladores semafóricos NTCIP 1202/1203/1204
- **UTMC Integration:** Compatibilidad con centros urbanos de gestión de tráfico
- **MQTT Broker:** IoT integrations y dispositivos de terceros
- **REST API Server:** Aplicaciones web y móviles
- **WebSocket Real-time Feed:** Dashboards en vivo

**Protocolos Soportados:** NTCIP, UTMC, MQTT 3.1.1/5.0, HTTP/2, WebSocket

### 3. **Motor de IA ATLAS | ATLAS AI Engine**

Núcleo de inteligencia del sistema (típicamente deployado en edge):

#### **QMIX Multi-Agent Coordinator**
- Coordina políticas de control entre múltiples intersecciones
- Cada intersección es un agente independiente con capacidad de comunicación
- Matriz de valores mixtos (QMIX) para factorización de valores globales
- Comunicación entre agentes mediante MQTT/REST

#### **Dueling DDQN Core**
- **Dueling Architecture:** Separación de evaluación de estado (V) y ventaja de acción (A)
- **Double DQN:** Reduce sobrevaloración en selección de acciones
- **Experiencia Prioritizada | Prioritized Experience Replay:** Muestreo inteligente de memoria
- **Target Networks:** Estabilidad en el entrenamiento

#### **MUSE Metacognition Engine**
- **Explicabilidad:** Desglose de decisiones por componente
- **Feature Importance:** Qué sensores influyen más en cada decisión
- **Anomaly Detection:** Identificación de patrones fuera de lo normal
- **Pattern Recognition:** Descubrimiento automático de situaciones especiales

### 4. **Capa de Almacenamiento de Datos | Data Storage Layer**

```
┌─────────────────────────────────────────┐
│     Local Edge Storage (SQLite/Redis)    │
│  - Estado actual de intersecciones       │
│  - Caché de decisiones recientes         │
│  - Métricas en tiempo real               │
└─────────────────────────────────────────┘
         │
         │ Sync (opcional)
         ▼
┌─────────────────────────────────────────┐
│     Cloud Storage (PostgreSQL/TimescaleDB)│
│  - Histórico completo de tráfico         │
│  - Modelos entrenados                    │
│  - Análisis a largo plazo                │
│  - Reportes y auditoría                  │
└─────────────────────────────────────────┘
```

### 5. **Capa de Presentación | Presentation Layer**

- **Dashboard Web:** Monitoreo en tiempo real, control manual, configuración
- **API REST:** Integración con sistemas terceros
- **Mobile App (opcional):** Alertas de tráfico, reportes
- **Reportes Automáticos:** Generación de KPIs para municipios

---

## Flujo de Datos | Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         ATLAS Pro Data Flow                       │
└──────────────────────────────────────────────────────────────────┘

INGESTA | INGESTION LAYER:
────────────────────────────────
  Bucles Inductivos      Cámaras      V2I      APIs Externas
       │                   │           │           │
       └───────────────────┼───────────┼───────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Data Normalization│  (frecuencia 100-500ms)
                  │  & Validation    │
                  └──────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Feature Extraction│
                  │ (velocidad, cola,│
                  │  ocupancia, etc) │
                  └──────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐
    │  AI Engine       │       │ MUSE Metacog.    │
    │  (QMIX + DDQN)   │       │ (Explicabilidad) │
    └──────────────────┘       └──────────────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Control Commands │   │ Local Storage    │
    │ (timing, offset) │   │ (métricas, logs) │
    └──────────────────┘   └──────────────────┘
             │                     │
             │                     │ Sync (opcional)
             │                     │
             ▼                     ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Traffic Signal   │   │ Cloud Analytics  │
    │ Controllers      │   │ & Reporting      │
    │ (NTCIP/UTMC)     │   │                  │
    └──────────────────┘   └──────────────────┘
             │                     │
             ▼                     ▼
    ┌──────────────────────────────────────┐
    │        Dashboard & Reports           │
    │     Real-time Monitoring & KPIs      │
    └──────────────────────────────────────┘
```

---

## Arquitectura de Despliegue en Edge | Edge Deployment Architecture

```
INTERSECCIÓN 1:
┌─────────────────────────────────────────────────┐
│  ATLAS Pro Edge Unit (Jetson Nano/Orin)         │
│ ┌───────────────────────────────────────────┐  │
│ │  AI Engine (Dueling DDQN + QMIX)          │  │
│ │  - ONNX Model Runtime                     │  │
│ │  - Local inference (~50ms latency)        │  │
│ └───────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────┐  │
│ │  Data Processing & Logging                │  │
│ │  - SQLite cache                           │  │
│ │  - Anomaly detection                      │  │
│ └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
           │         │
           │         └─ NTCIP/UTMC
           │
         MQTT
           │
     ┌─────┴──────────────────────┐
     │                            │
INTERSECCIÓN 2              INTERSECCIÓN 3
(Similar edge unit)        (Similar edge unit)
     │                            │
     └────────────┬───────────────┘
                  │
              MQTT/REST
                  │
         ┌────────┴─────────┐
         │                  │
    Cloud Hub         Dashboard
    (Opcional)        (Web/Mobile)
    - Sincronización
    - Análisis histórico
    - Reportes
```

**Ventajas del Edge Deployment:**
- ✅ Latencia ultra-baja (< 50ms)
- ✅ Privacidad: todos los datos en local
- ✅ Continuidad: funciona sin cloud si es necesario
- ✅ Escalable: sin límite de ancho de banda central
- ✅ Costo reducido: no requiere transmisión constante de datos

---

## Puntos de Integración | Integration Points

### **NTCIP 1202/1203/1204**
```
ATLAS Pro ←→ Controller NTCIP
│
├─ Phase timing (segundos)
├─ Offset coordination (ms)
├─ Pedestrian timing
├─ Signal plan selection
└─ Detector feedback
```

### **UTMC (Urban Traffic Management Centre)**
```
Local ATLAS Deployment ←→ Central UTMC Server
│
├─ Real-time intersection status
├─ Incident reporting
├─ Centralized analytics
├─ Coordination signals
└─ Manual override capability
```

### **MQTT IoT Integration**
```
ATLAS Pro ←→ MQTT Broker
│
├─ Sensor data ingestion
├─ Device commands
├─ Real-time telemetry
└─ Third-party integrations
```

### **REST JSON API**
```
ATLAS Pro API ←→ Third-party Systems
│
├─ GET /intersections/{id}/status
├─ POST /intersections/{id}/command
├─ GET /analytics/heatmap
├─ GET /analytics/incidents
└─ WebSocket /live/feed
```

---

## Escalabilidad | Scalability Design

### **Arquitectura Distribuida**

```
ZONE 1 (50 intersecciones)    ZONE 2 (100 intersecciones)
┌─────────────────────────┐   ┌──────────────────────────┐
│  Local QMIX Coordinator │   │  Local QMIX Coordinator  │
│  (1 Jetson Orin)        │   │  (2 Jetson Orins)        │
│  │                      │   │  │                       │
│  ├─ Int 1-50            │   │  ├─ Int 51-150           │
│  │ (MQTT cluster)       │   │  │ (MQTT cluster)        │
│  └─ Local analytics     │   │  └─ Local analytics      │
└─────────────────────────┘   └──────────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                    Inter-Zone MQTT
                    (loose coupling)
                          │
                          ▼
                  ┌────────────────────┐
                  │  Global Dashboard  │
                  │  + Cloud Analytics │
                  │  (opcional)        │
                  └────────────────────┘
```

**Capacidades de Escalado:**
- **1 Intersección:** Jetson Nano (€1,200)
- **10-50 Intersecciones:** 1 Jetson Orin (€3,500)
- **50-200 Intersecciones:** 2-4 Jetson Orins (distribuido)
- **200-500+ Intersecciones:** Edge cluster + cloud analytics

---

## Ciclo de Control | Control Loop

```
Período: 100-500ms (configurable)

┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. SENSORES                                        │
│     Leer estado actual de detectores                │
│     └─ Ocupancia, velocidad, conteo                │
│                                                     │
│  2. OBSERVACIÓN                                     │
│     Construir estado de red (graph neural)          │
│     └─ Considerar intersecciones vecinas            │
│                                                     │
│  3. AI INFERENCE                                    │
│     QMIX + Dueling DDQN forward pass                │
│     └─ Salida: timing + offset + plan              │
│                                                     │
│  4. VALIDACIÓN (MUSE)                              │
│     ├─ Verificar anomalías                         │
│     ├─ Evaluar explicabilidad                      │
│     └─ Comprobar restricciones (min/max)           │
│                                                     │
│  5. COMANDO                                         │
│     Enviar timing a controlador (NTCIP)             │
│     └─ Esperar confirmación                        │
│                                                     │
│  6. ALMACENAMIENTO                                  │
│     ├─ Guardar decisión + explicación              │
│     ├─ Actualizar métricas                         │
│     └─ Detectar anomalías                          │
│                                                     │
└─────────────────────────────────────────────────────┘
            (Latencia total: < 50ms)
```

---

## Comunicación Inter-Agentes | Inter-Agent Communication

**QMIX Multi-Agent Protocol:**

```
┌────────────────────────────────────────────┐
│  Cada Intersección = Agente Independiente  │
│  ┌──────────────────────────────────────┐ │
│  │ Local Policy Network (Dueling DDQN)  │ │
│  │ + Local State Representation          │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
         │
         │ Envía: estado local + valor-Q
         │
         ▼
┌────────────────────────────────────────────┐
│  QMIX Value Factorization                  │
│  Q_total = Q_1 + Q_2 + ... + Q_n           │
│  + Weighted mixing (QMIX network)          │
│                                             │
│  Beneficios:                               │
│  - Coordinación sin comunicación directa   │
│  - Escalable a 100+ agentes                │
│  - Convergencia garantizada                │
└────────────────────────────────────────────┘
         │
         │ Salida: política coordinada
         │
         ▼
┌────────────────────────────────────────────┐
│  Local Policy Execution (NTCIP Control)    │
└────────────────────────────────────────────┘
```

---

## Seguridad de la Arquitectura | Architecture Security

### **Capas de Protección**

1. **Comunicación:**
   - TLS 1.3 para todas las conexiones
   - mTLS entre edge units
   - JWT tokens con rotación

2. **Edge Security:**
   - Inferencia sin envío de datos (privacidad)
   - Encriptación de almacenamiento local
   - Aislamiento de procesos

3. **API Security:**
   - Rate limiting (100 req/seg por cliente)
   - Input validation en todos los endpoints
   - SQL injection protection
   - CORS configurado

4. **Auditoría:**
   - Log completo de todas las decisiones
   - Trazabilidad de cambios
   - GDPR compliance

---

## Requisitos de Red | Network Requirements

```
Edge Unit (Jetson)
└─ Ethernet o 4G/5G
   ├─ Ancho de banda mínimo: 1 Mbps upstream
   ├─ Latencia máxima a controladores: 100ms
   ├─ Jitter máximo: ±50ms
   └─ Disponibilidad: 99.9%

Sensor Data Ingestion
├─ Bucles inductivos: 9600 baud serial (~100 bytes/100ms)
├─ Cámaras: 1-10 Mbps (si se procesan en edge)
└─ V2I: Variable (típicamente < 1 Mbps)

Cloud Sync (opcional)
└─ MQTT/REST para análisis:
   ├─ ~100 KB por intersección por día (comprimido)
   ├─ Actualizaciones de modelos: ~50 MB mensual
   └─ Puede buffering local si conexión cae
```

---

## Disponibilidad y Resiliencia | Availability & Resilience

```
Escenario 1: Pérdida de Cloud (si se usa)
└─ Sistema continúa funcionando en edge
   - Decisiones autónomas locales
   - Sincronización cuando vuelva conectividad

Escenario 2: Fallo de una intersección
└─ Otras intersecciones continúan coordinadas
   - Degradación elegante del sistema

Escenario 3: Fallo de Edge Unit completo
└─ Fallback a controlador legacy (NTCIP)
   - Modo manual o planes fijos
```

**SLA Garantizado:** 99.9% uptime (máx 8.7 horas downtime/año)

---

## Monitoreo y Observabilidad | Monitoring & Observability

```
Métricas Capturadas:
├─ Sistema:
│  ├─ CPU/Memoria/Disco (edge units)
│  ├─ Latencia de decisión
│  └─ Tasa de error
│
├─ Tráfico:
│  ├─ Velocidad promedio
│  ├─ Tiempo de espera
│  ├─ Ocupancia de carril
│  └─ Throughput (veh/min)
│
├─ IA:
│  ├─ Pérdida de entrenamiento (si aplica)
│  ├─ Confianza de decisión
│  ├─ Anomalías detectadas
│  └─ Explicabilidad score
│
└─ Negocio:
   ├─ Reducción CO2 (t/mes)
   ├─ Ahorro de combustible ($)
   ├─ Mejora de movilidad (%)
   └─ Satisfacción ciudadana

Visualización:
├─ Dashboard web (en vivo)
├─ Alertas email/SMS (críticas)
└─ Reportes PDF mensuales
```

---

## Roadmap Técnico | Technical Roadmap

| Versión | Fecha | Cambios Arquitectónicos |
|---------|-------|------------------------|
| v4.0 | Feb 2026 | QMIX + DDQN + MUSE + ONNX |
| v4.1 | May 2026 | Soporte para V2X nativo |
| v4.2 | Aug 2026 | Federación de múltiples ciudades |
| v5.0 | Q1 2027 | Transformers en lugar de LSTM (si resulta beneficioso) |

---

## Referencias Técnicas | Technical References

- **Deep RL Frameworks:** OpenAI baselines, Ray RLlib, Stable Baselines3
- **Graph Neural Networks:** PyTorch Geometric, DGL
- **ONNX Runtime:** Para inferencia eficiente en edge
- **Time Series DB:** TimescaleDB (histórico, análisis)
- **Message Queue:** Mosquitto (MQTT) o RabbitMQ



