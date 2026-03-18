# Estrategia de Outreach a Municipios - ATLAS Pro

## 🎯 Objetivo
Posicionar ATLAS Pro como el Sistema Operativo de Tráfico líder para ayuntamientos en España, con foco en el ROI inmediato, la reducción de quejas ciudadanas y el cumplimiento de la Agenda 2030 (ZBE y emisiones).

## 📊 Segmentación de Objetivos

### Fase A: Grandes Municipios (100k - 250k Hab.)
*Perfil: Alta complejidad de tráfico, presupuestos consolidados, alta presión política por ZBE.*

| Ciudad | Población | Foco Comercial | Contacto / Área |
| :--- | :--- | :--- | :--- |
| **Elche** | 245k | Conexión campus-centro y palmeral | movilidadurbana@elche.es |
| **Granada** | 233k | Gestión de accesos turísticos y eventos | cgim@movilidadgranada.com |
| **Badalona** | 231k | Integración metropolitana (Barcelona) | sac@badalona.cat |
| **Oviedo** | 223k | Movilidad en casco histórico y accesos | movilidad@oviedo.es |
| **Cartagena** | 220k | Gestión puerto-centro y tráfico pesado | obrasyproyectos@ayto-cartagena.es |
| **Alcalá de Henares** | 196k | Corredor del Henares (flujo logístico) | movilidad@ayto-alcaladehenares.es |
| **Almería** | 200k | Movilidad estival y accesos playa | ayto@aytoalmeria.es |
| **Burgos** | 173k | Clima adverso e incidencias invernales | oficinademovilidad@aytoburgos.es |

### Fase B: Municipios de Referencia (10k - 100k Hab.)
*Perfil: Smart Cities emergentes, alta receptividad a pilotos de innovación.*

| Ciudad | Población | Foco Comercial | Contacto / Área |
| :--- | :--- | :--- | :--- |
| **Pozuelo de Alarcón** | 89k | Movilidad premium y fluidez residencial | c.movilidad@pozuelo.madrid |
| **Las Rozas** | 96k | Hub tecnológico y Smart City | concejaliamovilidad@lasrozas.es |
| **Gandía** | 83k | Tráfico turístico e intermodalidad | transitmobilitat@gandia.org |
| **Sagunto** | 70k | Eje industrial y logístico | info@aytosagunto.es |
| **Villena** | 34k | Primer paso a Smart City (Piloto) | pacoiniesta@villena.es |
| **Elda** | 52k | Revitalización comercial del centro | sibanez@elda.es |

---

## 📩 Plantillas de Comunicación

### 1. El "Gancho" de LinkedIn (Directivos)
> "Hola **[Nombre]**, un placer. He estado analizando los avances de **[Municipio]** en movilidad y me parece que el reto de la ZBE/Atascos en **[Punto Crítico]** es una oportunidad de oro. 
> 
> Estamos ayudando a ciudades de vuestro perfil a reducir un 30% los tiempos de espera sin cambiar un solo semáforo, solo aplicando nuestra IA (**ATLAS Pro**) sobre vuestra red actual. 
> 
> ¿Te apetecería que te enseñara el dashboard con los datos de reducción de emisiones que estamos consiguiendo? Un saludo."

### 2. Propuesta Formal de Piloto (Email)
**Asunto:** [Propuesta] Reducción del 30% en atascos y emisiones para [Municipio]

**Estimado/a [Nombre]:**

La gestión de la movilidad en **[Municipio]** se enfrenta al desafío de la sostenibilidad y la eficiencia operativa. Los sistemas tradicionales de control de tráfico a menudo no responden a la variabilidad real del flujo diario, generando congestión innecesaria y quejas vecinales.

Desde **ATLAS AI**, hemos desarrollado un Sistema de Control Adaptativo por IA que:
1.  **Reduce el tiempo de viaje en un 30%** mediante algoritmos Q-Learning.
2.  **Instalación 'Invisible'**: Sin obras, se integra con vuestras cámaras y controladores actuales.
3.  **ROI Inmediato**: Un 90% más económico que ampliar infraestructura o licencias legacy.

Le proponemos una **prueba de concepto de 30 días** sin coste en una de sus arterias principales para que su equipo técnico audite los resultados de fluidez en tiempo real.

¿Podemos agendar una breve vídeo-sesión de 10 minutos para mostrarle el impacto estimado en su ciudad?

Atentamente,

**[Tu Nombre]**  
[Tu Cargo]

---

## 🛠 Material de Apoyo (Sales Kit)
- **Comparativa de Costes**: `docs/PRICING.md` (Ahorro vs SCOOT).
- **Ventajas Técnicas**: `ESTRATEGIA_Y_VENTAJAS.md` (Zero-config, SIL-4 Safety).
- **Caso de Uso**: `docs/CASE_STUDIES.md` (Escenario para Ciudad Mediana).

## 📈 Próximos Pasos
1.  **Selección**: Elegir 5 ciudades de la Fase A y 5 de la Fase B para la primera oleada.
2.  **LinkedIn**: Contactar a los Concejales/Directores un martes o miércoles por la mañana.
3.  **Seguimiento**: Si no hay respuesta en 3 días, enviar el PDF de la Comparativa de Competencia.

## 🤖 Automatización de Outreach

Se ha implementado un script para automatizar el envío de los correos de la propuesta formal.

### Requisitos
1. Configurar el archivo `.env` (usa `.env.example` como base).
2. Tener instalado `python-dotenv`.

### Uso
- **Previsualizar a todos**:
  ```bash
  python scripts/send_outreach_emails.py
  ```
- **Previsualizar una ciudad específica**:
  ```bash
  python scripts/send_outreach_emails.py --city "Granada"
  ```
- **Enviar con confirmación individual**:
  ```bash
  python scripts/send_outreach_emails.py --city "Granada" --send
  ```
- **Usar un texto personalizado externo**:
  ```bash
  python scripts/send_outreach_emails.py --city "Granada" --draft-file "ruta/al/borrador.txt" --send
  ```

### Personalización Automática
El script detecta automáticamente el **Foco Comercial** de cada ciudad definido en las tablas superiores y lo integra en el primer párrafo del correo para maximizar la tasa de apertura y respuesta.

> [!CAUTION]
> Aunque el script pide confirmación antes de enviar cada correo, úsalo con responsabilidad para no saturar a los destinatarios.
