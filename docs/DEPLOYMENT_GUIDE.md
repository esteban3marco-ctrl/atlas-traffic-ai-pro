# ATLAS Pro - Deployment Guide

## Guía de Despliegue | Deployment Overview

This guide covers the end-to-end process of deploying ATLAS Pro in your city's traffic infrastructure. ATLAS Pro is designed for rapid deployment with minimal disruption to existing traffic operations.

---

## 1. Pre-Deployment Assessment

### Prerequisites Checklist

Before beginning deployment, ensure your city has:

- [ ] **Traffic Signal Controllers:** Compatible with NTCIP 1202/1203 protocol
  - Models: 2070, Econolite ASC, Siemens SCATS, others with NTCIP support
  - Legacy systems can be integrated via gateway devices

- [ ] **Sensor Infrastructure:** At least one of:
  - ✅ Inductive loops (standard)
  - ✅ Video detection systems
  - ✅ V2I communication (optional for enhanced performance)

- [ ] **Network Connectivity:**
  - Ethernet or 4G/5G to each intersection
  - Minimum: 1 Mbps bandwidth per edge unit
  - Latency: < 100ms to controllers

- [ ] **Power Supply:**
  - 110V/220V AC at each intersection enclosure
  - UPS backup recommended for edge units

- [ ] **Stakeholder Alignment:**
  - Municipal traffic authority approval
  - Police department coordination
  - Public transit agency buy-in
  - IT/Cybersecurity team readiness

### Compatibility Assessment

```
Step 1: Controller Inventory
├─ Document all intersection controllers
├─ Verify NTCIP 1202 support or gateway option
├─ Test NTCIP communication link
└─ Record current timing plans

Step 2: Sensor Survey
├─ Type of detection (loops, cameras, V2I)
├─ Sensor condition and age
├─ Integration requirements
└─ Data quality baseline

Step 3: Network Assessment
├─ Bandwidth available
├─ Latency measurement
├─ Redundancy/failover options
└─ Cloud connectivity (if hybrid)

Step 4: Power & Physical
├─ Available electrical capacity
├─ Enclosure space for edge unit
├─ Environmental conditions (temp, humidity)
└─ Physical security review
```

**Typical Assessment Duration:** 1-2 weeks

---

## 2. Pilot Program (Recommended)

### Benefits of Pilot Phase

- 🔍 Validate ATLAS Pro effectiveness in YOUR city
- 📊 Collect performance baseline
- 👥 Train operations team hands-on
- 📈 Build internal confidence before city-wide rollout
- 🏆 Generate success story for stakeholder approval

### Pilot Sizing

**Recommended:** 15-25 intersections representing:
- [ ] Different urban typologies (downtown, residential, highway)
- [ ] High-congestion corridors
- [ ] Mixed sensor types
- [ ] Geographic distribution (north/south/east/west)

### Pilot Timeline

```
Week 1-2: Setup & Configuration
├─ Hardware arrival and inventory
├─ Installation in pilot intersections
├─ Network configuration
├─ NTCIP controller integration
└─ Testing and verification

Week 3: Training & Launch
├─ Operations team training (2 days)
├─ Go-live preparation
├─ Parallel operation mode (ATLAS + legacy)
└─ Stakeholder communication

Week 4-12: Pilot Operation
├─ 24/7 monitoring
├─ Performance data collection
├─ Incident response drills
├─ Monthly stakeholder reports
└─ Model optimization

Week 13: Evaluation
├─ Performance analysis
├─ Cost-benefit calculation
├─ Lessons learned documentation
└─ Scale-up approval
```

**Typical Pilot Cost:** €80,000 - €150,000 (including hardware + support)

### Pilot Success Criteria

| Metric | Target | Typical Result |
|--------|--------|---|
| System Availability | > 99% | 99.8% |
| Avg Wait Time Reduction | 20-25% | 28-32% |
| Peak Period Flow (+%) | 15-20% | 18-24% |
| Incident Detection | > 90% | 94% |
| Controller Compatibility | 100% | 100% |
| Operations Team Proficiency | 4/5 stars | 4.6/5 |

---

## 3. Hardware Deployment

### Edge Unit Options

#### **Option A: Jetson Nano (Small Cities 1-20 intersections)**

```
Specifications:
├─ GPU: NVIDIA Maxwell (128 CUDA cores)
├─ CPU: ARM A57 quad-core @ 1.43 GHz
├─ Memory: 4GB LPDDR4
├─ Storage: 16-128GB microSD
├─ Power: 5W (typical)
├─ Cost: $99-199 hardware only

Deployment:
├─ 1 unit per 20 intersections
├─ Jetson Nano Bundle: €1,200 (with case, power, storage)
├─ Installation time: 30 minutes per location
└─ Maintenance: Minimal (fanless cooling option)

Best For:
├─ Smaller cities
├─ Budget-constrained pilots
├─ Proof-of-concept projects
└─ Rural areas
```

#### **Option B: Jetson Orin (Mid-Large Cities 20-150 intersections)**

```
Specifications:
├─ GPU: NVIDIA Ampere (2048 CUDA cores)
├─ CPU: ARM Cortex A78AE octa-core @ 2.2 GHz
├─ Memory: 8-12GB LPDDR5X
├─ Storage: NVMe SSD 256GB
├─ Power: 15W (typical)
├─ Cost: €2,000-3,500

Deployment:
├─ 1 unit per 50-150 intersections
├─ Jetson Orin Bundle: €3,500
├─ Installation time: 1 hour per location
└─ Maintenance: Quarterly checks

Best For:
├─ Medium to large cities
├─ Multiple corridor coordination
├─ Advanced analytics
└─ V2I integration ready
```

#### **Option C: Enterprise Edge PC (Large Cities 150+ intersections)**

```
Specifications:
├─ CPU: Intel Xeon or AMD EPYC
├─ GPU: NVIDIA RTX A4000/A5000
├─ Memory: 32GB+ RAM
├─ Storage: Redundant NVMe RAID
├─ Power: 50-100W
├─ Cost: €5,000-15,000

Deployment:
├─ Centralized data center deployment
├─ 1 unit per 150-300 intersections
├─ High availability active-passive config
└─ Full disaster recovery

Best For:
├─ Large city deployments
├─ Entire metropolitan areas
├─ Mission-critical traffic control
└─ Integration with smart city platforms
```

### Installation Procedure

```
Physical Installation (30-60 minutes per location):
┌─────────────────────────────────────────┐
│ 1. PREPARATION                          │
│    ├─ Safety barriers & traffic control │
│    ├─ Power supply verification         │
│    └─ Network cable routing             │
│                                         │
│ 2. HARDWARE PLACEMENT                   │
│    ├─ Mount edge unit in enclosure      │
│    ├─ Connect to UPS backup (optional)  │
│    └─ Cable management                  │
│                                         │
│ 3. NETWORK INTEGRATION                  │
│    ├─ Ethernet to controller            │
│    ├─ 4G/5G modem setup (if needed)    │
│    └─ Connectivity test                 │
│                                         │
│ 4. CONTROLLER INTEGRATION                │
│    ├─ NTCIP parameter sync              │
│    ├─ Sensor calibration                │
│    └─ SCATS coordination (if applicable)│
│                                         │
│ 5. TESTING & VALIDATION                 │
│    ├─ System boot & connectivity check  │
│    ├─ Inference latency test            │
│    ├─ Failover test (manual mode)       │
│    └─ Documentation & handover          │
│                                         │
└─────────────────────────────────────────┘
```

---

## 4. Software Configuration

### Initial Setup

**Configuration Tasks (1-2 hours per city):**

```
1. NETWORK CONFIGURATION
   ├─ DHCP or static IP assignment
   ├─ Firewall rules (port 8080, 1883, 5671)
   ├─ DNS configuration
   └─ NTP time sync

2. CONTROLLER PARAMETERS
   ├─ Import existing timing plans
   ├─ Set safety constraints
   │  ├─ Min/Max green times
   │  ├─ Pedestrian clearance
   │  └─ All-red intervals
   └─ Define coordination zones

3. SENSOR CONFIGURATION
   ├─ Detector mapping (loop → logical phase)
   ├─ Calibration data
   ├─ Sensor failure detection thresholds
   └─ Data validation rules

4. AI ENGINE INITIALIZATION
   ├─ Load pre-trained DDQN model
   ├─ Set QMIX coordination parameters
   ├─ Configure anomaly detection sensitivity
   └─ Define learning rate (exploration vs exploitation)

5. API & INTEGRATION
   ├─ REST API key generation
   ├─ MQTT broker connection
   ├─ Webhook configuration (optional)
   └─ Data export settings
```

### Model Tuning

ATLAS Pro comes with pre-trained models, but local optimization is recommended:

```
Phase 1: Observation (Week 1-2)
├─ Run in advisory mode (no control changes)
├─ Collect baseline performance metrics
├─ Learn local traffic patterns
└─ Validate sensor data quality

Phase 2: Guided Learning (Week 3-4)
├─ Enable control with safety guardrails
├─ Constrain action space (e.g., ±10% from baseline)
├─ Monitor performance improvements
└─ Adjust exploration parameters

Phase 3: Full Optimization (Week 5+)
├─ Remove constraints gradually
├─ Let AI find optimal policies
├─ Continuous performance monitoring
└─ Monthly performance review
```

---

## 5. Integration with Existing Systems

### NTCIP Controller Integration

```
ATLAS Pro ←→ NTCIP Controller (1202/1203)

Data Flow:
├─ READ: Current detector occupancy/presence
├─ READ: Current phase timing
├─ READ: Controller status/faults
├─ WRITE: New phase timing command
├─ WRITE: Coordination offset
└─ WRITE: Signal plan selection

Protocol Details:
├─ TCP/IP connection (SNMP over TCP)
├─ Port: 161 (SNMP default)
├─ Polling interval: 100-500ms
├─ Timeout: 2-5 seconds
└─ Fallback: Manual mode if ATLAS disconnected

Configuration:
├─ NTCIP device IP address
├─ SNMP community strings
├─ OID mappings for your controller type
└─ Failsafe timing plan
```

### UTMC (Central Management System) Integration

If your city has a UTMC center:

```
Local ATLAS Deployment → UTMC Server

Bidirectional Communication:
├─ ATLAS → UTMC: Real-time status, incidents, KPIs
├─ UTMC → ATLAS: Coordination signals, manual overrides
├─ Protocol: MQTT or REST API
└─ Frequency: Real-time push or 1-5 min polling

UTMC Dashboard Integration:
├─ View ATLAS-managed intersections
├─ Manual intervention capability
├─ Incident acknowledgment
├─ Historical analytics access
└─ Export to city-wide traffic dashboard
```

### Transit Integration (Optional)

For Bus Rapid Transit (BRT) or priority signals:

```
ATLAS Pro ←→ Transit Priority System

Features:
├─ Receive bus location (GPS)
├─ Request green wave for transit route
├─ Automatic priority phase extension
├─ Integration with existing transit priority
└─ Reporting on transit delay reduction

Integration Points:
├─ REST API: /transit/request-priority
├─ MQTT topic: atlas/transit/+/gps
└─ GTFS Real-time data (optional)
```

---

## 6. Operations & Maintenance

### Daily Operations

```
Morning Checklist (5-10 minutes):
├─ System health dashboard review
├─ Overnight incident log review
├─ Pending alerts/maintenance tasks
├─ Weather conditions impact assessment
└─ Coordination with police/transit ops

Throughout Day:
├─ Monitor peak period performance
├─ Respond to incident requests
├─ Manual override capability (if needed)
└─ Data collection for analytics

Evening Summary:
├─ KPI metrics review
├─ Incident documentation
├─ Preparation for next day
└─ Alert thresholds review
```

### Preventive Maintenance

```
Monthly:
├─ System health check
├─ Hardware temperature/status
├─ Network connectivity test
├─ Software update availability
└─ Performance trend analysis

Quarterly:
├─ Full system audit
├─ Sensor recalibration
├─ Controller firmware verification
├─ Backup restoration test
└─ Security audit

Annually:
├─ Hardware replacement evaluation
├─ Model retraining with latest data
├─ Contract/license renewal
├─ Disaster recovery drill
└─ Stakeholder reporting
```

### Support Response Times

```
Issue Severity | ATLAS Response | Your Action
─────────────────────────────────────────────
Critical       | < 1 hour       | Fallback to manual timing
(System down)  | Escalation     | Override ATLAS if needed
               |                |

High           | < 4 hours      | Monitor situation
(Degraded      | Troubleshooting| Prepare communication
performance)   |                |

Medium         | < 24 hours     | Log issue
(Minor issues) | Root cause     | Schedule maintenance
               |                |

Low            | < 5 days       | Document
(Notifications)| Analysis       | Include in next update
               |                |
```

---

## 7. Scaling from Pilot to City-Wide

### Phase Deployment Strategy

**Recommended Approach: Incremental Geographic Expansion**

```
PHASE 1: Pilot (Week 1-12)
├─ 15-25 intersections (sample across city)
├─ Prove concept locally
└─ Expected results: -25-35% wait times

PHASE 2: Corridor (Month 4-5)
├─ Expand successful pilot zone
├─ Add 50-75 intersections
├─ Activate green wave optimization
└─ Expect: 15-20% additional improvement from coordination

PHASE 3: District (Month 6-9)
├─ Roll out to full district/zone
├─ 100-150 intersections
├─ QMIX multi-agent coordination active
└─ Realization: 30-40% systemwide wait time reduction

PHASE 4: City-Wide (Month 10-15)
├─ Remaining intersections
├─ Full ATLAS network optimization
└─ Maximum benefits: 35-45% reduction achieved
```

### Timeline & Resource Planning

```
City Size: 100 Intersections
Timeline: 6 months

Month 1: Deployment & Training
├─ Weeks 1-4: Hardware installation (20 int/week)
├─ Week 2-3: Operations team training (40 hours)
├─ Week 4: All systems go-live
└─ Resources: 4 technicians, 1 project manager

Month 2-3: Optimization
├─ Fine-tune models per intersection
├─ Corridor coordination tuning
├─ 24/7 support active
└─ Resources: 2 technicians, support team

Month 4-6: Scaling
├─ Install remaining 80 intersections
├─ 20 new intersections/month
└─ Resources: 3 technicians

Ongoing:
├─ 1-2 FTE operations engineer
├─ Remote support from ATLAS team
└─ Monthly optimization updates
```

---

## 8. Performance Monitoring & Reporting

### Key Performance Indicators (KPIs)

```
Traffic Efficiency:
├─ Average delay (seconds)
├─ Wait time reduction (%)
├─ Queue length
├─ Throughput (vehicles/hour)
├─ Travel time variability
└─ Peak period performance

Environmental:
├─ CO2 emissions (tons/month)
├─ Fuel consumption (liters)
├─ Vehicle idle time
├─ Emissions per vehicle/mile
└─ Green score

Operational:
├─ System availability (%)
├─ Incident detection rate (%)
├─ Manual intervention frequency
├─ Controller reliability
└─ Support ticket response time

Business:
├─ Cost per intersection/month
├─ ROI (months to payback)
├─ Citizen complaints (trending)
└─ Media sentiment
```

### Reporting Schedule

```
Daily:
└─ Automated dashboard (24-hour trending)

Weekly:
└─ Operations team summary report

Monthly:
├─ Detailed KPI analysis
├─ Stakeholder briefing
├─ Trend identification
└─ Recommendations for optimization

Quarterly:
├─ Executive summary for municipal leaders
├─ Cost-benefit analysis update
├─ Photo/video case study development
└─ Public communication assets

Annual:
├─ Full year-over-year comparison
├─ ROI verification
├─ Citizen satisfaction survey
├─ Conference presentation/publication
└─ Contract renewal assessment
```

---

## 9. Troubleshooting & Support

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High latency (>100ms) | Network congestion | Check bandwidth, reduce polling frequency |
| Inconsistent timing | Sensor calibration drift | Recalibrate detectors |
| Controller not responding | NTCIP connection lost | Verify network, restart controller |
| Anomaly false positives | Model too sensitive | Adjust sensitivity threshold |
| Performance regression | Model staleness | Retrain model with recent data |
| Hardware thermal issues | Ambient temp too high | Add ventilation, relocate unit |

### Support Channels

```
ATLAS Pro Support:
├─ Email: support@atlas-ai.tech
├─ Phone: +[Country] (during business hours)
├─ Portal: https://support.atlas-ai.tech
├─ Slack (Enterprise): Dedicated channel
└─ On-site (if needed): Travel costs apply

Expected Response Time:
├─ Critical (system down): < 1 hour
├─ High (degraded): < 4 hours
├─ Medium: < 24 hours
└─ Low: < 5 business days
```

---

## 10. Success Checklist

### Go-Live Checklist

Before going live with your ATLAS Pro deployment:

- [ ] All hardware installed and tested
- [ ] Network connectivity verified (all intersections)
- [ ] NTCIP controllers responding to ATLAS commands
- [ ] Sensor data quality baseline established
- [ ] Operations team trained (sign-off required)
- [ ] Safety constraints configured correctly
- [ ] Backup plans in place (manual override)
- [ ] Incident response procedures documented
- [ ] Monitoring dashboards operational
- [ ] Stakeholder communications ready
- [ ] Media/public messaging prepared
- [ ] 24/7 support team briefed

### Post-Deployment Success (30 days)

- [ ] System availability > 99%
- [ ] Initial KPI targets achieved (-20% wait time minimum)
- [ ] Zero critical incidents
- [ ] Operations team confidence level 4+/5
- [ ] Public feedback positive
- [ ] Maintenance routine established
- [ ] Optimization cycle started

### 6-Month Success Milestone

- [ ] All phases deployed on schedule
- [ ] KPI targets exceeded (-30%+ wait times)
- [ ] Full coordination benefits realized
- [ ] Cost-benefit positive
- [ ] Stakeholder confidence high
- [ ] Media coverage / case study ready
- [ ] Contract renewal planned

---

## Contact & Next Steps

Ready to deploy ATLAS Pro?

📧 **Email:** esteban3marco@gmail.com
📱 **Phone:** (Available in deployment proposal)
🌐 **Web:** (Coming soon)

**Next Steps:**
1. Schedule pre-deployment assessment
2. Review compatibility report
3. Plan pilot program
4. Finalize timeline & budget
5. Begin hardware procurement

---

