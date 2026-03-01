# ATLAS Pro - Frequently Asked Questions

## General Questions

### Q: What is ATLAS Pro?
**A:** ATLAS Pro is an AI-powered traffic control system that uses Deep Reinforcement Learning (Dueling DDQN + QMIX) to automatically optimize traffic signal timing in real-time. Unlike legacy systems that use fixed plans or simple heuristics from the 1970s, ATLAS Pro adapts continuously to traffic conditions, reducing wait times and emissions.

### Q: How is ATLAS Pro different from SCOOT or SCATS?
**A:**
- **Technology:** ATLAS Pro uses modern Deep RL (2026), while SCOOT/SCATS use heuristics from 1979
- **Cost:** ATLAS Pro is 60-70% cheaper (€49-89/month vs $40K-60K upfront)
- **Speed:** ATLAS Pro makes decisions in <50ms vs legacy systems in 5-10 seconds
- **Privacy:** ATLAS Pro is edge-first (data stays local), competitors require cloud
- **Explainability:** ATLAS Pro offers MUSE metacognition (auditable), legacy systems are black-box

### Q: How does ATLAS Pro work?
**A:**
1. **Collects Data:** Real-time sensor inputs (occupancy, speed, queue length)
2. **AI Analysis:** Dueling DDQN network analyzes traffic state
3. **Coordination:** QMIX multi-agent system coordinates with neighboring intersections
4. **Control:** Issues timing commands to traffic signal controllers (NTCIP standard)
5. **Learning:** Continuously improves through experience

### Q: Is ATLAS Pro open source?
**A:** No, ATLAS Pro is proprietary software. This repository contains documentation, deployment guides, and API reference for integrators and partners, but the AI models and source code remain confidential. This protects our intellectual property and ensures consistent performance across all deployments.

---

## For Municipalities & Cities

### Q: How long does implementation take?
**A:** Typically 2-4 weeks for most cities:
- **Assessment:** 1-2 weeks
- **Pilot (optional):** 8-12 weeks
- **Full deployment:** 2-4 weeks
- **Optimization:** Ongoing

Compare to SCOOT (6-12 months) or SCATS (8-12 months).

### Q: What are the typical costs?
**A:**
**SaaS Model (Recommended):**
- €49/intersection/month (Basic)
- €89/intersection/month (Pro with advanced features)
- 100 intersections = €58,800-106,800/year

**Perpetual License:**
- €2,500/intersection (one-time)
- 100 intersections = €250,000 upfront
- Optional annual updates: €300/intersection

**Hardware:**
- Jetson Nano bundle: €1,200
- Jetson Orin bundle: €3,500

### Q: What's the ROI?
**A:** Typical payback period is 18-24 months through:
- Reduced fuel consumption: €300-500 per vehicle per year saved
- Increased throughput: More vehicles served per cycle
- Reduced emissions (environmental value)
- Lower operational costs vs SCOOT/SCATS support

**Example:** City of 100 intersections saves ~€350K-500K annually → ROI in 18-24 months

### Q: Will ATLAS Pro work with my existing traffic signal controllers?
**A:** Yes, if they support NTCIP 1202/1203 protocol:
- ✅ Econolite ASC/ASCFIRE
- ✅ Siemens SCATS compatibility units
- ✅ Most modern 2070-series controllers
- ⚠️ Older controllers may need NTCIP gateway device (~€5K-10K each)

Contact support for specific controller compatibility assessment.

### Q: What if we lose internet connection?
**A:** ATLAS Pro continues operating safely:
- **Edge Unit:** Has local AI models, operates autonomously
- **Fallback:** Automatically reverts to pre-programmed timing plans
- **Resumes:** Resynchronizes when connection restores
- **No Disruption:** Traffic keeps flowing normally

This is why edge-first deployment is more reliable than pure cloud solutions.

### Q: How does ATLAS Pro protect privacy?
**A:**
- **Data Stays Local:** No sensitive traffic data sent to cloud
- **GDPR Compliant:** No personal data collected (only aggregate traffic flows)
- **No Cameras Required:** Works with existing inductive loops or video (processed locally)
- **V2I Optional:** Vehicle data never leaves the edge unit

Compare to SCOOT/SCATS which require sending all data to external clouds.

### Q: Can ATLAS Pro handle special events (concerts, sports, demonstrations)?
**A:** Yes, ATLAS Pro includes:
- **Automatic Detection:** Identifies unusual patterns
- **Event Calendar Integration:** Manual event input
- **Dynamic Response:** Automatically adjusts timing
- **Emergency Protocols:** Can override for emergency vehicles
- **Manual Control:** Operators can take control anytime

### Q: How do you ensure equity and fairness in traffic control?
**A:**
- **MUSE Explainability:** Every decision is auditable and justifiable
- **Constraint Setting:** Can enforce fairness rules (e.g., min wait times per approach)
- **Public Transparency:** Can publish decision logs and analytics
- **Citizen Feedback:** System can be adjusted based on community input

This addresses concerns that "AI black boxes" make biased decisions.

### Q: What if ATLAS Pro makes a bad decision?
**A:**
- **Human Override:** Operators can always take manual control
- **Safety Constraints:** System has hard limits on timing changes
- **Learning:** If a decision was suboptimal, ATLAS Pro learns and improves
- **Explainability:** Can see exactly why a decision was made
- **Fallback:** If system has issues, reverts to pre-programmed plans

### Q: How is ATLAS Pro supported?
**A:**
**SaaS Tier:** Includes 24/7 support
**Perpetual License:** Optional support plans (€5K-20K/year)

Support includes:
- Remote monitoring and alerts
- Incident response (< 4 hours for high priority)
- Regular optimization updates
- Training for operations staff
- Annual performance review

---

## Technical Questions

### Q: What hardware does ATLAS Pro require?
**A:** Minimum requirements:
- **Edge Unit:** NVIDIA Jetson Nano (€99-199) or better
- **Memory:** 4GB RAM minimum (8GB recommended)
- **Storage:** 16GB SD card (SSD for 50+ intersections)
- **Network:** Ethernet or 4G/5G
- **Power:** 5-15W typical

No server farm or expensive data center needed.

### Q: How much network bandwidth does ATLAS Pro need?
**A:** Very minimal:
- **Per Intersection:** <1 Mbps typical
- **Cloud Sync (optional):** ~100KB per intersection per day
- **Can Buffer Offline:** Works fine without cloud for hours/days

Compare to video-based systems that need 5-10 Mbps per intersection.

### Q: Can ATLAS Pro integrate with our existing traffic management system?
**A:** Yes, multiple integration options:
- **NTCIP:** Standard protocol for signal control
- **UTMC:** For central traffic management centers
- **REST API:** For modern web-based systems
- **MQTT:** For IoT and smart city integration
- **Custom APIs:** Can be developed for legacy systems

### Q: What about latency and response time?
**A:**
- **Decision Latency:** <50ms (inference on edge)
- **Control Latency:** <200ms (send timing to controller)
- **Total Response:** <250ms (very fast compared to human-operated systems)

This is critical for intersection safety and coordination.

### Q: Does ATLAS Pro require special sensors?
**A:** No:
- ✅ Works with existing inductive loops
- ✅ Works with video detection systems
- ✅ Works with V2I communication (if available)
- ✅ Works with hybrid sensor combinations
- ❌ Does NOT require expensive new infrastructure

### Q: How does ATLAS Pro handle sensor failures?
**A:** Graceful degradation:
- **Single Sensor Fail:** System continues with available sensors
- **Multiple Sensor Failures:** Falls back to pre-programmed plans
- **Automatic Recovery:** Resumes optimal control when sensors recover
- **Alerts:** Notifies operators of sensor issues

### Q: Can ATLAS Pro scale to multiple cities?
**A:** Yes, fully scalable:
- **Single City:** 1 edge unit can manage 50-200 intersections
- **Multiple Cities:** Federated deployment (each city independent)
- **Cloud Coordination:** Optional cloud hub for inter-city coordination
- **Test Proven:** Architecture verified for 500+ intersections

### Q: What's the AI model size?
**A:** Efficient models for edge deployment:
- **Dueling DDQN:** ~50-200 MB
- **QMIX Network:** ~10-20 MB
- **Total Model:** <500 MB (fits on any device)
- **Inference:** <50ms on Jetson Nano

No need for expensive GPUs compared to some alternatives.

### Q: How often is the AI model updated?
**A:** Two approaches:
1. **Continuous Learning:** Local model improves every day
2. **Monthly Retraining:** New model pushed if significant improvements found
3. **Annual Major Update:** Significant algorithm improvements (v4.0 → v4.1)

You control update frequency.

---

## Security & Compliance

### Q: Is ATLAS Pro secure against hacking?
**A:** Yes, multiple security layers:
- **TLS Encryption:** All network communication encrypted
- **JWT Authentication:** API token-based access control
- **Edge Isolation:** No direct internet access to controllers
- **Input Validation:** All API inputs validated
- **Audit Logging:** Complete record of all changes
- **Regular Updates:** Security patches deployed promptly

### Q: Does ATLAS Pro comply with GDPR?
**A:** Yes, fully GDPR compliant:
- **No Personal Data:** Doesn't collect names, IDs, or other PII
- **Aggregate Data Only:** Traffic flows and statistics
- **Data Minimization:** Only collects needed data
- **Data Retention:** Configurable (default 1 year)
- **User Rights:** Can request data deletion

### Q: What about CCPA and other privacy regulations?
**A:** ATLAS Pro is designed for compliance:
- **California CCPA:** Compliant
- **UK GDPR:** Compliant
- **International:** Follows strict privacy standards

### Q: Can ATLAS Pro be used for surveillance?
**A:** No, ATLAS Pro is designed to prevent surveillance:
- **Aggregate Data Only:** No individual vehicle tracking
- **No Cameras (Optional):** Can use inductive loops instead
- **Data Anonymization:** No linking to vehicle owners
- **Privacy-First:** Architecture prioritizes citizen privacy

### Q: Who owns the traffic data ATLAS Pro collects?
**A:** The municipality/city:
- **Contractually:** Data ownership stays with city
- **No Third-Party Access:** ATLAS doesn't share data
- **Exportable:** Can extract your data anytime
- **No Licensing:** You own your own traffic data

---

## Operational Questions

### Q: Who operates ATLAS Pro daily?
**A:** Typically your existing traffic operations team:
- **Training Provided:** 2-3 days of hands-on training
- **Dashboard:** Intuitive interface (most operators learn in first week)
- **Support:** Remote support team available 24/7
- **Automation:** Most tasks fully automated (minimal daily work)

### Q: What if operators prefer manual control?
**A:** Full manual override available anytime:
- **Override Button:** One-click to take control
- **Phase Selection:** Can manually set any phase timing
- **Easy Switch Back:** Returns to AI control with one click
- **No Penalties:** System doesn't "learn wrong" from manual control

### Q: How do you handle peak hours and emergencies?
**A:**
- **Peak Hour Aware:** ATLAS learns peak patterns
- **Emergency Protocols:** Can preset emergency timing
- **Manual Override:** Operators can take control for emergencies
- **Incident Response:** Automatic detection and response

### Q: Does ATLAS Pro help with traffic incident management?
**A:** Yes, significant features:
- **Automatic Detection:** Identifies accidents, stalls, debris
- **Response:** Automatically adjusts timing to mitigate impact
- **Rerouting Coordination:** Works with routing apps
- **Alert Broadcast:** Can send alerts to navigation apps
- **Recovery:** Automatically recovers timing when incident clears

### Q: Can ATLAS Pro coordinate with traffic cameras?
**A:** Yes, if available:
- **Input:** Can ingest video detection data
- **Incident Detection:** Video feeds enhance incident detection
- **Privacy:** Video only processed locally, never sent to cloud
- **Optional:** Not required, works with loops alone

### Q: What about air quality monitoring?
**A:** Indirect benefits:
- **Reduced Idling:** Less fuel burn = less pollution
- **Improved Flow:** Better traffic flow = lower emissions
- **CO2 Tracking:** Reports on emission reductions achieved
- **Not a Sensor:** ATLAS doesn't measure air quality, but reduces pollution

---

## Business & Partnership

### Q: Is ATLAS Pro suitable for private companies?
**A:** Yes, ATLAS Pro can be deployed for:
- **Parking Lots:** Smart lot optimization
- **Private Transportation:** Company vehicle fleets
- **Campus Management:** University/corporate campus
- **Commercial Centers:** Shopping malls, entertainment venues

Contact sales for commercial deployment pricing.

### Q: Can integrators/developers build on ATLAS Pro?
**A:** Yes, we have partner programs:
- **API Integration:** Public REST/MQTT APIs
- **Reseller Programs:** License ATLAS and white-label
- **System Integration:** Build custom integrations
- **Certification Program:** Partner certification available

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Q: Are there free trials available?
**A:** Yes:
- **3-Month Pilot:** Free for municipalities (25-50 intersections)
- **Proof of Concept:** Can demonstrate on your hardware
- **No Long-term Commitment:** Pure evaluation

Contact sales@atlas-ai.tech to arrange.

### Q: What if we want to switch from SCOOT/SCATS?
**A:** We have migration programs:
- **Coexistence:** Can run ATLAS alongside legacy system
- **Gradual Migration:** Replace intersections one at a time
- **Data Migration:** Can import existing timing plans
- **Zero Risk:** Fallback to legacy if needed

### Q: Do you offer training programs?
**A:** Yes, comprehensive training:
- **Operator Training:** 2-3 days for operations staff (included)
- **Advanced Training:** 1-week for system administrators
- **Webinars:** Monthly training sessions (free for customers)
- **Certification:** Can certify trainers for your organization

---

## Contact & Support

### Where can I get more information?
📧 **Email:** estebanmarcojobs@gmail.com
📞 **Phone:** (Available upon request)
🌐 **Website:** (Coming soon)

### How do I request a demo?
1. Email estebanmarcojobs@gmail.com with:
   - City name and size
   - Number of intersections
   - Current system (SCOOT/SCATS/Fixed/Other)
   - Key challenges

2. We'll schedule a demo call or on-site visit
3. Personalized ROI analysis provided
4. Pilot program proposal if interested

### How do I request a pilot program?
1. Complete compatibility assessment
2. Select 20-25 representative intersections
3. 2-week installation window
4. 12-week pilot period with full support
5. Performance evaluation and go/no-go decision

### What if I have other questions?
Email: estebanmarcojobs@gmail.com with your question, and our team will respond within 24 hours.

---



