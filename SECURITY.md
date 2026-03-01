# Security Policy

## Overview

ATLAS Pro takes security and privacy extremely seriously. This document outlines our security practices, how to report vulnerabilities, and our commitment to protecting data and systems.

---

## Security Principles

1. **Privacy by Design:** Data minimization, local processing, no unnecessary centralization
2. **Defense in Depth:** Multiple layers of security controls
3. **Transparency:** Clear communication about security practices
4. **Compliance:** Adherence to international standards (GDPR, CCPA, etc.)
5. **Continuous Improvement:** Regular security audits and updates

---

## Data Security

### Data Classification

ATLAS Pro handles three types of data:

```
PUBLIC DATA
├─ Aggregated traffic statistics
├─ Historical anonymized trends
└─ Public reports and dashboards

INTERNAL OPERATIONAL DATA
├─ Real-time traffic flows
├─ Sensor measurements
├─ Control decisions
└─ Incident reports

PERSONAL/SENSITIVE DATA
├─ User accounts and credentials
├─ Configuration settings
├─ Audit logs
└─ Integration credentials (API keys)
```

### Data Protection Measures

**Encryption in Transit:**
- TLS 1.3 for all network communication
- mTLS between edge units and cloud (if used)
- Perfect forward secrecy (PFS) enabled

**Encryption at Rest:**
- AES-256 for sensitive data storage
- Database encryption (if using cloud)
- Local edge storage encrypted with device keys

**Data Retention:**
- Configurable retention policies (default 1 year)
- Automatic purging of old data
- Manual deletion on request
- Audit trail preservation (5 years recommended)

---

## Authentication & Authorization

### API Authentication

```
Method: JWT Bearer Token
Expiration: Configurable (default 24 hours)
Refresh: Via secure refresh token mechanism
Revocation: Immediate token blacklisting supported
```

### Access Control

- **Role-Based Access Control (RBAC):** Admin, Operator, Viewer roles
- **Principle of Least Privilege:** Users get minimum required permissions
- **Multi-Factor Authentication (MFA):** Available for sensitive operations
- **Session Management:** Secure session handling with timeouts

### API Key Management

- Unique keys per application/integration
- Rotation policy: 90-day recommended
- Revocation: Immediate and auditable
- Scope limitations: Keys can be restricted to specific endpoints

---

## Network Security

### Communication Protocols

```
NTCIP Control:
├─ TCP/IP (port 161 - SNMP)
├─ TLS optional (recommended)
└─ NTCIP authentication supported

MQTT (IoT Integration):
├─ TLS encryption (port 8883)
├─ MQTT username/password
├─ TLS client certificates (optional)
└─ Topic-based access control

REST API:
├─ HTTPS only (TLS 1.2+)
├─ CORS policy enforcement
├─ Rate limiting per client
└─ Input validation and sanitization

WebSocket:
├─ WSS (WebSocket Secure)
├─ TLS encryption
├─ JWT authentication
└─ Periodic ping/pong for connection validity
```

### Firewall Rules

Recommended firewall configuration:

```
Inbound to ATLAS Edge Unit:
├─ Port 8080: REST API (HTTPS, restricted to admin network)
├─ Port 1883: MQTT (TLS, restricted to local/trusted networks)
├─ Port 161: SNMP/NTCIP (restricted to local traffic)
├─ Port 5671: AMQP (if using, restricted)
└─ All others: DENY

Outbound from ATLAS:
├─ Port 443: Cloud sync (if enabled, outgoing only)
├─ Port 123: NTP time sync
└─ All others: DENY
```

---

## Vulnerability Disclosure & Responsible Disclosure

### Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**Email:** security@atlas-ai.tech
**PGP Key:** [Available upon request]

**Include:**
1. Description of vulnerability
2. Steps to reproduce
3. Potential impact
4. Your contact information (optional anonymity supported)

### Our Response

- **Initial Response:** Within 24 hours
- **Assessment:** Severity evaluation
- **Timeline:** 90-day disclosure timeline agreed with reporter
- **Fix:** Patch development and testing
- **Release:** Security update released
- **Credit:** Researcher credited (unless declined)

### Supported Versions

Security updates are provided for:
- Current major version: Full support
- Previous major version: 12 months of support
- Older versions: No guaranteed support (upgrade recommended)

---

## Compliance & Certifications

### GDPR (General Data Protection Regulation)

ATLAS Pro is fully compliant with GDPR:

- ✅ **Data Minimization:** Only collects necessary aggregate data
- ✅ **Purpose Limitation:** Data only used for traffic optimization
- ✅ **Storage Limitation:** Configurable retention (default 1 year)
- ✅ **Data Subject Rights:** Can request data access/deletion
- ✅ **Breach Notification:** Incident response plan in place
- ✅ **Data Agreements:** Signing DPA (Data Processing Agreement) available

### CCPA (California Consumer Privacy Act)

Compliance features:
- ✅ Privacy notice clarity
- ✅ Opt-out mechanisms (for cloud sync if used)
- ✅ Data access rights
- ✅ Deletion rights
- ✅ Non-discrimination clause

### Additional Standards

- **ISO 27001:** Information Security Management (implementation in progress)
- **NIST Cybersecurity Framework:** Aligned with NIST controls
- **IEC 62443:** Industrial Cybersecurity (for traffic systems)

---

## Infrastructure Security

### Edge Unit Security

```
Physical Security:
├─ Secure enclosure recommended
├─ Tamper detection (optional)
├─ Environmental monitoring (temp/humidity)
└─ Physical access logging

System Hardening:
├─ Minimal OS footprint (no unnecessary services)
├─ Automatic security updates
├─ Secure boot enabled
├─ SELinux/AppArmor enforcing
└─ Regular security scans
```

### Cloud Infrastructure (If Used)

```
Hosting:
├─ Reputable cloud provider (AWS/Azure/GCP)
├─ Multi-region redundancy
├─ Automatic backups
└─ Disaster recovery plan

Monitoring:
├─ 24/7 security monitoring
├─ Intrusion detection systems
├─ DDoS protection
├─ Automated threat response
```

### Database Security

```
Access:
├─ Strong authentication
├─ Principle of least privilege
├─ No default credentials
└─ Audit logging of all access

Hardening:
├─ Regular patching
├─ Automated backups
├─ Point-in-time recovery
└─ Encrypted backups
```

---

## Operational Security

### Incident Response Plan

```
1. Detection: Security event identified
2. Assessment: Severity and scope evaluation
3. Containment: Limit impact, isolate affected systems
4. Investigation: Root cause analysis
5. Remediation: Fix vulnerability
6. Testing: Verify fix
7. Notification: Inform affected parties
8. Improvement: Update security controls
```

### Security Monitoring

- **Real-time Alerting:** Suspicious activity triggers alerts
- **Log Collection:** Centralized logging (SIEM)
- **Anomaly Detection:** Unusual patterns flagged
- **Automated Response:** Some threats auto-mitigated

### Regular Assessments

- **Vulnerability Scans:** Monthly automated scans
- **Penetration Testing:** Annual professional testing
- **Code Review:** Security-focused code reviews before release
- **Dependency Scanning:** Third-party library vulnerability checking

---

## Third-Party & Dependency Management

### Vendor Assessment

All third-party software is assessed for:
- Security track record
- Maintenance status
- Licensing compatibility
- Support level

### Dependency Monitoring

- Continuous monitoring of dependencies for vulnerabilities
- Automated alerting of new vulnerabilities
- Regular updates and patches
- Bill of Materials (BOM) maintained

### Open Source Compliance

- All open source licenses properly documented
- Compliance with license terms
- No GPL/copyleft in proprietary code
- License manifests in deployments

---

## Cryptographic Standards

### Algorithms Used

```
Encryption:
├─ AES-256 (symmetric)
├─ RSA-2048+ or ECDSA (asymmetric)
└─ ChaCha20-Poly1305 (authenticated encryption)

Hashing:
├─ SHA-256 (general purpose)
├─ PBKDF2 (password hashing)
└─ Argon2 (newer deployments)

Key Management:
├─ HSM (Hardware Security Module) recommended
├─ Key rotation policies (90-180 days)
├─ Secure key storage
└─ No hardcoded keys in code
```

### TLS Configuration

```
Minimum Version: TLS 1.2
Recommended: TLS 1.3

Cipher Suites (TLS 1.3):
├─ TLS_AES_256_GCM_SHA384
├─ TLS_CHACHA20_POLY1305_SHA256
└─ TLS_AES_128_GCM_SHA256

Weak Ciphers: DISABLED
Export-grade: DISABLED
```

---

## User Security Responsibilities

### For Municipalities/Operators

1. **Keep Credentials Secure:** Don't share API keys or passwords
2. **Use Strong Passwords:** Minimum 12 characters, complexity requirements
3. **Enable MFA:** Multi-factor authentication recommended
4. **Update Systems:** Apply security patches promptly
5. **Network Isolation:** Restrict API access to trusted networks
6. **Audit Logs:** Regularly review access logs

### For Integration Partners

1. **Secure API Keys:** Store in secure vaults, never hardcode
2. **Validate Inputs:** Sanitize all API inputs
3. **Use HTTPS:** Always use encrypted communication
4. **Keep Libraries Updated:** Update dependencies regularly
5. **Error Handling:** Don't expose sensitive info in error messages
6. **Testing:** Include security testing in QA

---

## Security Training

### Team Training

- Annual security awareness training mandatory
- Incident response drills twice per year
- Secure coding training for developers
- Privacy awareness training

### Customer Resources

- Security best practices guide
- Configuration hardening checklist
- Incident response procedures
- Regular security briefings

---

## Contact

**Security Questions:** security@atlas-ai.tech
**Responsible Disclosure:** security@atlas-ai.tech
**General Support:** support@atlas-ai.tech

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial security policy |

---

## Acknowledgments

We thank the following organizations for security best practices and standards:
- OWASP (Open Web Application Security Project)
- NIST (National Institute of Standards and Technology)
- SANS Institute
- Cloud Security Alliance

---

**Last Updated:** March 1, 2026
**Next Review:** June 1, 2026

