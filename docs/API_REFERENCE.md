# ATLAS Pro - API Reference

## API Overview

ATLAS Pro provides multiple integration points for accessing traffic control data, sending commands, and receiving real-time updates. This document covers the public API surface suitable for municipal systems, transit authorities, and third-party integrators.

**Base URL:** `https://api.atlas-ai.tech/v1` (or local edge unit IP)

**Authentication:** JWT Bearer token in Authorization header

---

## 1. Authentication

### Generate API Key

First, obtain an API key from the ATLAS Pro dashboard.

```bash
# Request format
curl -X POST https://api.atlas-ai.tech/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your_api_key_here",
    "expires_in": 3600
  }'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Using JWT in Requests

```bash
# All subsequent requests include the JWT
curl -H "Authorization: Bearer {access_token}" \
  https://api.atlas-ai.tech/v1/intersections
```

### Rate Limiting

```
Standard Tier:
├─ 1,000 requests per minute per API key
├─ 100,000 requests per day
└─ Burst capacity: 50 req/sec

Enterprise Tier:
├─ 10,000 requests per minute
├─ 1,000,000 requests per day
└─ Priority support
```

Response Headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
X-RateLimit-Reset: 1645123456
```

---

## 2. Core Endpoints

### Intersections

#### **GET /intersections**
List all intersections under ATLAS Pro control.

```bash
Request:
GET /intersections?limit=50&offset=0&zone=downtown

Query Parameters:
├─ limit (int): Max results, default 50, max 500
├─ offset (int): Pagination offset, default 0
├─ zone (string): Filter by zone name (optional)
├─ status (string): Filter by status [active, inactive, maintenance]
└─ type (string): Filter by intersection type [urban, highway, rural]

Response (200 OK):
{
  "data": [
    {
      "id": "int_001",
      "name": "Main St & 5th Ave",
      "location": {
        "latitude": 40.7128,
        "longitude": -74.0060
      },
      "status": "active",
      "type": "urban",
      "phase_count": 4,
      "detector_count": 12,
      "zone": "downtown",
      "last_update": "2026-03-01T12:34:56Z"
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

#### **GET /intersections/{id}**
Get detailed status of specific intersection.

```bash
Request:
GET /intersections/int_001

Response (200 OK):
{
  "id": "int_001",
  "name": "Main St & 5th Ave",
  "status": "active",
  "current_state": {
    "phase": 1,
    "time_in_phase": 32,
    "green_duration": 45,
    "next_phase": 2,
    "time_to_next": 13
  },
  "detectors": {
    "detector_001": {
      "name": "EBL",
      "occupancy": 78,
      "speed": 24.5,
      "vehicle_count": 14,
      "status": "operational"
    }
  },
  "performance": {
    "avg_wait_time": 32.4,
    "vehicles_served": 847,
    "throughput": 42.3,
    "queue_length": 8.2
  },
  "coordination": {
    "zone": "downtown",
    "offset": 12,
    "next_green": 2450
  }
}
```

#### **POST /intersections/{id}/command**
Send control command to intersection.

```bash
Request:
POST /intersections/int_001/command
Content-Type: application/json

{
  "command_type": "set_phase",
  "phase": 2,
  "duration": 50,
  "priority": "normal",
  "reason": "Incident response",
  "duration": 45
}

Command Types:
├─ set_phase: Force specific phase
├─ extend_phase: Add seconds to current phase
├─ skip_phase: Move to next phase
├─ manual_override: Take manual control
├─ release_control: Return to AI control
└─ emergency: Special emergency signal

Response (202 Accepted):
{
  "command_id": "cmd_12345",
  "status": "sent",
  "acknowledged_by": "int_001",
  "execution_time": "2026-03-01T12:34:58Z"
}
```

---

### Analytics & Reporting

#### **GET /analytics/intersections/{id}/kpi**
Get Key Performance Indicators for intersection.

```bash
Request:
GET /intersections/int_001/kpi?start=2026-02-01&end=2026-03-01&granularity=daily

Query Parameters:
├─ start (ISO 8601): Start date
├─ end (ISO 8601): End date
├─ granularity (string): [hourly, daily, weekly, monthly]
└─ metrics (string): Comma-separated metric names (optional)

Metric Names Available:
├─ avg_delay
├─ wait_time_reduction
├─ throughput
├─ queue_length
├─ co2_emissions
├─ fuel_consumption
└─ vehicle_speed

Response (200 OK):
{
  "intersection_id": "int_001",
  "period": "2026-02-01T00:00:00Z to 2026-03-01T00:00:00Z",
  "granularity": "daily",
  "data": [
    {
      "timestamp": "2026-02-01T00:00:00Z",
      "metrics": {
        "avg_delay": 34.2,
        "wait_time_reduction": -28.3,
        "throughput": 856,
        "queue_length": 7.8,
        "co2_emissions": 12.4,
        "fuel_consumption": 4.2,
        "vehicle_speed": 24.6
      }
    }
  ],
  "summary": {
    "avg_delay_monthly": 33.8,
    "total_co2_reduction": 245.6,
    "total_vehicles_served": 25620
  }
}
```

#### **GET /analytics/heatmap**
Get real-time traffic heatmap for visualization.

```bash
Request:
GET /analytics/heatmap?zone=downtown&granularity=5min

Response (200 OK):
{
  "timestamp": "2026-03-01T12:34:56Z",
  "zone": "downtown",
  "intersections": [
    {
      "id": "int_001",
      "location": {"lat": 40.7128, "lng": -74.0060},
      "congestion_level": "high",
      "congestion_score": 78,
      "avg_wait_time": 42.3,
      "throughput": 34.2
    }
  ]
}
```

#### **GET /analytics/incidents**
Get detected and reported incidents.

```bash
Request:
GET /incidents?start=2026-03-01T00:00:00Z&severity=high&limit=50

Query Parameters:
├─ start (ISO 8601): Start timestamp
├─ end (ISO 8601): End timestamp
├─ severity (string): [low, medium, high, critical]
├─ type (string): [congestion, accident, event, system, other]
└─ resolved (boolean): Filter by resolution status

Response (200 OK):
{
  "incidents": [
    {
      "incident_id": "inc_001",
      "timestamp": "2026-03-01T12:15:34Z",
      "type": "accident",
      "severity": "high",
      "location": {
        "intersection_id": "int_001",
        "latitude": 40.7128,
        "longitude": -74.0060
      },
      "description": "Multi-vehicle collision at Main & 5th",
      "affected_phases": [1, 2],
      "automated_response": "extended_phase_2_by_30s",
      "manual_response": "police_dispatch_requested",
      "resolved": false,
      "resolution_time": null,
      "impact": {
        "estimated_delay": 240,
        "affected_vehicles": 156
      }
    }
  ],
  "summary": {
    "total_detected": 4,
    "total_reported": 2,
    "high_severity": 1,
    "average_response_time": 42
  }
}
```

---

## 3. Real-Time Data Streams

### WebSocket Connection

Connect to real-time data stream for live updates.

```javascript
// JavaScript example
const ws = new WebSocket('wss://api.atlas-ai.tech/v1/stream');
const token = 'your_jwt_token_here';

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: token
  }));

  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['int_001:status', 'int_001:metrics', 'incidents:all']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### WebSocket Channel Topics

```
Intersection Data:
├─ int_{id}:status     → Phase, timing, state changes
├─ int_{id}:metrics    → Real-time KPIs
├─ int_{id}:detectors  → Detector occupancy/speed
└─ int_{id}:alerts     → Alerts specific to intersection

System Wide:
├─ incidents:all       → All detected incidents
├─ incidents:critical  → Only critical incidents
├─ system:health       → System health metrics
└─ system:alerts       → System-wide alerts
```

### Example WebSocket Messages

```json
{
  "type": "status_update",
  "intersection_id": "int_001",
  "timestamp": "2026-03-01T12:34:56.789Z",
  "phase": 1,
  "time_in_phase": 25,
  "green_end_in": 20
}
```

```json
{
  "type": "incident_detected",
  "incident_id": "inc_001",
  "timestamp": "2026-03-01T12:15:34Z",
  "type": "congestion",
  "location": {"intersection_id": "int_001"},
  "confidence": 94.2,
  "severity": "high"
}
```

---

## 4. MQTT Integration

ATLAS Pro can publish/subscribe to MQTT broker for IoT integration.

### MQTT Topics

**Published Topics (ATLAS → Subscribers):**

```
atlas/intersection/{id}/status     → Current phase & timing
atlas/intersection/{id}/metrics    → KPIs and performance
atlas/intersection/{id}/detectors  → Detector data
atlas/incidents/{severity}         → Incident notifications
atlas/system/health                → System health
atlas/alerts/active                → Active system alerts
```

**Subscribed Topics (ATLAS listens for):**

```
transit/request/{corridor}         → Transit priority requests
parking/occupancy/{zone}           → Parking availability (future)
weather/conditions/{zone}          → Weather data
events/calendar/{city}             → City events/incidents
external/incidents/{source}        → External incident reports
```

### MQTT Configuration

```
Broker: mqtt.atlas-ai.tech
Port: 8883 (TLS)
Username: (API key)
Password: (API key)
QoS: 1 (At least once)
Retain: true for status topics
```

### MQTT Message Format

```json
Topic: atlas/intersection/int_001/status
{
  "timestamp": "2026-03-01T12:34:56Z",
  "phase": 1,
  "time_in_phase": 25,
  "green_until": 50,
  "next_phase": 2,
  "confidence": 97.3
}
```

---

## 5. Data Export & Reporting

### Export Historical Data

#### **POST /export/request**
Request bulk data export for analysis.

```bash
Request:
POST /export/request
{
  "format": "csv",
  "period": {
    "start": "2026-01-01T00:00:00Z",
    "end": "2026-03-01T23:59:59Z"
  },
  "intersections": ["int_001", "int_002"],
  "metrics": ["delay", "throughput", "occupancy", "speed"],
  "granularity": "15min"
}

Response (200 OK):
{
  "export_id": "exp_001",
  "status": "processing",
  "estimated_time": 300,
  "download_url": "https://api.atlas-ai.tech/v1/exports/exp_001/download"
}
```

#### **GET /export/{export_id}/status**
Check export progress.

```bash
Response:
{
  "export_id": "exp_001",
  "status": "completed",
  "progress": 100,
  "file_size": "2.4 MB",
  "download_url": "https://api.atlas-ai.tech/v1/exports/exp_001/download",
  "expires_in": 604800
}
```

---

## 6. System Administration

### System Configuration

#### **GET /system/config**
Get current system configuration.

```bash
Request:
GET /system/config

Response:
{
  "system": {
    "name": "Downtown Traffic Management",
    "version": "4.0",
    "deployment_type": "edge",
    "zones": ["downtown", "midtown", "uptown"],
    "total_intersections": 150
  },
  "ai_engine": {
    "algorithm": "Dueling DDQN + QMIX",
    "model_version": "4.0.2",
    "last_training": "2026-02-28T14:30:00Z",
    "inference_latency_ms": 45.2
  },
  "network": {
    "protocol": "NTCIP 1202/1203",
    "polling_interval_ms": 200,
    "heartbeat_interval_s": 30
  }
}
```

#### **GET /system/health**
System health and diagnostics.

```bash
Response:
{
  "status": "healthy",
  "uptime_days": 42,
  "last_incident": "2026-02-28T09:15:00Z",
  "components": {
    "ai_engine": "healthy",
    "database": "healthy",
    "api_server": "healthy",
    "mqtt_broker": "healthy",
    "intersections_online": 148,
    "intersections_offline": 2
  },
  "resource_usage": {
    "cpu_percent": 34.2,
    "memory_percent": 58.6,
    "disk_percent": 45.3,
    "network_mbps": 2.4
  }
}
```

#### **GET /system/alerts**
Active system alerts and warnings.

```bash
Response:
{
  "alerts": [
    {
      "alert_id": "alr_001",
      "severity": "medium",
      "type": "detector_failure",
      "intersection": "int_045",
      "message": "Detector EBL offline for 12 minutes",
      "detected_at": "2026-03-01T12:22:34Z",
      "action_required": true
    }
  ],
  "count": {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3
  }
}
```

---

## 7. Error Handling

### HTTP Status Codes

```
200 OK              ✅ Successful request
201 Created         ✅ Resource created
202 Accepted        ✅ Async operation accepted
204 No Content      ✅ Successful, no response body
400 Bad Request     ❌ Invalid parameters
401 Unauthorized    ❌ Authentication failed
403 Forbidden       ❌ Insufficient permissions
404 Not Found       ❌ Resource not found
429 Too Many        ❌ Rate limit exceeded
500 Server Error    ❌ Internal error
503 Unavailable     ❌ Service temporarily down
```

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_PHASE",
    "message": "Phase must be between 1 and 4",
    "details": {
      "provided": 5,
      "valid_range": "1-4"
    }
  },
  "request_id": "req_12345"
}
```

---

## 8. Code Examples

### Python Example

```python
import requests
import json

API_KEY = "your_api_key"
BASE_URL = "https://api.atlas-ai.tech/v1"

# Authenticate
auth_response = requests.post(
    f"{BASE_URL}/auth/token",
    json={"api_key": API_KEY}
)
token = auth_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Get intersection status
response = requests.get(
    f"{BASE_URL}/intersections/int_001",
    headers=headers
)
intersection = response.json()["data"][0]
print(f"Phase: {intersection['current_state']['phase']}")

# Get analytics
response = requests.get(
    f"{BASE_URL}/intersections/int_001/kpi",
    params={"start": "2026-02-01", "end": "2026-03-01"},
    headers=headers
)
kpi = response.json()
print(f"Average delay: {kpi['summary']['avg_delay_monthly']}s")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.atlas-ai.tech/v1";

// Create client
const client = axios.create({ baseURL: BASE_URL });

async function getIntersectionStatus(intersectionId) {
  try {
    // Authenticate
    const authResp = await client.post('/auth/token', {
      api_key: API_KEY
    });
    const token = authResp.data.access_token;

    // Set auth header
    client.defaults.headers.common['Authorization'] =
      `Bearer ${token}`;

    // Get intersection
    const resp = await client.get(`/intersections/${intersectionId}`);
    return resp.data;
  } catch (error) {
    console.error('API Error:', error.response?.data);
  }
}

getIntersectionStatus('int_001').then(console.log);
```

---

## 9. Rate Limits & Quotas

### Request Limits

```
API Tier             Requests/min  Requests/day  Burst
────────────────────────────────────────────────────
Free                 100           10,000        5/sec
Starter              500           50,000        10/sec
Professional         1,000         100,000       50/sec
Enterprise           10,000        1,000,000     100/sec
```

### Quota Headers

Every API response includes rate limit information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 997
X-RateLimit-Reset: 1645123456
```

---

## 10. Webhook Events (Optional)

### Webhook Configuration

For Enterprise tier, register webhooks for event notifications:

```bash
POST /webhooks
{
  "url": "https://your-system.com/webhooks/atlas",
  "events": ["incident.detected", "system.alert", "performance.degraded"],
  "secret": "webhook_secret_for_verification"
}
```

### Webhook Event Format

```json
{
  "event_id": "evt_001",
  "event_type": "incident.detected",
  "timestamp": "2026-03-01T12:15:34Z",
  "data": {
    "incident_id": "inc_001",
    "severity": "high",
    "type": "accident",
    "location": {"intersection_id": "int_001"}
  }
}
```

---

## Support

**API Support:** support@atlas-ai.tech
**Status Page:** status.atlas-ai.tech
**Changelog:** /changelog

