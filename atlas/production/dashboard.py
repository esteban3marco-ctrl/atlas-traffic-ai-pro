
import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from collections import deque

logger = logging.getLogger("ATLAS.Dashboard")

try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

def create_dashboard_html() -> str:
    """The Mega-Premium HUD Dashboard based on the USER's reference image."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATLAS Pro // Neural Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --neon-cyan: #00f2ff;
            --neon-blue: #0066ff;
            --neon-green: #39ff14;
            --neon-red: #ff0055;
            --neon-purple: #bc13fe;
            --bg-deep: #020617;
            --card-bg: rgba(15, 23, 42, 0.4);
            --glass-border: rgba(0, 242, 255, 0.3);
            --glass-blur: blur(25px);
        }

        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            background: var(--bg-deep);
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            overflow-y: auto;
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(0, 102, 255, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(188, 19, 254, 0.1) 0%, transparent 40%);
        }

        /* Scanline HUD effect */
        body::after {
            content: "";
            position: absolute; top:0; left:0; width:100%; height:100%;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%),
                        repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0px, transparent 1px, transparent 2px);
            background-size: 100% 4px, 100% 3px;
            pointer-events: none; z-index: 1000;
        }

        .hud-container {
            display: grid;
            grid-template-columns: 280px 1fr 350px;
            grid-template-rows: 80px minmax(500px, auto) auto;
            gap: 15px;
            padding: 15px;
            width: 100%;
            min-height: 100vh;
            position: relative;
            z-index: 10;
        }

        /* Glass HUD Panel */
        .hud-panel {
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 4px;
            backdrop-filter: var(--glass-blur);
            position: relative;
            padding: 20px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 242, 255, 0.05);
        }

        /* Corner glow accents like the image */
        .hud-panel::before {
            content: ''; position: absolute; top: -1px; left: -1px; 
            width: 20px; height: 20px; border-top: 3px solid var(--neon-cyan); border-left: 3px solid var(--neon-cyan);
        }
        .hud-panel::after {
            content: ''; position: absolute; bottom: -1px; right: -1px; 
            width: 20px; height: 20px; border-bottom: 3px solid var(--neon-cyan); border-right: 3px solid var(--neon-cyan);
        }

        /* TOP BAR */
        .top-bar {
            grid-column: span 3;
            display: flex; justify-content: space-between; align-items: center;
        }
        .logo-text { font-family: 'Orbitron'; letter-spacing: 5px; font-weight: 900; font-size: 24px; color: var(--neon-cyan); text-shadow: 0 0 15px var(--neon-cyan); }
        .system-status { font-family: 'JetBrains Mono'; font-size: 12px; color: var(--neon-green); }

        /* LEFT COLUMN (STATS) */
        .left-col { grid-row: span 2; display: flex; flex-direction: column; gap: 20px; }
        
        .stat-card { flex: 1; }
        .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 2px; }
        .stat-value { font-size: 42px; font-weight: 900; line-height: 1; margin: 10px 0; font-family: 'Orbitron'; }
        .stat-trend { font-size: 14px; font-weight: bold; display: flex; align-items: center; gap: 5px; }

        /* CIRCLE CHART (Like the Safety Score) */
        .progress-circle {
            position: relative; width: 120px; height: 120px; margin: 0 auto;
        }
        .progress-circle svg { transform: rotate(-90deg); }
        .progress-circle .bg { fill: none; stroke: rgba(255,255,255,0.1); stroke-width: 8; }
        .progress-circle .fg { fill: none; stroke: var(--neon-green); stroke-width: 8; stroke-dasharray: 283; stroke-dashoffset: 0; stroke-linecap: round; transition: 1s; }
        .progress-circle .text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 22px; font-weight: 900; }

        /* CENTER MAP (CANVAS) */
        .center-map { grid-row: span 2; padding: 5px; position: relative; overflow: hidden; display: flex; justify-content: center; align-items: center; background: #000; border-radius: 4px; }
        #intersection-canvas { width: 100%; height: 100%; max-width: 500px; max-height: 500px; object-fit: contain; }
        
        /* Floating HUD Elements for Map */
        .map-overlay-text { position: absolute; top: 30px; left: 30px; font-family: 'Orbitron'; font-size: 12px; color: var(--neon-cyan); z-index: 1000; pointer-events: none; }

        /* RIGHT COLUMN (MUSE & QMIX) */
        .right-col { grid-row: span 2; display: flex; flex-direction: column; gap: 15px; }
        .muse-item { margin-top: 10px; }
        .muse-label { font-size: 11px; display: flex; justify-content: space-between; color: var(--neon-cyan); font-family: 'JetBrains Mono'; }
        .muse-bar-bg { height: 6px; background: rgba(255,255,255,0.1); margin-top: 5px; border-radius: 3px; overflow: hidden; }
        .muse-bar-fg { height: 100%; background: var(--neon-cyan); width: 0%; transition: width 0.5s; }
        .muse-bar-fg.warning { background: #ffbf00; box-shadow: 0 0 10px #ffbf00; }
        .muse-bar-fg.danger { background: var(--neon-red); box-shadow: 0 0 10px var(--neon-red); }

        .qmix-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
        .qmix-agent { background: rgba(0,0,0,0.5); border: 1px solid #334155; padding: 10px; text-align: center; border-radius: 4px; }
        .qmix-agent .a-name { font-size: 10px; color: rgba(255,255,255,0.5); }
        .qmix-agent .a-val { font-size: 16px; font-weight: bold; color: var(--neon-green); margin-top: 5px; }

        .diagnosis-text { font-size: 13px; color: var(--neon-cyan); line-height: 1.4; font-weight: 500; font-family: 'JetBrains Mono'; }
        
        /* BOTTOM PANEL (SALIENCY / LOG) */
        .bottom-panel { grid-column: span 2; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .brain-viz { display: flex; align-items: center; justify-content: center; position: relative; }
        .neural-bars { display: flex; align-items: flex-end; gap: 5px; height: 100px; }
        .n-bar { width: 8px; background: linear-gradient(to top, var(--neon-blue), var(--neon-cyan)); transition: 0.5s; height: 10px; }

        /* Brain Icon Styling */
        .brain-svg { width: 100px; height: 100px; fill: var(--neon-cyan); filter: drop-shadow(0 0 10px var(--neon-cyan)); }

        /* MARKERS */
        .marker-glow { width: 20px; height: 20px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 20px currentColor; position: relative; }
        .marker-glow::after { content: ''; position: absolute; top: -10px; left: 50%; width: 1px; height: 10px; background: #fff; }

        /* GLITCH ANIMATION */
        .glitch { animation: glitch 3s infinite; }
        @keyframes glitch { 0%, 100% { transform: translate(0); } 90% { transform: translate(1px, -1px); } 92% { transform: translate(-1px, 1px); } }

        /* PERSPECTIVE WRAPPER */
        .map-perspective { perspective: 1000px; height: 100%; width: 100%; }
        /* Note: Leaflet doesn't play well with 3D transform, keeping it 2D but very styled. */
    </style>
</head>
<body>
    <div class="hud-container">
        <!-- TOP LOGO & STATUS -->
        <div class="hud-panel top-bar">
            <div class="logo-text glitch">ATLAS PRO // NEURAL INTERFACE</div>
            <div class="system-status">
                <button class="hud-btn" id="incident-btn" style="border-color:var(--neon-red); color:var(--neon-red); margin-right:20px; padding:5px 15px; background:transparent; cursor:pointer;">STRESS_TEST: ACCIDENT</button>
                <span id="sync-icon">●</span> SYNCED_TO_GRID [STANDALONE_V2] | <span id="clock">11:25:32</span>
            </div>
        </div>

        <!-- LEFT STATS -->
        <div class="left-col">
            <div class="hud-panel stat-card">
                <div class="card-header">🛡️ SAFETY PROTOCOL</div>
                <div class="progress-circle">
                    <svg width="120" height="120">
                        <circle class="bg" cx="60" cy="60" r="45"></circle>
                        <circle class="fg" id="safety-circle" cx="60" cy="60" r="45" style="stroke-dashoffset: 0;"></circle>
                    </svg>
                    <div class="text" id="safety-val">100%</div>
                </div>
                <div style="text-align: center; margin-top: 20px; font-size: 11px; color: var(--neon-green);">MONITORING_LOCKED_SIL-4</div>
            </div>

            <div class="hud-panel stat-card">
                <div class="card-header">⏱️ WAIT DURATION</div>
                <div class="stat-value" id="val-wait">-30%</div>
                <div class="stat-trend" style="color: var(--neon-green);">
                    ▼ 4.2s IMPROVEMENT
                </div>
                <div style="height: 60px; margin-top: 15px; border-bottom: 1px solid rgba(0, 242, 255, 0.2);">
                    <!-- Line chart placeholder -->
                </div>
            </div>

            <div class="hud-panel stat-card">
                <div class="card-header">⚙️ URBAN THROUGHPUT</div>
                <div class="stat-value" id="val-throughput">+25%</div>
                <div class="stat-trend" style="color: var(--neon-green);">
                    ▲ 42 VEH/MIN
                </div>
            </div>

            <div class="hud-panel stat-card" style="margin-top: 15px;">
                <div class="card-header">🎮 MANUAL OVERRIDE OVERRIDE</div>
                <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
                    <button class="hud-btn" id="btn-force-ns" style="border-color:var(--neon-green); color:var(--neon-green); padding:8px; background:var(--card-bg); cursor:pointer; font-family:'Orbitron'; font-weight:bold;">FORCE N-S GREEN</button>
                    <button class="hud-btn" id="btn-force-ew" style="border-color:var(--neon-cyan); color:var(--neon-cyan); padding:8px; background:var(--card-bg); cursor:pointer; font-family:'Orbitron'; font-weight:bold;">FORCE E-W GREEN</button>
                    <button class="hud-btn" id="btn-auto" style="border-color:var(--neon-purple); color:var(--neon-purple); padding:8px; background:var(--card-bg); cursor:pointer; font-family:'Orbitron'; font-weight:bold;">RESUME NEURAL AUTO</button>
                </div>
            </div>
        </div>

        <!-- CENTER MAP -->
        <div class="hud-panel center-map">
            <div class="map-overlay-text">
                > TARGET: GENERIC_INTERSECTION_01<br>
                > STATUS: ACTIVE_COORDINATION<br>
                > MODE: NEURAL_AUTONOMY
            </div>
            <div id="incident-banner" style="display:none; position:absolute; top:80px; left:50%; transform:translateX(-50%); background:rgba(255,0,85,0.8); border:2px solid #fff; padding:10px 20px; z-index:2000; font-family:'Orbitron'; font-weight:900; color:#fff; animation: blink 0.5s infinite alternate; text-align:center; width: 80%;">
                ⚠️ INCIDENT DETECTED: MAJOR BLOCKAGE AT NORTH AVENUE
            </div>
            <canvas id="intersection-canvas" width="500" height="500"></canvas>
        </div>

        <!-- RIGHT DIAGNOSIS & MUSE -->
        <div class="right-col">
            <div class="hud-panel">
                <div class="card-header">🧠 MUSE v2 METACOGNITION</div>
                
                <div class="muse-item">
                    <div class="muse-label"><span>COMPETENCE</span> <span id="muse-comp-val">94%</span></div>
                    <div class="muse-bar-bg"><div class="muse-bar-fg" id="muse-comp-bar" style="width: 94%;"></div></div>
                </div>
                
                <div class="muse-item">
                    <div class="muse-label"><span>NOVELTY DETECTOR</span> <span id="muse-nov-val">12%</span></div>
                    <div class="muse-bar-bg"><div class="muse-bar-fg" id="muse-nov-bar" style="width: 12%;"></div></div>
                </div>

                <div class="muse-item">
                    <div class="muse-label"><span>SAFETY GUARD</span> <span id="muse-safe-val">98%</span></div>
                    <div class="muse-bar-bg"><div class="muse-bar-fg" id="muse-safe-bar" style="width: 98%;"></div></div>
                </div>
            </div>

            <div class="hud-panel">
                <div class="card-header">🌐 QMIX MULTI-AGENT STATUS</div>
                <div class="qmix-grid">
                    <div class="qmix-agent">
                        <div class="a-name">AGENT_N</div>
                        <div class="a-val" id="qmix-n">SYNCED</div>
                    </div>
                    <div class="qmix-agent">
                        <div class="a-name">AGENT_S</div>
                        <div class="a-val" id="qmix-s">SYNCED</div>
                    </div>
                    <div class="qmix-agent">
                        <div class="a-name">AGENT_E</div>
                        <div class="a-val" id="qmix-e">SYNCED</div>
                    </div>
                    <div class="qmix-agent">
                        <div class="a-name">AGENT_W</div>
                        <div class="a-val" id="qmix-w">SYNCED</div>
                    </div>
                </div>
                <div style="margin-top: 15px; font-size: 11px; text-align: center; color: var(--neon-purple);">MIXER NETWORK: ACTIVE</div>
            </div>

            <div class="hud-panel" style="flex: 1;">
                <div class="card-header">🧠 STRATEGIC RATIONALE</div>
                <div class="diagnosis-text" id="ai-explanation">
                    Coordinating QMIX Joint-Action for optimal global throughput...
                </div>
            </div>
        </div>

        <!-- BOTTOM SALIENCY -->
        <div class="hud-panel bottom-panel">
            <div style="display: flex; gap: 30px; align-items: center;">
                <div class="brain-viz">
                    <svg class="brain-svg" viewBox="0 0 24 24">
                        <path d="M12,2C10.89,2 10,2.89 10,4C10,5.11 10.89,6 12,6C13.11,6 14,5.11 14,4C14,2.89 13.11,2 12,2M12,18C13.11,18 14,18.89 14,20C14,21.11 13.11,22 12,22C10.89,22 10,21.11 10,20C10,18.89 10.89,18 12,18M8,4.27C6.7,4.27 5.65,5.32 5.65,6.62C5.65,7.92 6.7,8.97 8,8.97C9.3,8.97 10.35,7.92 10.35,6.62C10.35,5.32 9.3,4.27 8,4.27M16,4.27C17.3,4.27 18.35,5.32 18.35,6.62C18.35,7.92 17.3,8.97 16,8.97C14.7,8.97 13.65,7.92 13.65,6.62C13.65,5.32 14.7,4.27 16,4.27M8,15.03C6.7,15.03 5.65,16.08 5.65,17.38C5.65,18.68 6.7,19.73 8,19.73C9.3,19.73 10.35,18.68 10.35,17.38C10.35,16.08 9.3,15.03 8,15.03M16,15.03C14.7,15.03 13.65,16.08 13.65,17.38C13.65,18.68 14.7,19.73 16,19.73C17.3,19.73 18.35,18.68 18.35,17.38C18.35,16.08 17.3,15.03 16,15.03M12,10C10.89,10 10,10.89 10,12C10,13.11 10.89,14 12,14C13.11,14 14,13.11 14,12C14,10.89 13.11,10 12,10M18,10C16.89,10 16,10.89 16,12C16,13.11 16.89,14 18,14C19.11,14 20,13.11 20,12C20,10.89 19.11,10 18,10M6,10C4.89,10 4,10.89 4,12C4,13.11 4.89,14 6,14C7.11,14 8,13.11 8,12C8,10.89 7.11,10 6,10Z" />
                    </svg>
                </div>
                <div>
                    <div class="card-header">⚡ GLOBAL TRANSFORMER SALIENCY</div>
                    <div class="neural-bars" id="neural-bars">
                        <!-- Bars by JS -->
                    </div>
                </div>
            </div>
            
            <div style="display: flex; flex-direction: column; justify-content: center;">
                <div class="card-header">🛡️ WATCHDOG INTEGRITY LOG</div>
                <div id="decision-log" style="font-size: 11px; opacity: 0.8; height: 100px; overflow-y: hidden; font-family: 'JetBrains Mono';">
                    <!-- Real-time logs -->
                </div>
            </div>
        </div>
    </div>

    <script>
        window.onerror = function(msg, url, line, col, error) {
            document.body.innerHTML += '<div style="position:fixed;top:0;left:0;background:red;color:white;z-index:9999;padding:20px;font-size:20px;">' + msg + ' at ' + line + ':' + col + '</div>';
        };

        try {
            const startTime = Date.now();
            
            // Canvas Rendering System
            const canvas = document.getElementById('intersection-canvas');
            const ctx = canvas.getContext('2d');
            const W = canvas.width;
            const H = canvas.height;
            const MID = W / 2;
            const ROAD_WIDTH = 80;
        
        let currentPhase = 'NORTH-SOUTH'; // Or 'EAST-WEST'
        let currentVehicles = [];
        let isIncidentActive = false;

        function drawIntersection() {
            // Background
            ctx.fillStyle = '#050a15';
            ctx.fillRect(0, 0, W, H);
            
            // Draw Roads (Cross)
            ctx.fillStyle = '#111827';
            ctx.fillRect(MID - ROAD_WIDTH/2, 0, ROAD_WIDTH, H); // NS Road
            ctx.fillRect(0, MID - ROAD_WIDTH/2, W, ROAD_WIDTH); // EW Road
            
            // Dashed lines
            ctx.strokeStyle = '#334155';
            ctx.lineWidth = 2;
            ctx.setLineDash([10, 15]);
            
            // NS Dashed
            ctx.beginPath();
            ctx.moveTo(MID, 0); ctx.lineTo(MID, MID - ROAD_WIDTH/2);
            ctx.moveTo(MID, MID + ROAD_WIDTH/2); ctx.lineTo(MID, H);
            ctx.stroke();
            
            // EW Dashed
            ctx.beginPath();
            ctx.moveTo(0, MID); ctx.lineTo(MID - ROAD_WIDTH/2, MID);
            ctx.moveTo(MID + ROAD_WIDTH/2, MID); ctx.lineTo(W, MID);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Stop Lines & Traffic Lights
            const nsColor = currentPhase === 'NORTH-SOUTH' ? '#39ff14' : '#ff0055';
            const ewColor = currentPhase === 'EAST-WEST' ? '#39ff14' : '#ff0055';
            
            ctx.lineWidth = 4;
            // North Stop
            ctx.strokeStyle = nsColor;
            ctx.beginPath(); ctx.moveTo(MID - ROAD_WIDTH/2, MID - ROAD_WIDTH/2); ctx.lineTo(MID, MID - ROAD_WIDTH/2); ctx.stroke();
            // South Stop
            ctx.beginPath(); ctx.moveTo(MID, MID + ROAD_WIDTH/2); ctx.lineTo(MID + ROAD_WIDTH/2, MID + ROAD_WIDTH/2); ctx.stroke();
            
            // East Stop
            ctx.strokeStyle = ewColor;
            ctx.beginPath(); ctx.moveTo(MID + ROAD_WIDTH/2, MID - ROAD_WIDTH/2); ctx.lineTo(MID + ROAD_WIDTH/2, MID); ctx.stroke();
            // West Stop
            ctx.beginPath(); ctx.moveTo(MID - ROAD_WIDTH/2, MID); ctx.lineTo(MID - ROAD_WIDTH/2, MID + ROAD_WIDTH/2); ctx.stroke();
            
            // Draw Vehicles
            ctx.fillStyle = '#00f2ff';
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#00f2ff';
            // Let's compute a dynamic scale if cars go off-bounds
            let maxCoord = 100;
            currentVehicles.forEach(v => {
                const absX = Math.abs(v.x);
                const absY = Math.abs(v.y);
                if (absX > maxCoord) maxCoord = absX;
                if (absY > maxCoord) maxCoord = absY;
            });
            // Add a 10% padding
            maxCoord = maxCoord * 1.1;

            currentVehicles.forEach(v => {
                // Map logical physics coords (-maxCoord to maxCoord) to Canvas (0 to 500)
                const cx = (v.x / maxCoord) * 250 + 250;
                const cy = (-v.y / maxCoord) * 250 + 250;
                
                // Draw car as a glowing rectangle
                ctx.save();
                ctx.translate(cx, cy);
                
                // Rotation based on origin
                if (v.origin === 'N') ctx.rotate(0);
                else if (v.origin === 'S') ctx.rotate(Math.PI);
                else if (v.origin === 'E') ctx.rotate(Math.PI / 2);
                else if (v.origin === 'W') ctx.rotate(-Math.PI / 2);
                
                ctx.fillRect(-6, -10, 12, 20); // Car shape
                ctx.restore();
            });
            ctx.shadowBlur = 0;
            
            // Incident graphic
            if (isIncidentActive) {
                ctx.fillStyle = 'rgba(255,0,85,0.4)';
                ctx.fillRect(MID - ROAD_WIDTH/2, 20, ROAD_WIDTH, 80);
                
                // Draw X
                ctx.strokeStyle = '#ff0055';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(MID - 15, 60 - 15); ctx.lineTo(MID + 15, 60 + 15);
                ctx.moveTo(MID + 15, 60 - 15); ctx.lineTo(MID - 15, 60 + 15);
                ctx.stroke();
            }
        }
        
        // Render loop for smoothness regardless of websocket rate
        function renderLoop() {
            try {
                drawIntersection();
            } catch (e) {
                console.error(e);
                ctx.fillStyle = 'red';
                ctx.fillRect(0,0,W,H);
                ctx.fillStyle = 'white';
                ctx.fillText(e.toString(), 20, 20);
            }
            requestAnimationFrame(renderLoop);
        }
        requestAnimationFrame(renderLoop);

        function updateUI(data) {
            const now = new Date();
            document.getElementById('clock').textContent = now.toLocaleTimeString();

            // Stats
            if (data.vehicles !== undefined) {
                const throughputVal = Math.floor(data.vehicles * 1.5);
                document.getElementById('val-throughput').textContent = `+${throughputVal}%`;
            }
            if (data.safety_score) {
                document.getElementById('safety-val').textContent = data.safety_score + '%';
                const offset = 283 - (283 * data.safety_score / 100);
                document.getElementById('safety-circle').style.strokeDashoffset = offset;
            }

            // Diagnosis & Wave
            if (data.explanation) {
                document.getElementById('ai-explanation').textContent = data.explanation;
                const waveCont = document.getElementById('wave-bars');
                waveCont.innerHTML = '';
                for(let i=0; i<30; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'wave-bar';
                    bar.style.animationDelay = (i * 0.05) + 's';
                    waveCont.appendChild(bar);
                }
            }

            // Saliency Bars
            if (data.saliency) {
                const neuralCont = document.getElementById('neural-bars');
                neuralCont.innerHTML = '';
                data.saliency.weights.slice(0, 15).forEach(w => {
                    const bar = document.createElement('div');
                    bar.className = 'n-bar';
                    bar.style.height = (w * 100) + 'px';
                    neuralCont.appendChild(bar);
                });
            }

            // Decision Log
            if (data.new_decision) {
                const log = document.getElementById('decision-log');
                const p = document.createElement('div');
                p.textContent = `[SYNC] PHASE_SHIFT: ${data.current_phase} ACTIVATED AT ${now.toLocaleTimeString()}`;
                log.insertBefore(p, log.firstChild);
                if (log.children.length > 5) log.removeChild(log.lastChild);
            }

            // MUSE Metacognition Updates
            if (data.muse) {
                document.getElementById('muse-comp-val').textContent = data.muse.competence + '%';
                document.getElementById('muse-comp-bar').style.width = data.muse.competence + '%';
                
                document.getElementById('muse-nov-val').textContent = data.muse.novelty + '%';
                const novBar = document.getElementById('muse-nov-bar');
                novBar.style.width = data.muse.novelty + '%';
                novBar.className = 'muse-bar-fg' + (data.muse.novelty > 70 ? ' danger' : (data.muse.novelty > 40 ? ' warning' : ''));

                document.getElementById('muse-safe-val').textContent = data.muse.safety + '%';
                document.getElementById('muse-safe-bar').style.width = data.muse.safety + '%';
            }

            // QMIX Multi-Agent Updates
            if (data.qmix) {
                const q_colors = {'SYNCED': 'var(--neon-green)', 'LAGGING': '#ffbf00', 'FAULT': 'var(--neon-red)'};
                ['n', 's', 'e', 'w'].forEach(dir => {
                    const el = document.getElementById('qmix-' + dir);
                    const state = data.qmix[dir] || 'SYNCED';
                    if (el) {
                        el.textContent = state;
                        el.style.color = q_colors[state] || q_colors['SYNCED'];
                    }
                });
            }

            // Store global state for render loop
            if (data.current_phase) currentPhase = data.current_phase;
            if (data.vehicles_pos) currentVehicles = data.vehicles_pos;
            if (data.incident_active !== undefined) isIncidentActive = data.incident_active;
            
            // Incident UI
            const banner = document.getElementById('incident-banner');
            const btn = document.getElementById('incident-btn');
            if (data.incident_active) {
                banner.style.display = 'block';
                btn.textContent = 'STOP_SIMULATION';
                btn.style.background = 'var(--neon-red)';
                btn.style.color = '#fff';
            } else {
                banner.style.display = 'none';
                btn.textContent = 'STRESS_TEST: ACCIDENT';
                btn.style.background = 'transparent';
                btn.style.color = 'var(--neon-red)';
            }
        }

        document.getElementById('incident-btn').onclick = function() {
            fetch('/api/incident', {method: 'POST'})
                .then(r => r.json())
                .then(d => console.log('Incident Toggled:', d.active));
        };

        function forcePhase(phaseStr) {
            fetch('/api/force_phase', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phase: phaseStr})
            }).then(r => r.json()).then(console.log);
        }

        document.getElementById('btn-force-ns').onclick = () => forcePhase('NS');
        document.getElementById('btn-force-ew').onclick = () => forcePhase('EW');
        document.getElementById('btn-auto').onclick = () => forcePhase('AUTO');

        function connectWS() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.onmessage = (e) => updateUI(JSON.parse(e.data));
            ws.onclose = () => {
                setTimeout(connectWS, 2000);
                document.getElementById('sync-icon').style.color = 'gray';
            };
            ws.onopen = () => {
                document.getElementById('sync-icon').style.color = 'var(--neon-green)';
            };
        }

        window.onload = () => {
            connectWS();
        };

        } catch (globalErr) {
            document.body.innerHTML += '<div style="position:fixed;top:50px;left:0;background:red;color:white;z-index:9999;padding:20px;font-size:20px;">Global Error: ' + globalErr + '</div>';
        }
    </script>
</body>
</html>"""

_dashboard_data = {
    "uptime_str": "00:00:00",
    "total_decisions": 0,
    "vehicles": 0,
    "wait_time": "-30%",
    "throughput": "+25%",
    "safety_score": 100,
    "current_phase": "N-S",
    "new_decision": False,
    "saliency": {"weights": [0.1]*20, "labels": ["F"+str(i) for i in range(20)]},
    "explanation": "READY FOR URBAN SYNCHRONIZATION.",
    "map_data": {}
}

_data_lock = threading.Lock()

def start_dashboard(host: str = "0.0.0.0", port: int = 8888):
    if not HAS_FASTAPI: return
    app = FastAPI()
    active_ws = []

    @app.get("/", response_class=HTMLResponse)
    async def index(): return create_dashboard_html()

    @app.post("/api/update")
    async def update(data: dict):
        with _data_lock:
            _dashboard_data.update(data)
            for ws in active_ws:
                try: await ws.send_json(_dashboard_data)
                except: active_ws.remove(ws)
        return {"ok": True}

    @app.post("/api/incident")
    async def toggle_incident():
        with _data_lock:
            _dashboard_data["incident_active"] = not _dashboard_data.get("incident_active", False)
            return {"active": _dashboard_data["incident_active"]}

    @app.get("/api/incident_status")
    async def get_incident():
        return {
            "active": _dashboard_data.get("incident_active", False),
            "force_phase": _dashboard_data.get("force_phase", "AUTO")
        }

    @app.post("/api/force_phase")
    async def override_phase(data: dict):
        with _data_lock:
            _dashboard_data["force_phase"] = data.get("phase", "AUTO")
            return {"status": "ok", "force_phase": _dashboard_data["force_phase"]}

    @app.post("/api/sumo_launch")
    async def sumo_launch():
        import subprocess
        import os
        try:
            # Assuming dashboard is run from demo_live_city.py in project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            script_path = os.path.join(base_dir, "demo_sumo_real.py")
            
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NEW_CONSOLE

            subprocess.Popen([sys.executable, script_path], cwd=base_dir, creationflags=creationflags)
            return {"status": "launched"}
        except Exception as e:
            return {"error": str(e)}

    @app.websocket("/ws")
    async def websocket_ep(ws: WebSocket):
        await ws.accept()
        active_ws.append(ws)
        try:
            while True: await ws.receive_text()
        except: 
            if ws in active_ws: active_ws.remove(ws)

    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    start_dashboard()
