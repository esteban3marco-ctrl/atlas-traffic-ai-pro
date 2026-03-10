"""
ATLAS Pro — Dashboard & API Backend
======================================
FastAPI backend with WebSocket for real-time training metrics.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger("ATLAS.Dashboard")

# Global state for WebSocket broadcasting
_training_state = {
    "is_training": False,
    "episode": 0,
    "total_episodes": 0,
    "reward": 0.0,
    "mean_reward": 0.0,
    "best_reward": 0.0,
    "loss": 0.0,
    "wait_time": 0.0,
    "throughput": 0,
    "max_queue": 0,
    "phase": "N-S",
    "queues": {"N": 0, "S": 0, "E": 0, "W": 0},
    "algorithm": "dueling_ddqn",
    "device": "cpu",
    "elapsed_time": 0,
}

_connected_clients: List = []


def create_app() -> 'FastAPI':
    """Create the FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn websockets")
    
    app = FastAPI(
        title="ATLAS Pro — Traffic AI Dashboard",
        description="Real-time monitoring and control of ATLAS traffic AI system",
        version="2.0.0",
    )
    
    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # === Routes ===
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return HTMLResponse("<h1>ATLAS Pro Dashboard</h1><p>Static files not found.</p>")
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket):
        await websocket.accept()
        _connected_clients.append(websocket)
        logger.info(f"📡 Client connected ({len(_connected_clients)} total)")
        
        try:
            # Send initial state
            await websocket.send_json(_training_state)
            
            while True:
                # Keep connection alive and listen for commands
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
        except WebSocketDisconnect:
            _connected_clients.remove(websocket)
            logger.info(f"📡 Client disconnected ({len(_connected_clients)} total)")
        except Exception:
            if websocket in _connected_clients:
                _connected_clients.remove(websocket)
    
    @app.get("/api/status")
    async def get_status():
        return _training_state
    
    @app.get("/api/models")
    async def list_models():
        checkpoint_dir = "checkpoints"
        models = []
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith(".pt"):
                    path = os.path.join(checkpoint_dir, f)
                    models.append({
                        "name": f,
                        "size_mb": os.path.getsize(path) / 1024 / 1024,
                        "modified": os.path.getmtime(path),
                    })
        return {"models": models}
    
    @app.get("/api/config")
    async def get_config():
        config_path = "checkpoints/config_latest.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {"config": f.read()}
        return {"config": None}
    
    @app.post("/api/predict")
    async def predict(state: Dict):
        """Get action prediction from loaded model."""
        return {"action": 0, "message": "Model must be loaded via CLI first"}
    
    return app


async def broadcast_update(data: Dict):
    """Broadcast training update to all connected WebSocket clients."""
    _training_state.update(data)
    
    disconnected = []
    for ws in _connected_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    
    for ws in disconnected:
        _connected_clients.remove(ws)


def update_training_state(data: Dict):
    """Synchronous update for use from training thread."""
    _training_state.update(data)
    
    # Best-effort async broadcast
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast_update(data))
    except RuntimeError:
        pass


def run_dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Start the dashboard server."""
    app = create_app()
    logger.info(f"🌐 Dashboard starting at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# Init for dashboard sub-package
__all__ = ["create_app", "run_dashboard", "update_training_state", "broadcast_update"]
