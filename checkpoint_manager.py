"""
ATLAS Pro - Sistema de Checkpoints, Versionado y Rollback
==========================================================
Gestión profesional de modelos PyTorch:
- Checkpoints automáticos durante entrenamiento
- Versionado semántico de modelos
- Rollback a versiones anteriores
- Comparación de rendimiento entre versiones
- Registro de metadatos de entrenamiento
"""

import os
import json
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("ATLAS.CheckpointManager")

try:
    import torch
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False


@dataclass
class CheckpointMetadata:
    """Metadatos de un checkpoint"""
    version: str
    timestamp: str
    episode: int
    step: int
    algorithm: str
    avg_reward: float
    avg_wait_time: float
    avg_throughput: float
    avg_queue_length: float
    loss: float
    epsilon: float
    config: Dict
    description: str
    tags: List[str]
    hash: str = ""

    def to_dict(self):
        return asdict(self)


class CheckpointManager:
    """
    Gestor de checkpoints con versionado semántico y rollback.

    Estructura de directorio:
    checkpoints/
    ├── registry.json          # Registro de todos los checkpoints
    ├── best_model.pt          # Mejor modelo (symlink/copy)
    ├── latest_model.pt        # Último modelo (symlink/copy)
    ├── v1.0.0/
    │   ├── model.pt
    │   ├── metadata.json
    │   └── metrics.json
    ├── v1.0.1/
    │   ├── model.pt
    │   ├── metadata.json
    │   └── metrics.json
    └── ...
    """

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 max_checkpoints: int = 20,
                 keep_best_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best_n = keep_best_n
        self.registry_path = self.checkpoint_dir / "registry.json"
        self.registry = self._load_registry()

        logger.info(f"CheckpointManager inicializado: {checkpoint_dir}")
        logger.info(f"  Checkpoints existentes: {len(self.registry.get('checkpoints', []))}")

    def _load_registry(self) -> Dict:
        """Carga el registro de checkpoints"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {
            'checkpoints': [],
            'best_version': None,
            'latest_version': None,
            'creation_date': datetime.now().isoformat()
        }

    def _save_registry(self):
        """Guarda el registro"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def _compute_hash(self, model_path: str) -> str:
        """Calcula hash SHA256 del modelo"""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def _next_version(self, bump: str = "patch") -> str:
        """Calcula siguiente versión semántica"""
        if not self.registry['checkpoints']:
            return "1.0.0"

        last = self.registry['checkpoints'][-1]['version']
        parts = [int(x) for x in last.split('.')]

        if bump == "major":
            return f"{parts[0] + 1}.0.0"
        elif bump == "minor":
            return f"{parts[0]}.{parts[1] + 1}.0"
        else:
            return f"{parts[0]}.{parts[1]}.{parts[2] + 1}"

    def save_checkpoint(self, model, optimizer=None,
                       episode: int = 0, step: int = 0,
                       algorithm: str = "dueling_ddqn",
                       metrics: Dict = None,
                       config: Dict = None,
                       description: str = "",
                       tags: List[str] = None,
                       version_bump: str = "patch") -> str:
        """
        Guarda un checkpoint con metadatos completos.

        Args:
            model: Modelo PyTorch (nn.Module) o dict con state_dicts
            optimizer: Optimizador (opcional)
            episode: Número de episodio
            step: Número de paso global
            algorithm: Nombre del algoritmo
            metrics: Dict con métricas (avg_reward, avg_wait_time, etc.)
            config: Configuración de entrenamiento
            description: Descripción del checkpoint
            tags: Tags para categorización
            version_bump: "major", "minor", o "patch"

        Returns:
            Version string del checkpoint guardado
        """
        metrics = metrics or {}
        config = config or {}
        tags = tags or []

        version = self._next_version(version_bump)
        version_dir = self.checkpoint_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Guardar modelo
        model_path = version_dir / "model.pt"
        save_dict = {
            'version': version,
            'episode': episode,
            'step': step,
            'algorithm': algorithm,
            'timestamp': datetime.now().isoformat()
        }

        if isinstance(model, dict):
            save_dict.update(model)
        elif TORCH_DISPONIBLE and isinstance(model, torch.nn.Module):
            save_dict['model_state_dict'] = model.state_dict()
        else:
            save_dict['model'] = model

        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()

        if TORCH_DISPONIBLE:
            torch.save(save_dict, model_path)
        else:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(save_dict, f)

        # Calcular hash
        model_hash = self._compute_hash(str(model_path))

        # Metadata
        metadata = CheckpointMetadata(
            version=version,
            timestamp=datetime.now().isoformat(),
            episode=episode,
            step=step,
            algorithm=algorithm,
            avg_reward=metrics.get('avg_reward', 0.0),
            avg_wait_time=metrics.get('avg_wait_time', 0.0),
            avg_throughput=metrics.get('avg_throughput', 0.0),
            avg_queue_length=metrics.get('avg_queue_length', 0.0),
            loss=metrics.get('loss', 0.0),
            epsilon=metrics.get('epsilon', 0.0),
            config=config,
            description=description,
            tags=tags,
            hash=model_hash
        )

        # Guardar metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # Guardar métricas detalladas
        with open(version_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Actualizar registro
        self.registry['checkpoints'].append(metadata.to_dict())
        self.registry['latest_version'] = version

        # Actualizar mejor modelo
        if self._is_best(metrics):
            self.registry['best_version'] = version
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(str(model_path), str(best_path))
            logger.info(f"  ⭐ Nuevo mejor modelo: v{version}")

        # Copiar como latest
        latest_path = self.checkpoint_dir / "latest_model.pt"
        shutil.copy2(str(model_path), str(latest_path))

        self._save_registry()
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint guardado: v{version} (ep={episode}, step={step})")
        return version

    def _is_best(self, metrics: Dict) -> bool:
        """Determina si las métricas son las mejores hasta ahora"""
        if not self.registry['best_version']:
            return True

        best_meta = self.get_metadata(self.registry['best_version'])
        if not best_meta:
            return True

        current_reward = metrics.get('avg_reward', float('-inf'))
        best_reward = best_meta.get('avg_reward', float('-inf'))
        return current_reward > best_reward

    def load_checkpoint(self, version: str = None, load_best: bool = False,
                       load_latest: bool = False) -> Optional[Dict]:
        """
        Carga un checkpoint.

        Args:
            version: Versión específica (e.g., "1.2.3")
            load_best: Cargar mejor modelo
            load_latest: Cargar último modelo

        Returns:
            Dict con model_state_dict, optimizer_state_dict, metadata
        """
        if load_best:
            model_path = self.checkpoint_dir / "best_model.pt"
            version = self.registry.get('best_version')
        elif load_latest:
            model_path = self.checkpoint_dir / "latest_model.pt"
            version = self.registry.get('latest_version')
        elif version:
            model_path = self.checkpoint_dir / f"v{version}" / "model.pt"
        else:
            logger.error("Especifica version, load_best, o load_latest")
            return None

        if not model_path.exists():
            logger.error(f"Checkpoint no encontrado: {model_path}")
            return None

        if TORCH_DISPONIBLE:
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)

        logger.info(f"Checkpoint cargado: v{version}")
        return checkpoint

    def rollback(self, version: str) -> bool:
        """
        Rollback a una versión anterior.
        Marca la versión como la nueva 'latest' y copia el modelo.
        """
        version_dir = self.checkpoint_dir / f"v{version}"
        if not version_dir.exists():
            logger.error(f"Versión {version} no encontrada")
            return False

        model_path = version_dir / "model.pt"
        latest_path = self.checkpoint_dir / "latest_model.pt"
        shutil.copy2(str(model_path), str(latest_path))

        self.registry['latest_version'] = version
        self._save_registry()

        logger.info(f"Rollback a v{version} completado")
        return True

    def get_metadata(self, version: str) -> Optional[Dict]:
        """Obtiene metadatos de una versión"""
        for ckpt in self.registry['checkpoints']:
            if ckpt['version'] == version:
                return ckpt
        return None

    def list_checkpoints(self) -> List[Dict]:
        """Lista todos los checkpoints con sus metadatos"""
        return self.registry['checkpoints']

    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        """Compara métricas entre dos versiones"""
        meta_a = self.get_metadata(version_a)
        meta_b = self.get_metadata(version_b)

        if not meta_a or not meta_b:
            return {"error": "Versión no encontrada"}

        metrics_to_compare = ['avg_reward', 'avg_wait_time', 'avg_throughput',
                             'avg_queue_length', 'loss']

        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'metrics': {}
        }

        for metric in metrics_to_compare:
            val_a = meta_a.get(metric, 0)
            val_b = meta_b.get(metric, 0)
            diff = val_b - val_a
            pct = (diff / abs(val_a) * 100) if val_a != 0 else 0

            comparison['metrics'][metric] = {
                'version_a': val_a,
                'version_b': val_b,
                'difference': diff,
                'percentage_change': round(pct, 2)
            }

        return comparison

    def get_best_n(self, n: int = 5, metric: str = 'avg_reward') -> List[Dict]:
        """Obtiene los N mejores checkpoints según una métrica"""
        checkpoints = self.registry['checkpoints']
        reverse = metric in ['avg_reward', 'avg_throughput']  # Mayor es mejor
        sorted_ckpts = sorted(checkpoints, key=lambda x: x.get(metric, 0), reverse=reverse)
        return sorted_ckpts[:n]

    def _cleanup_old_checkpoints(self):
        """Limpia checkpoints antiguos manteniendo los mejores"""
        checkpoints = self.registry['checkpoints']
        if len(checkpoints) <= self.max_checkpoints:
            return

        # Mantener mejores N y más recientes N
        best = set(c['version'] for c in self.get_best_n(self.keep_best_n))
        recent = set(c['version'] for c in checkpoints[-self.keep_best_n:])
        keep = best | recent

        # Eliminar el resto
        to_remove = [c for c in checkpoints if c['version'] not in keep]
        for ckpt in to_remove[:-1]:  # Siempre mantener al menos max_checkpoints - keep
            version_dir = self.checkpoint_dir / f"v{ckpt['version']}"
            if version_dir.exists():
                shutil.rmtree(str(version_dir))
                logger.info(f"Checkpoint eliminado: v{ckpt['version']}")

        self.registry['checkpoints'] = [c for c in checkpoints if c['version'] in keep or c in to_remove[-1:]]
        self._save_registry()

    def export_summary(self) -> Dict:
        """Exporta resumen del historial de checkpoints"""
        checkpoints = self.registry['checkpoints']
        if not checkpoints:
            return {"message": "No hay checkpoints"}

        rewards = [c.get('avg_reward', 0) for c in checkpoints]
        return {
            'total_checkpoints': len(checkpoints),
            'best_version': self.registry['best_version'],
            'latest_version': self.registry['latest_version'],
            'best_reward': max(rewards) if rewards else 0,
            'reward_trend': rewards[-10:],
            'algorithms_used': list(set(c.get('algorithm', '') for c in checkpoints)),
            'total_episodes': checkpoints[-1].get('episode', 0) if checkpoints else 0,
            'total_steps': checkpoints[-1].get('step', 0) if checkpoints else 0
        }


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def ejemplo_checkpoint_manager():
    """Demo del checkpoint manager"""
    print("\n" + "=" * 70)
    print("💾 ATLAS Pro - Sistema de Checkpoints")
    print("=" * 70)

    manager = CheckpointManager("checkpoints_demo")

    if TORCH_DISPONIBLE:
        import torch.nn as nn

        # Crear modelo dummy
        model = nn.Sequential(
            nn.Linear(26, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )
        optimizer = torch.optim.Adam(model.parameters())

        # Simular entrenamiento
        for ep in range(5):
            metrics = {
                'avg_reward': -50 + ep * 15 + np.random.uniform(-5, 5),
                'avg_wait_time': 120 - ep * 20,
                'avg_throughput': 50 + ep * 10,
                'avg_queue_length': 30 - ep * 5,
                'loss': 0.5 - ep * 0.08,
                'epsilon': max(0.01, 1.0 - ep * 0.2)
            }

            version = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=(ep + 1) * 50,
                step=(ep + 1) * 5000,
                algorithm="dueling_ddqn",
                metrics=metrics,
                description=f"Entrenamiento episodio {(ep+1)*50}",
                tags=["training", "simple_intersection"]
            )
            print(f"  Guardado v{version}: reward={metrics['avg_reward']:.1f}")

        # Listar
        print(f"\n📋 Checkpoints: {len(manager.list_checkpoints())}")

        # Comparar
        comparison = manager.compare_versions("1.0.0", "1.0.4")
        print(f"\n📊 Comparación v1.0.0 vs v1.0.4:")
        for metric, vals in comparison.get('metrics', {}).items():
            print(f"   {metric}: {vals['percentage_change']:+.1f}%")

        # Rollback
        manager.rollback("1.0.2")
        print(f"\n🔄 Rollback a v1.0.2")

        # Resumen
        summary = manager.export_summary()
        print(f"\n📈 Resumen: {summary}")

        # Limpiar demo
        shutil.rmtree("checkpoints_demo", ignore_errors=True)
    else:
        print("❌ PyTorch necesario para demo completa")

    print("\n✅ Demo completada")


if __name__ == "__main__":
    import numpy as np
    ejemplo_checkpoint_manager()
