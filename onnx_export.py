"""
ATLAS Pro - Exportación ONNX y Pipeline de Cuantización
========================================================
Optimización de modelos para edge deployment:
- Exportación a formato ONNX
- Cuantización INT8 para hardware embebido
- Cuantización FP16 para GPUs
- Benchmarking de latencia pre/post optimización
- Validación de precisión post-cuantización
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger("ATLAS.ONNXExport")

try:
    import torch
    import torch.nn as nn
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False

try:
    import onnx
    ONNX_DISPONIBLE = True
except ImportError:
    ONNX_DISPONIBLE = False

try:
    import onnxruntime as ort
    ORT_DISPONIBLE = True
except ImportError:
    ORT_DISPONIBLE = False


@dataclass
class BenchmarkResult:
    """Resultado de benchmark de latencia"""
    model_name: str
    format: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops: float
    model_size_mb: float
    precision_loss: float


class ONNXExporter:
    """
    Exportador de modelos PyTorch a ONNX con optimización.
    """

    def __init__(self, output_dir: str = "models_optimized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ONNXExporter inicializado: {output_dir}")

    def export_to_onnx(self, model: nn.Module, input_shape: Tuple[int, ...],
                       model_name: str = "atlas_dqn",
                       opset_version: int = 17,
                       dynamic_axes: Dict = None) -> Optional[str]:
        """
        Exporta modelo PyTorch a ONNX.

        Args:
            model: Modelo PyTorch
            input_shape: Forma del input (e.g., (1, 26))
            model_name: Nombre del archivo
            opset_version: Versión del opset ONNX
            dynamic_axes: Ejes dinámicos para batching

        Returns:
            Ruta al archivo ONNX exportado
        """
        if not TORCH_DISPONIBLE:
            logger.error("PyTorch no disponible")
            return None

        model.eval()
        dummy_input = torch.randn(*input_shape)

        output_path = str(self.output_dir / f"{model_name}.onnx")

        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )

            # Verificar modelo exportado
            if ONNX_DISPONIBLE:
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info(f"Modelo ONNX verificado: {output_path}")

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Exportado: {output_path} ({size_mb:.2f} MB)")
            return output_path

        except Exception as e:
            logger.error(f"Error exportando a ONNX: {e}")
            return None

    def quantize_int8(self, onnx_path: str,
                      calibration_data: np.ndarray = None,
                      model_name: str = None) -> Optional[str]:
        """
        Cuantización INT8 para edge deployment.
        Reduce tamaño ~4x y mejora latencia en CPU.

        Args:
            onnx_path: Ruta al modelo ONNX
            calibration_data: Datos para calibración (opcional)
            model_name: Nombre del modelo cuantizado

        Returns:
            Ruta al modelo cuantizado
        """
        if not ORT_DISPONIBLE:
            logger.error("onnxruntime no disponible para cuantización")
            return None

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            if model_name is None:
                base = Path(onnx_path).stem
                model_name = f"{base}_int8"

            output_path = str(self.output_dir / f"{model_name}.onnx")

            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True
            )

            size_orig = os.path.getsize(onnx_path) / (1024 * 1024)
            size_quant = os.path.getsize(output_path) / (1024 * 1024)
            reduction = (1 - size_quant / size_orig) * 100

            logger.info(f"Cuantizado INT8: {output_path}")
            logger.info(f"  Tamaño: {size_orig:.2f}MB → {size_quant:.2f}MB ({reduction:.1f}% reducción)")

            return output_path

        except Exception as e:
            logger.error(f"Error en cuantización INT8: {e}")
            return None

    def quantize_fp16(self, onnx_path: str, model_name: str = None) -> Optional[str]:
        """
        Cuantización FP16 para GPUs.
        Reduce tamaño ~2x con pérdida mínima de precisión.
        """
        if not ONNX_DISPONIBLE:
            logger.error("ONNX no disponible")
            return None

        try:
            from onnx import numpy_helper
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(model)

            if model_name is None:
                base = Path(onnx_path).stem
                model_name = f"{base}_fp16"

            output_path = str(self.output_dir / f"{model_name}.onnx")
            onnx.save(model_fp16, output_path)

            logger.info(f"Cuantizado FP16: {output_path}")
            return output_path

        except ImportError:
            logger.warning("onnxconverter-common no disponible, usando método alternativo")
            return self._manual_fp16_conversion(onnx_path, model_name)

        except Exception as e:
            logger.error(f"Error en cuantización FP16: {e}")
            return None

    def _manual_fp16_conversion(self, onnx_path: str, model_name: str = None) -> Optional[str]:
        """Conversión manual a FP16 sin dependencias extra"""
        if not ONNX_DISPONIBLE:
            return None

        try:
            model = onnx.load(onnx_path)

            # Convertir inicializadores a FP16
            for initializer in model.graph.initializer:
                if initializer.data_type == onnx.TensorProto.FLOAT:
                    arr = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    arr_fp16 = arr.astype(np.float16)
                    initializer.raw_data = arr_fp16.tobytes()
                    initializer.data_type = onnx.TensorProto.FLOAT16

            if model_name is None:
                base = Path(onnx_path).stem
                model_name = f"{base}_fp16"

            output_path = str(self.output_dir / f"{model_name}.onnx")
            onnx.save(model, output_path)
            return output_path

        except Exception as e:
            logger.error(f"Error en conversión FP16 manual: {e}")
            return None


class InferenceEngine:
    """
    Motor de inferencia optimizado usando ONNX Runtime.
    """

    def __init__(self, model_path: str, use_gpu: bool = False):
        if not ORT_DISPONIBLE:
            raise ImportError("onnxruntime no disponible")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu \
            else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"InferenceEngine: {model_path}")
        logger.info(f"  Provider: {self.session.get_providers()}")

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Inferencia rápida"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        state = state.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: state})
        return result[0]

    def predict_action(self, state: np.ndarray) -> int:
        """Retorna la mejor acción"""
        q_values = self.predict(state)
        return int(np.argmax(q_values[0]))


class ModelBenchmark:
    """
    Benchmark de rendimiento para comparar modelos.
    """

    def __init__(self):
        self.results = []

    def benchmark_pytorch(self, model: nn.Module, input_shape: Tuple[int, ...],
                         n_runs: int = 1000, warmup: int = 100,
                         model_name: str = "pytorch") -> BenchmarkResult:
        """Benchmark de modelo PyTorch"""
        if not TORCH_DISPONIBLE:
            raise ImportError("PyTorch no disponible")

        model.eval()
        dummy = torch.randn(*input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                model(dummy)
                latencies.append((time.perf_counter() - start) * 1000)

        return self._compute_result(model_name, "pytorch", latencies, 0)

    def benchmark_onnx(self, model_path: str, input_shape: Tuple[int, ...],
                      n_runs: int = 1000, warmup: int = 100,
                      model_name: str = "onnx") -> BenchmarkResult:
        """Benchmark de modelo ONNX"""
        engine = InferenceEngine(model_path)
        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            engine.predict(dummy)

        # Benchmark
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            engine.predict(dummy)
            latencies.append((time.perf_counter() - start) * 1000)

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return self._compute_result(model_name, "onnx", latencies, size_mb)

    def _compute_result(self, name: str, fmt: str,
                       latencies: List[float], size_mb: float) -> BenchmarkResult:
        latencies = np.array(latencies)
        result = BenchmarkResult(
            model_name=name,
            format=fmt,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_ops=1000.0 / np.mean(latencies),
            model_size_mb=size_mb,
            precision_loss=0.0
        )
        self.results.append(result)
        return result

    def compare_precision(self, model_pytorch: nn.Module,
                         onnx_path: str, input_shape: Tuple[int, ...],
                         n_samples: int = 1000) -> float:
        """Compara precisión entre PyTorch y ONNX"""
        if not TORCH_DISPONIBLE or not ORT_DISPONIBLE:
            return 0.0

        model_pytorch.eval()
        engine = InferenceEngine(onnx_path)

        errors = []
        for _ in range(n_samples):
            x = np.random.randn(*input_shape).astype(np.float32)
            x_torch = torch.FloatTensor(x)

            with torch.no_grad():
                pytorch_out = model_pytorch(x_torch).numpy()
            onnx_out = engine.predict(x)

            error = np.mean(np.abs(pytorch_out - onnx_out))
            errors.append(error)

        avg_error = np.mean(errors)
        logger.info(f"Error medio PyTorch vs ONNX: {avg_error:.6f}")
        return float(avg_error)

    def summary(self) -> str:
        """Resumen de todos los benchmarks"""
        lines = ["\n📊 Benchmark Summary", "=" * 70]
        for r in self.results:
            lines.append(f"\n  {r.model_name} ({r.format}):")
            lines.append(f"    Avg latency:  {r.avg_latency_ms:.3f} ms")
            lines.append(f"    P95 latency:  {r.p95_latency_ms:.3f} ms")
            lines.append(f"    Throughput:   {r.throughput_ops:.0f} ops/sec")
            if r.model_size_mb > 0:
                lines.append(f"    Model size:   {r.model_size_mb:.2f} MB")
        return "\n".join(lines)


# =============================================================================
# PIPELINE COMPLETO DE OPTIMIZACIÓN
# =============================================================================

def optimize_model_pipeline(model: nn.Module, state_dim: int = 26,
                           model_name: str = "atlas_dqn",
                           output_dir: str = "models_optimized") -> Dict:
    """
    Pipeline completo de optimización:
    1. Export a ONNX
    2. Cuantización INT8
    3. Benchmark comparativo
    4. Validación de precisión
    """
    if not TORCH_DISPONIBLE:
        return {"error": "PyTorch no disponible"}

    results = {}
    exporter = ONNXExporter(output_dir)
    benchmark = ModelBenchmark()

    print(f"\n🔧 Pipeline de optimización: {model_name}")
    print("=" * 60)

    # 1. Export ONNX
    print("\n1️⃣  Exportando a ONNX...")
    onnx_path = exporter.export_to_onnx(model, (1, state_dim), model_name)
    if onnx_path:
        results['onnx_path'] = onnx_path
        results['onnx_size_mb'] = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   ✅ {results['onnx_size_mb']:.2f} MB")

    # 2. Cuantización INT8
    if onnx_path:
        print("\n2️⃣  Cuantizando a INT8...")
        int8_path = exporter.quantize_int8(onnx_path)
        if int8_path:
            results['int8_path'] = int8_path
            results['int8_size_mb'] = os.path.getsize(int8_path) / (1024 * 1024)
            reduction = (1 - results['int8_size_mb'] / results['onnx_size_mb']) * 100
            print(f"   ✅ {results['int8_size_mb']:.2f} MB ({reduction:.1f}% reducción)")

    # 3. Benchmark
    if ORT_DISPONIBLE and onnx_path:
        print("\n3️⃣  Ejecutando benchmark...")
        input_shape = (1, state_dim)

        # PyTorch
        pt_result = benchmark.benchmark_pytorch(model, input_shape, model_name="PyTorch FP32")
        print(f"   PyTorch: {pt_result.avg_latency_ms:.3f}ms ({pt_result.throughput_ops:.0f} ops/s)")

        # ONNX FP32
        onnx_result = benchmark.benchmark_onnx(onnx_path, input_shape, model_name="ONNX FP32")
        print(f"   ONNX:    {onnx_result.avg_latency_ms:.3f}ms ({onnx_result.throughput_ops:.0f} ops/s)")

        # ONNX INT8
        if 'int8_path' in results:
            int8_result = benchmark.benchmark_onnx(results['int8_path'], input_shape, model_name="ONNX INT8")
            print(f"   INT8:    {int8_result.avg_latency_ms:.3f}ms ({int8_result.throughput_ops:.0f} ops/s)")

        speedup = pt_result.avg_latency_ms / onnx_result.avg_latency_ms
        print(f"\n   Speedup ONNX vs PyTorch: {speedup:.2f}x")

        results['benchmark'] = benchmark.summary()

    # 4. Validación de precisión
    if ORT_DISPONIBLE and onnx_path:
        print("\n4️⃣  Validando precisión...")
        precision_loss = benchmark.compare_precision(model, onnx_path, (1, state_dim))
        results['precision_loss'] = precision_loss
        print(f"   Error medio: {precision_loss:.8f}")

    print(f"\n✅ Pipeline completado")
    return results


# =============================================================================
# EJEMPLO
# =============================================================================

def ejemplo_onnx_export():
    """Demo de exportación y optimización"""
    print("\n" + "=" * 70)
    print("⚡ ATLAS Pro - Exportación ONNX y Cuantización")
    print("=" * 70)

    if not TORCH_DISPONIBLE:
        print("❌ PyTorch necesario")
        return

    # Crear modelo de ejemplo
    model = nn.Sequential(
        nn.Linear(26, 256), nn.ReLU(), nn.LayerNorm(256),
        nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 4)
    )

    results = optimize_model_pipeline(model, state_dim=26, model_name="atlas_demo")

    for key, val in results.items():
        if key != 'benchmark':
            print(f"  {key}: {val}")

    print("\n✅ Demo completada")


if __name__ == "__main__":
    ejemplo_onnx_export()
