"""
ATLAS Pro Scalability Stress Test Suite

Comprehensive scalability testing for ATLAS Pro multi-intersection system.
Evaluates performance across 4 to 200+ intersections with detailed metrics.

Author: ATLAS Pro Development Team
Version: 1.0.0
"""

import json
import logging
import argparse
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from datetime import datetime
from enum import Enum
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Edge device types for deployment"""
    RPI4 = "Raspberry Pi 4"
    JETSON_NANO = "NVIDIA Jetson Nano"
    JETSON_ORIN = "NVIDIA Jetson Orin"


@dataclass
class DeviceSpecification:
    """Edge device hardware specifications"""
    name: str
    device_type: DeviceType
    cpu_cores: int
    ram_gb: float
    compute_capability_tops: float  # Tera Operations Per Second
    max_memory_footprint_mb: float
    typical_power_watts: float

    @staticmethod
    def get_specs() -> Dict[str, 'DeviceSpecification']:
        """Get specifications for standard devices"""
        return {
            "rpi4": DeviceSpecification(
                name="Raspberry Pi 4 Model B",
                device_type=DeviceType.RPI4,
                cpu_cores=4,
                ram_gb=4.0,
                compute_capability_tops=0.015,  # Very limited
                max_memory_footprint_mb=3500.0,
                typical_power_watts=5.0
            ),
            "jetson_nano": DeviceSpecification(
                name="NVIDIA Jetson Nano",
                device_type=DeviceType.JETSON_NANO,
                cpu_cores=4,
                ram_gb=4.0,
                compute_capability_tops=0.5,  # 500 GFLOPS
                max_memory_footprint_mb=3500.0,
                typical_power_watts=10.0
            ),
            "jetson_orin": DeviceSpecification(
                name="NVIDIA Jetson Orin",
                device_type=DeviceType.JETSON_ORIN,
                cpu_cores=12,
                ram_gb=8.0,
                compute_capability_tops=275.0,  # 275 TFLOPS
                max_memory_footprint_mb=7500.0,
                typical_power_watts=60.0
            )
        }


@dataclass
class ScalabilityMetrics:
    """Metrics for a specific scale level"""
    num_intersections: int
    qmix_forward_latency_ms: float
    memory_usage_mb: float
    communication_overhead_mb_s: float
    green_wave_bandwidth_vehicles_s: float
    decision_throughput_decisions_s: float
    simulation_time_s: float
    stability: float  # 0-1, 1 is perfect stability
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ScalabilityReport:
    """Complete scalability test report"""
    test_id: str
    timestamp: str
    system_name: str
    rapid_mode: bool
    scales_tested: List[int]
    metrics: Dict[int, ScalabilityMetrics]
    bottleneck_analysis: Dict
    device_compatibility: Dict[str, Dict]
    summary: Dict


class QMIXSimulator:
    """Simulates QMIX network forward pass and compute requirements"""

    def __init__(self, num_agents: int):
        """
        Initialize QMIX simulator for multi-agent system.

        Args:
            num_agents: Number of traffic agents (intersections)
        """
        self.num_agents = num_agents
        self.state_dim = 13  # State features per agent
        self.action_dim = 4  # Phase actions
        self.hidden_dim = 128
        logger.debug(f"Initialized QMIX simulator for {num_agents} agents")

    def estimate_forward_latency(self) -> float:
        """
        Estimate QMIX forward pass latency (milliseconds).
        Scales with agent count and network complexity.

        Formula: base_latency + (agent_count * agent_overhead) + (agent_count^1.2 * network_overhead)
        """
        base_latency = 2.0  # ms
        agent_overhead = 0.5  # ms per agent
        network_scaling = 0.01  # ms per agent^1.2

        # Polynomial scaling with network complexity
        latency = (
            base_latency +
            (self.num_agents * agent_overhead) +
            (math.pow(self.num_agents, 1.2) * network_scaling)
        )

        # Add small random variance (±5%)
        variance = latency * 0.05 * (random.random() - 0.5)
        return max(latency + variance, 1.0)

    def estimate_memory_requirement(self) -> float:
        """
        Estimate memory requirement for QMIX network (MB).
        Includes model weights, activations, and buffers.
        """
        # Base model weights
        qmix_weight_mb = (
            (self.num_agents * self.state_dim * self.hidden_dim * 4) +  # Input layers
            (self.hidden_dim * self.hidden_dim * 4) +  # Hidden layers
            (self.hidden_dim * self.action_dim * 4)  # Output layers
        ) / (1024 * 1024)  # Convert bytes to MB

        # Experience replay buffer (10k transitions per agent)
        replay_buffer_mb = (
            self.num_agents * 10000 *
            (self.state_dim + self.action_dim + 8) *  # State, action, reward, done
            4  # Float32
        ) / (1024 * 1024)

        # Activation buffers and intermediate tensors
        activation_buffer_mb = (self.num_agents * self.hidden_dim * 100) / (1024 * 1024)

        total_mb = qmix_weight_mb + replay_buffer_mb + activation_buffer_mb
        return total_mb


class MultiIntersectionSystem:
    """Simulates ATLAS Pro multi-intersection system"""

    def __init__(self, num_intersections: int):
        """Initialize multi-intersection system"""
        self.num_intersections = num_intersections
        self.qmix_simulator = QMIXSimulator(num_intersections)
        self.communication_protocol = "5G/LTE"
        logger.debug(f"Initialized multi-intersection system with {num_intersections} intersections")

    def estimate_communication_overhead(self) -> float:
        """
        Estimate communication overhead in MB/s.
        Each agent communicates state and actions to neighbors and central server.
        """
        # Typical message size per intersection
        message_size_bytes = 512  # State + metadata

        # Each agent communicates with neighbors and central server
        connectivity_degree = min(8, self.num_intersections - 1)  # Max 8 neighbors

        # Messages per second (10 Hz update rate)
        update_frequency = 10.0

        # Calculate total bandwidth
        total_messages_per_second = (
            self.num_intersections *  # Agents sending
            (connectivity_degree + 1) *  # To neighbors + central
            update_frequency
        )

        overhead_mb_s = (total_messages_per_second * message_size_bytes) / (1024 * 1024)
        return overhead_mb_s

    def estimate_green_wave_bandwidth(self) -> float:
        """
        Estimate green wave progression bandwidth (vehicles/second).
        Maximum vehicles that can pass through coordinated intersections per second.
        """
        # Typical intersection green wave capacity
        vehicles_per_lane = 1.5  # vehicles/second during green
        typical_lanes_per_phase = 2
        phases_in_cycle = 4
        coordination_efficiency = 0.85  # How well coordinated

        single_intersection_capacity = (
            vehicles_per_lane *
            typical_lanes_per_phase *
            phases_in_cycle *
            coordination_efficiency
        )

        # Green wave bandwidth increases sublinearly with intersections
        network_effect = math.sqrt(self.num_intersections) * 0.8

        bandwidth = single_intersection_capacity * network_effect
        return bandwidth

    def estimate_decision_throughput(self) -> float:
        """
        Estimate decision throughput (decisions per second).
        """
        # Base decisions per intersection per second (10 Hz control loop)
        base_rate = 10.0

        # Coordination overhead (agents wait for neighbor info)
        coordination_penalty = 1.0 - (0.1 * math.log(self.num_intersections + 1))
        coordination_penalty = max(0.5, coordination_penalty)

        # Total throughput
        throughput = self.num_intersections * base_rate * coordination_penalty
        return throughput

    def estimate_stability(self) -> float:
        """
        Estimate system stability (0-1).
        Higher numbers indicate more stable operation.
        """
        # Base stability
        stability = 0.95

        # Scale degrades stability slightly (communication issues, deadlocks)
        scale_factor = math.exp(-0.01 * self.num_intersections)

        # Stability ranges from ~0.7 at 200 intersections to 0.95 at 4
        stability = 0.7 + (0.25 * scale_factor)

        # Add small random variation
        variation = 0.02 * (random.random() - 0.5)
        return max(0.1, min(1.0, stability + variation))

    def run_simulation(self, duration_seconds: int = 60) -> ScalabilityMetrics:
        """
        Run scalability simulation for specified duration.

        Args:
            duration_seconds: Simulation duration in seconds

        Returns:
            ScalabilityMetrics with collected data
        """
        logger.info(f"Running scalability simulation for {self.num_intersections} intersections ({duration_seconds}s)")

        # Collect metrics
        latencies = []

        # Simulate control loop at 10 Hz
        steps = duration_seconds * 10
        for _ in range(steps):
            latency = self.qmix_simulator.estimate_forward_latency()
            latencies.append(latency)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        memory_usage = self.qmix_simulator.estimate_memory_requirement()
        comm_overhead = self.estimate_communication_overhead()
        green_wave = self.estimate_green_wave_bandwidth()
        throughput = self.estimate_decision_throughput()
        stability = self.estimate_stability()

        metrics = ScalabilityMetrics(
            num_intersections=self.num_intersections,
            qmix_forward_latency_ms=avg_latency,
            memory_usage_mb=memory_usage,
            communication_overhead_mb_s=comm_overhead,
            green_wave_bandwidth_vehicles_s=green_wave,
            decision_throughput_decisions_s=throughput,
            simulation_time_s=duration_seconds,
            stability=stability
        )

        logger.info(f"Simulation complete: {self.num_intersections} intersections, "
                   f"latency={avg_latency:.2f}ms, memory={memory_usage:.1f}MB, "
                   f"stability={stability:.2f}")

        return metrics


class ScalabilityTester:
    """Main scalability testing orchestrator"""

    def __init__(self, rapid_mode: bool = False):
        """
        Initialize scalability tester.

        Args:
            rapid_mode: If True, use shorter simulation durations
        """
        self.rapid_mode = rapid_mode
        self.simulation_duration = 5 if rapid_mode else 60  # seconds
        self.scales_to_test = [4, 9, 16, 25, 50, 100, 200]
        self.metrics: Dict[int, ScalabilityMetrics] = {}
        self.device_specs = DeviceSpecification.get_specs()
        logger.info(f"Initialized scalability tester (rapid_mode={rapid_mode})")

    def test_scale(self, num_intersections: int) -> ScalabilityMetrics:
        """Test a specific scale level"""
        system = MultiIntersectionSystem(num_intersections)
        metrics = system.run_simulation(self.simulation_duration)
        self.metrics[num_intersections] = metrics
        return metrics

    def run_all_scales(self) -> Dict[int, ScalabilityMetrics]:
        """Run tests for all scale levels"""
        logger.info(f"Running scalability tests for {len(self.scales_to_test)} scale levels")

        for scale in self.scales_to_test:
            self.test_scale(scale)

        return self.metrics

    def check_device_compatibility(self, metrics: ScalabilityMetrics) -> Dict[str, Tuple[bool, str]]:
        """
        Check if a scale level can run on available edge devices.

        Args:
            metrics: Scalability metrics for the scale level

        Returns:
            Dict mapping device names to (compatible, reason) tuples
        """
        compatibility = {}

        for device_key, device_spec in self.device_specs.items():
            memory_ok = metrics.memory_usage_mb <= device_spec.max_memory_footprint_mb
            latency_ok = metrics.qmix_forward_latency_ms <= 50  # Target: <50ms

            if memory_ok and latency_ok:
                compatible = True
                reason = f"Memory: {metrics.memory_usage_mb:.0f}MB <= {device_spec.max_memory_footprint_mb:.0f}MB"
            elif not memory_ok:
                compatible = False
                reason = f"Insufficient memory: {metrics.memory_usage_mb:.0f}MB > {device_spec.max_memory_footprint_mb:.0f}MB"
            else:
                compatible = False
                reason = f"Latency too high: {metrics.qmix_forward_latency_ms:.1f}ms > 50ms"

            compatibility[device_spec.name] = (compatible, reason)

        return compatibility

    def analyze_bottlenecks(self) -> Dict:
        """Analyze system bottlenecks across scale levels"""
        logger.info("Analyzing bottlenecks")

        bottlenecks = {
            "latency_critical": [],
            "memory_critical": [],
            "communication_critical": [],
            "stability_concerns": []
        }

        for scale, metrics in sorted(self.metrics.items()):
            # Latency bottleneck (>50ms is concerning for real-time control)
            if metrics.qmix_forward_latency_ms > 50:
                bottlenecks["latency_critical"].append({
                    "scale": scale,
                    "latency_ms": metrics.qmix_forward_latency_ms,
                    "threshold_ms": 50
                })

            # Memory bottleneck (>7500MB for largest device)
            if metrics.memory_usage_mb > 7500:
                bottlenecks["memory_critical"].append({
                    "scale": scale,
                    "memory_mb": metrics.memory_usage_mb,
                    "threshold_mb": 7500
                })

            # Communication overhead (>100MB/s is excessive)
            if metrics.communication_overhead_mb_s > 100:
                bottlenecks["communication_critical"].append({
                    "scale": scale,
                    "overhead_mb_s": metrics.communication_overhead_mb_s,
                    "threshold_mb_s": 100
                })

            # Stability concerns (<0.75)
            if metrics.stability < 0.75:
                bottlenecks["stability_concerns"].append({
                    "scale": scale,
                    "stability": metrics.stability,
                    "threshold": 0.75
                })

        return bottlenecks

    def generate_report(self) -> ScalabilityReport:
        """Generate comprehensive scalability report"""
        logger.info("Generating scalability report")

        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks()

        # Check device compatibility for all scales
        device_compat = {}
        for scale, metrics in self.metrics.items():
            device_compat[scale] = self.check_device_compatibility(metrics)

        # Generate summary
        max_scale = max(self.metrics.keys())
        max_metrics = self.metrics[max_scale]

        summary = {
            "test_duration_seconds": self.simulation_duration,
            "rapid_mode": self.rapid_mode,
            "scales_tested": sorted(self.scales_to_test),
            "max_scale_tested": max_scale,
            "latency_at_max": f"{max_metrics.qmix_forward_latency_ms:.2f}ms",
            "memory_at_max": f"{max_metrics.memory_usage_mb:.1f}MB",
            "stability_at_max": f"{max_metrics.stability:.2f}",
            "recommended_max_scale": self._recommend_max_scale(),
            "bottleneck_summary": f"{sum(len(v) for v in bottlenecks.values())} critical issues found"
        }

        report = ScalabilityReport(
            test_id=f"scalability-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            system_name="ATLAS Pro Traffic Management System",
            rapid_mode=self.rapid_mode,
            scales_tested=sorted(self.scales_to_test),
            metrics=self.metrics,
            bottleneck_analysis=bottlenecks,
            device_compatibility=device_compat,
            summary=summary
        )

        return report

    def _recommend_max_scale(self) -> int:
        """Determine recommended maximum scale based on metrics"""
        for scale in sorted(self.metrics.keys(), reverse=True):
            metrics = self.metrics[scale]
            # Recommend scale if latency <50ms and stability >0.75
            if metrics.qmix_forward_latency_ms <= 50 and metrics.stability >= 0.75:
                return scale
        return self.scales_to_test[0]

    def print_results(self, report: ScalabilityReport) -> None:
        """Print formatted scalability test results"""
        print("\n" + "="*90)
        print("ATLAS PRO - SCALABILITY STRESS TEST REPORT")
        print("="*90)
        print(f"Test ID: {report.test_id}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Rapid Mode: {report.rapid_mode}")
        print("-"*90)

        # Metrics table
        print("\nSCALABILITY METRICS BY SCALE LEVEL")
        print("-"*90)
        print(f"{'Scale':<8} {'Latency':<12} {'Memory':<12} {'Comm OH':<12} {'GW BW':<12} {'Throughput':<12} {'Stability':<10}")
        print(f"{'(#)':<8} {'(ms)':<12} {'(MB)':<12} {'(MB/s)':<12} {'(v/s)':<12} {'(dec/s)':<12} {'(0-1)':<10}")
        print("-"*90)

        for scale in sorted(report.metrics.keys()):
            m = report.metrics[scale]
            print(f"{m.num_intersections:<8} {m.qmix_forward_latency_ms:<12.2f} "
                  f"{m.memory_usage_mb:<12.1f} {m.communication_overhead_mb_s:<12.2f} "
                  f"{m.green_wave_bandwidth_vehicles_s:<12.1f} {m.decision_throughput_decisions_s:<12.1f} "
                  f"{m.stability:<10.2f}")

        # Device compatibility
        print("\n" + "-"*90)
        print("EDGE DEVICE COMPATIBILITY MATRIX")
        print("-"*90)

        devices = list(self.device_specs.values())
        device_names = [d.name for d in devices]

        print(f"{'Scale':<8}", end="")
        for device_name in device_names:
            print(f"{device_name:<30}", end="")
        print()
        print("-"*90)

        for scale in sorted(report.device_compatibility.keys()):
            print(f"{scale:<8}", end="")
            for device_name in device_names:
                compat, reason = report.device_compatibility[scale][device_name]
                status = "✓ OK" if compat else "✗ FAIL"
                print(f"{status:<30}", end="")
            print()

        # Bottleneck analysis
        print("\n" + "-"*90)
        print("BOTTLENECK ANALYSIS")
        print("-"*90)

        bottlenecks = report.bottleneck_analysis
        total_issues = sum(len(v) for v in bottlenecks.values())

        if total_issues == 0:
            print("✓ No critical bottlenecks detected")
        else:
            if bottlenecks["latency_critical"]:
                print(f"\n⚠ Latency Issues ({len(bottlenecks['latency_critical'])}):")
                for issue in bottlenecks["latency_critical"]:
                    print(f"  Scale {issue['scale']}: {issue['latency_ms']:.1f}ms (threshold: {issue['threshold_ms']}ms)")

            if bottlenecks["memory_critical"]:
                print(f"\n⚠ Memory Issues ({len(bottlenecks['memory_critical'])}):")
                for issue in bottlenecks["memory_critical"]:
                    print(f"  Scale {issue['scale']}: {issue['memory_mb']:.0f}MB (threshold: {issue['threshold_mb']}MB)")

            if bottlenecks["communication_critical"]:
                print(f"\n⚠ Communication Issues ({len(bottlenecks['communication_critical'])}):")
                for issue in bottlenecks["communication_critical"]:
                    print(f"  Scale {issue['scale']}: {issue['overhead_mb_s']:.1f}MB/s (threshold: {issue['threshold_mb_s']}MB/s)")

            if bottlenecks["stability_concerns"]:
                print(f"\n⚠ Stability Concerns ({len(bottlenecks['stability_concerns'])}):")
                for issue in bottlenecks["stability_concerns"]:
                    print(f"  Scale {issue['scale']}: {issue['stability']:.2f} (threshold: {issue['threshold']})")

        # Summary
        print("\n" + "-"*90)
        print("SUMMARY")
        print("-"*90)
        for key, value in report.summary.items():
            print(f"{key:<30}: {value}")

        print("\n" + "="*90 + "\n")

    def save_report(self, report: ScalabilityReport, filepath: str = "scalability_report.json") -> str:
        """Save report to JSON file"""
        try:
            # Convert dataclasses to dicts for JSON serialization
            report_dict = {
                "test_id": report.test_id,
                "timestamp": report.timestamp,
                "system_name": report.system_name,
                "rapid_mode": report.rapid_mode,
                "scales_tested": report.scales_tested,
                "metrics": {k: asdict(v) for k, v in report.metrics.items()},
                "bottleneck_analysis": report.bottleneck_analysis,
                "device_compatibility": {
                    str(k): {device: (compat, reason) for device, (compat, reason) in v.items()}
                    for k, v in report.device_compatibility.items()
                },
                "summary": report.summary
            }

            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)

            logger.info(f"Report saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise


def main():
    """Main entry point for scalability testing"""
    parser = argparse.ArgumentParser(
        description="ATLAS Pro Scalability Stress Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Run in rapid mode with shorter simulation durations (5s instead of 60s)"
    )

    parser.add_argument(
        "--output",
        default="scalability_report.json",
        help="Output JSON file for results (default: scalability_report.json)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to JSON file"
    )

    args = parser.parse_args()

    print("ATLAS Pro Scalability Stress Test Suite")
    print(f"Rapid Mode: {args.rapido}")
    print("Starting comprehensive scalability tests...\n")

    try:
        # Create tester and run tests
        tester = ScalabilityTester(rapid_mode=args.rapido)
        tester.run_all_scales()

        # Generate report
        report = tester.generate_report()

        # Save report
        if not args.no_save:
            tester.save_report(report, args.output)

        # Print results
        tester.print_results(report)

        return 0

    except Exception as e:
        logger.error(f"Scalability test failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
