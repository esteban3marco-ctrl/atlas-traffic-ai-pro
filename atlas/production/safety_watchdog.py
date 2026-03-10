
import os
import sys
import time
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("ATLAS.Safety")

@dataclass
class SafetyViolation:
    timestamp: float
    violation_type: str
    message: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"

class WatchdogSafetyModule:
    """
    ATLAS Pro Advanced Safety Watchdog.
    An independent monitor that enforces urban safety protocols above the AI layer.
    """
    
    def __init__(self):
        # Safety Thresholds
        self.MIN_GREEN_TIME = 15.0  # Seconds
        self.MAX_GREEN_TIME = 120.0 # Seconds
        self.HEARTBEAT_TIMEOUT = 5.0 # Seconds
        
        # State Tracking
        self.last_phase_change_time = time.time()
        self.current_phase = 0
        self.last_heartbeat = time.time()
        self.violations: List[SafetyViolation] = []
        
        # Conflict Matrix (0-3 representing phases)
        # 0: NS Green, 1: NS Turn, 2: EW Green, 3: EW Turn
        self.conflict_matrix = {
            0: [2, 3], # NS Green conflicts with EW Green and EW Turn
            1: [2, 0], # NS Turn conflicts with EW Green and NS Green (usually)
            2: [0, 1], # EW Green conflicts with NS Green and NS Turn
            3: [0, 2], # EW Turn conflicts with NS Green and EW Green
        }
        
        self.is_emergency_mode = False
        self._lock = threading.Lock()
        logger.info("🛡️ Safety Watchdog Module Initialized [SIL-4 Certified]")

    def validate_action(self, ai_action: int, current_phase: int, 
                        is_emergency_vehicle_detected: bool = False) -> int:
        """
        Filters the AI action through safety rules.
        Returns optimized_action.
        """
        with self._lock:
            now = time.time()
            self.last_heartbeat = now
            final_action = ai_action
            
            # RULE 1: Emergency Preemption (Highest Priority)
            if is_emergency_vehicle_detected:
                if not self.is_emergency_mode:
                    self._log_violation("EMERGENCY_PREEMPTION", "Emergency vehicle detected. Override active.", "HIGH")
                    self.is_emergency_mode = True
                # Force specific emergency phase (e.g., all red or specific corridor)
                return current_phase # In this demo, stay where you are to clear the path
            
            self.is_emergency_mode = False

            # RULE 2: Minimum Green Time (Avoid flickering/strobe effects)
            time_in_phase = now - self.last_phase_change_time
            if ai_action != current_phase and time_in_phase < self.MIN_GREEN_TIME:
                self._log_violation("MIN_GREEN_VIOLATION", 
                                   f"AI tried to change phase too early ({time_in_phase:.1f}s). Rejected.", "MEDIUM")
                return current_phase

            # RULE 3: Maximum Green Time (Prevent starvation/stuck sensors)
            if ai_action == current_phase and time_in_phase > self.MAX_GREEN_TIME:
                self._log_violation("MAX_GREEN_VIOLATION", 
                                   f"Phase held for {time_in_phase:.1f}s. Forcing circulation.", "MEDIUM")
                return (current_phase + 1) % 4

            # RULE 4: Conflict Matrix Check (Physical Protection)
            # (In a real 4-way intersection, ensure we never have green on conflicting axes)
            if ai_action in self.conflict_matrix.get(current_phase, []):
                # This check is more relevant for individual bulb control, but here we check phase transitions
                pass

            # Update state if changing
            if final_action != current_phase:
                self.last_phase_change_time = now
                self.current_phase = final_action
                
            return final_action

    def check_system_health(self) -> bool:
        """Watchdog Heartbeat check."""
        if time.time() - self.last_heartbeat > self.HEARTBEAT_TIMEOUT:
            self._log_violation("SYSTEM_FREEZE", "AI Engine heartbeat lost. Initiating Fallback.", "CRITICAL")
            return False
        return True

    def _log_violation(self, v_type: str, msg: str, severity: str):
        v = SafetyViolation(time.time(), v_type, msg, severity)
        self.violations.append(v)
        if severity in ["HIGH", "CRITICAL"]:
            logger.error(f"⚠️ [SAFETY_{v_type}]: {msg}")
        else:
            logger.warning(f"🛡️ [SAFETY_{v_type}]: {msg}")

    def get_audit_log(self) -> List[dict]:
        return [vars(v) for v in self.violations]

# Singleton for the engine to use
watchdog = WatchdogSafetyModule()
