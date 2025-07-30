# app/utils/performance_monitor.py - Real-time performance monitoring
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTarget:
    stt_max: float = 300  # ms
    llm_max: float = 500  # ms
    tts_max: float = 400  # ms
    total_max: float = 1200  # ms

class PerformanceMonitor:
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.targets = PerformanceTarget()
        
        # Performance metrics storage
        self.metrics = {
            "response_times": deque(maxlen=max_history_size),
            "stt_times": deque(maxlen=max_history_size),
            "llm_times": deque(maxlen=max_history_size),
            "tts_times": deque(maxlen=max_history_size),
            "success_count": 0,
            "failure_count": 0,
            "total_requests": 0
        }
        
        # Real-time averages
        self.current_averages = {
            "response_time": 0.0,
            "stt_time": 0.0,
            "llm_time": 0.0,
            "tts_time": 0.0,
            "success_rate": 100.0
        }
        
        # Target achievement tracking
        self.target_achievement = {
            "stt": 0.0,
            "llm": 0.0,
            "tts": 0.0,
            "total": 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "response_time_critical": 2000,  # ms
            "error_rate_critical": 10,      # %
            "target_miss_critical": 50      # %
        }
        
        # Performance windows for analysis
        self.performance_windows = {
            "1m": {"size": 60, "data": deque(maxlen=60)},
            "5m": {"size": 300, "data": deque(maxlen=300)},
            "15m": {"size": 900, "data": deque(maxlen=900)}
        }
        
        # Active alerts
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        logger.info("Ultra-fast performance monitor initialized")

    def record_voice_processing(self, request_id: str, processing_times: Dict[str, float], 
                               success: bool = True, metadata: Optional[Dict] = None):
        """Record voice processing performance metrics"""
        try:
            timestamp = time.time()
            total_time = processing_times.get("total", 0) * 1000  # Convert to ms
            stt_time = processing_times.get("stt", 0) * 1000
            llm_time = processing_times.get("llm", 0) * 1000
            tts_time = processing_times.get("tts", 0) * 1000
            
            # Store raw metrics
            self.metrics["response_times"].append(total_time)
            self.metrics["stt_times"].append(stt_time)
            self.metrics["llm_times"].append(llm_time)
            self.metrics["tts_times"].append(tts_time)
            
            # Update counters
            self.metrics["total_requests"] += 1
            if success:
                self.metrics["success_count"] += 1
            else:
                self.metrics["failure_count"] += 1
            
            # Update real-time averages
            self._update_averages()
            
            # Update target achievement rates
            self._update_target_achievement(stt_time, llm_time, tts_time, total_time)
            
            # Add to performance windows
            self._update_performance_windows(timestamp, total_time, success, metadata)
            
            # Check for performance alerts
            self._check_performance_alerts(total_time, stt_time, llm_time, tts_time, success)
            
            # Log performance warnings
            if total_time > self.targets.total_max:
                logger.warning(f"Response time exceeded target: {total_time:.0f}ms > {self.targets.total_max}ms")
            
            logger.debug(f"Performance recorded: {request_id} - {total_time:.0f}ms total")
            
        except Exception as e:
            logger.error(f"Error recording performance metrics: {str(e)}")

    def _update_averages(self):
        """Update rolling averages efficiently"""
        try:
            if self.metrics["response_times"]:
                self.current_averages["response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            
            if self.metrics["stt_times"]:
                self.current_averages["stt_time"] = sum(self.metrics["stt_times"]) / len(self.metrics["stt_times"])
            
            if self.metrics["llm_times"]:
                self.current_averages["llm_time"] = sum(self.metrics["llm_times"]) / len(self.metrics["llm_times"])
            
            if self.metrics["tts_times"]:
                self.current_averages["tts_time"] = sum(self.metrics["tts_times"]) / len(self.metrics["tts_times"])
            
            if self.metrics["total_requests"] > 0:
                self.current_averages["success_rate"] = (self.metrics["success_count"] / self.metrics["total_requests"]) * 100
                
        except Exception as e:
            logger.error(f"Error updating averages: {str(e)}")

    def _update_target_achievement(self, stt_time: float, llm_time: float, tts_time: float, total_time: float):
        """Update target achievement rates"""
        try:
            # Calculate if targets were met
            targets_met = {
                "stt": stt_time <= self.targets.stt_max,
                "llm": llm_time <= self.targets.llm_max,
                "tts": tts_time <= self.targets.tts_max,
                "total": total_time <= self.targets.total_max
            }
            
            # Update rolling achievement rates (exponential moving average)
            alpha = 0.1  # Smoothing factor
            for component, met in targets_met.items():
                current_rate = self.target_achievement[component]
                self.target_achievement[component] = (1 - alpha) * current_rate + alpha * (1.0 if met else 0.0)
                
        except Exception as e:
            logger.error(f"Error updating target achievement: {str(e)}")

    def _update_performance_windows(self, timestamp: float, total_time: float, success: bool, metadata: Optional[Dict]):
        """Update sliding window performance data"""
        try:
            data_point = {
                "timestamp": timestamp,
                "response_time": total_time,
                "success": success,
                "metadata": metadata or {}
            }
            
            for window_name, window_data in self.performance_windows.items():
                window_data["data"].append(data_point)
                
        except Exception as e:
            logger.error(f"Error updating performance windows: {str(e)}")

    def _check_performance_alerts(self, total_time: float, stt_time: float, llm_time: float, tts_time: float, success: bool):
        """Check for performance alerts and trigger notifications"""
        try:
            current_time = time.time()
            
            # Critical response time alert
            if total_time > self.alert_thresholds["response_time_critical"]:
                self._trigger_alert("critical_response_time", {
                    "response_time": total_time,
                    "threshold": self.alert_thresholds["response_time_critical"],
                    "severity": "critical"
                })
            
            # Component performance alerts
            component_times = {
                "stt": (stt_time, self.targets.stt_max),
                "llm": (llm_time, self.targets.llm_max),
                "tts": (tts_time, self.targets.tts_max)
            }
            
            for component, (actual_time, target_time) in component_times.items():
                if actual_time > target_time * 1.5:  # 50% over target
                    self._trigger_alert(f"{component}_performance_degraded", {
                        "component": component,
                        "actual_time": actual_time,
                        "target_time": target_time,
                        "severity": "warning"
                    })
            
            # Target achievement rate alerts
            for component, achievement_rate in self.target_achievement.items():
                if achievement_rate < 0.5:  # Less than 50% achievement
                    self._trigger_alert(f"{component}_target_missed", {
                        "component": component,
                        "achievement_rate": achievement_rate * 100,
                        "severity": "warning"
                    })
            
            # Error rate alert
            if self.metrics["total_requests"] >= 10:  # Only check after minimum requests
                error_rate = (self.metrics["failure_count"] / self.metrics["total_requests"]) * 100
                if error_rate > self.alert_thresholds["error_rate_critical"]:
                    self._trigger_alert("high_error_rate", {
                        "error_rate": error_rate,
                        "threshold": self.alert_thresholds["error_rate_critical"],
                        "severity": "critical"
                    })
                    
        except Exception as e:
            logger.error(f"Error checking performance alerts: {str(e)}")

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger performance alert"""
        try:
            current_time = time.time()
            alert_key = f"{alert_type}_{details.get('component', 'system')}"
            
            # Prevent alert spam (5 minute cooldown)
            if alert_key in self.active_alerts:
                last_alert_time = self.active_alerts[alert_key]["timestamp"]
                if current_time - last_alert_time < 300:  # 5 minutes
                    return
            
            alert = {
                "type": alert_type,
                "details": details,
                "timestamp": current_time,
                "severity": details.get("severity", "info")
            }
            
            # Store active alert
            self.active_alerts[alert_key] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Log alert
            severity = details.get("severity", "info")
            log_level = logging.ERROR if severity == "critical" else logging.WARNING
            logger.log(log_level, f"Performance Alert [{alert_type}]: {details}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")

    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            return {
                "averages": self.current_averages.copy(),
                "target_achievement": {k: v * 100 for k, v in self.target_achievement.items()},
                "total_requests": self.metrics["total_requests"],
                "success_count": self.metrics["success_count"],
                "failure_count": self.metrics["failure_count"],
                "active_alerts": len(self.active_alerts),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting current performance: {str(e)}")
            return {}

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            current_time = time.time()

        # Compute current performance averages
            averages = self.current_averages.copy()
        
        # Convert achievement rates to percentages
            target_rates = {k: v * 100 for k, v in self.target_achievement.items()}

        # Build sliding-window summaries
            window_stats = {}
            for name, window in self.performance_windows.items():
                data = window["data"]
                if data:
                    latencies = [d["response_time"] for d in data]
                    window_stats[name] = {
                        "count": len(latencies),
                        "p50": np.percentile(latencies, 50),
                        "p95": np.percentile(latencies, 95),
                        "p99": np.percentile(latencies, 99),
                    }
                else:
                    window_stats[name] = {"count": 0, "p50": 0, "p95": 0, "p99": 0}

            return {
                "timestamp": current_time,
                "averages": averages,
                "target_achievement_percent": target_rates,
                "total_requests": self.metrics["total_requests"],
                "success_rate": self.current_averages["success_rate"],
                "windowed_statistics": window_stats,
                "active_alerts": len(self.active_alerts),
                "alert_history_last_5": list(self.alert_history)[-5:]
            }

        except Exception as e:
            logger.error(f"Error generating performance report: {e}", exc_info=True)
        # Return an empty or minimal safe report on failure
            return {
                "timestamp": time.time(),
                "error": str(e),
                "averages": {},
                "target_achievement_percent": {},
                "total_requests": 0,
                "success_rate": 0.0,
                "windowed_statistics": {},
                "active_alerts": 0,
                "alert_history_last_5": []
            }

