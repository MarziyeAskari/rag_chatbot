import datetime
import json
from pathlib import Path
from typing import Optional, Dict


import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging




# Prometheus metrics
query_counter = Counter('rag_queries_total', 'Total number of queries', ['status'])
query_duration = Histogram('rag_query_duration_seconds', 'Query duration in seconds')
vector_store_size = Gauge('rag_vector_store_size', 'Number of documents in vector store')
system_cpu = Gauge('rag_system_cpu_percent', 'System CPU usage percentage')
system_memory = Gauge('rag_system_memory_percent', 'System memory usage percentage')
error_counter = Counter('rag_errors_total', 'Total number of errors', ['error_type'])

logger = logging.getLogger(__name__)
class MonitoringSystem:
    def __init__(self,
                 metric_port: int = 9090,
                 alert_threshold: Optional[Dict] = None,
                 log_file:str = "./logs/monitoring.log",):
        self.metric_port = metric_port
        self.alert_threshold = alert_threshold or {
            "cpu_percent": 80.0,
            "memory_percent": 80.0,
            "error_rate": 0.1,
            "response_time": 5.0,
        }
        self.log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            start_http_server(self.metric_port)
            logger.info(f"Starting monitoring system at port {self.metric_port}")
        except Exception as e:
            logger.error(f"Failed to start monitoring system at port {self.metric_port}: {e}")
        self.query_history : list[Dict]=[]
        self.error_history : list[Dict]=[]
        self.max_history =1000
        
    def record_query(self,
                     question: str,
                     answer: str,
                     duration: float,
                     success: bool= True,
                     error: Optional[str] = None, query_duration=None):
        status = "success" if success else "error"
        query_counter.labels(status=status).inc()
        query_duration.observe(duration)

        if not success and error:
            error_counter.labels(error_type=type(error).__name__).inc()
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question[:100],
            "answer_length": len(answer),
            "duration": duration,
            "success": success,
            "error": error,
        }
        self.query_history.append(record)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)

        self._check_alerts()

    def _check_alerts(self):
        summary = self.get_metrics_summary(hours = 1)
        system_metrics = self.update_system_metrics()
        alerts = []
        if system_metrics["cpu_percent"] > self.alert_threshold["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "value": system_metrics["cpu_percent"],
                "threshold": self.alert_threshold["cpu_percent"],
                "message": f"High CPU usage:  {system_metrics['cpu_percent']:.1f}%",
            })
        if system_metrics["memory_percent"] > self.alert_threshold["memory_percent"]:
            alerts.append({
                "type": "high_memory",
                "value":system_metrics["memory_percent"],
                "threshold": self.alert_threshold["memory_percent"],
                "message":f"High memory usage: {system_metrics['memory_percent']:.1f}%"
            })
        if summary["total_queries"] > 10:
            error_rat = summary ["error_count"] / summary["total_queries"]
            if error_rat > self.alert_threshold["error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "value": error_rat,
                    "threshold": self.alert_threshold["error_rate"],
                    "message": f"High error rate: {error_rat:.1f}%"
                })

        if summary["avg_duration"] > self.alert_threshold["response_time"]:
            alerts.append({
                "type": "slow_response",
                "value": summary["avg_duration"],
                "threshold": self.alert_threshold["response_time"],
                "message": f"Slow response time: {summary['avg_duration']:2.f}s"
            })

        for alert in alerts:
            self._handle_alert(alert)

    def get_metrics_summary(self, hours):
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        recent_queries = [
           q for q in self.query_history
            if datetime.datetime.fromisoformat(q["timestamp"]) > cutoff_time
        ]
        if not recent_queries:
            return {
                "total_queries": 0,
                "success_rate": False,
                "avg_duration": 0.0,
                "error_count": 0,
            }

        total = len(recent_queries)
        successful = sum(1 for q in recent_queries if q["success"])
        duration = [q["duration"] for q in recent_queries]
        return {
            "total_queries": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_duration": sum(duration) / len(duration) if duration else 0.0,
            "error_count": total - successful,
            "max_duration": max(duration),
            "min_duration": min(duration),
        }

    def update_system_metrics(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        system_cpu.set(cpu_percent)
        system_memory.set(memory_percent)
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
        }
    def update_vector_store_size(self, size:int):
        vector_store_size.set(size)

    def _handle_alert(self, alert):
        alert_message =f"[ALERT] {alert['message']} (threshold: {alert['threshold']}) "
        logger.warning(alert_message)
        with open (self.log_file, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {alert_message}\n")

    def get_health_status(self)-> Dict:
        summary = self.get_metrics_summary(hours = 1)
        system_metrics = self.update_system_metrics()
        health_score = 100
        if system_metrics["cpu_percent"] > 80:
            health_score -= 20
        elif system_metrics["memory_percent"] > 60:
            health_score -= 10

        if system_metrics["memory_percent"] > 80:
            health_score -= 20
        elif system_metrics["memory_percent"] > 60:
            health_score -= 10

        if summary["total_queries"] > 0:
            error_rate = summary["error_count"] / summary["total_queries"]
            if error_rate > 0.1:
                health_score -= 30
            elif error_rate > 0.05:
                health_score -= 15

        if summary["avg_duration"] > 5.0:
            health_score -= 20
        elif summary["avg_duration"] > 3.0:
            health_score -= 10

        health_status = "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy"

        return {
            "status": health_status,
            "score": max(0, health_score),
            "metrics":{
                **summary,
                **system_metrics,
            },
            "timestamp": datetime.datetime.now().isoformat()
        }

    def export_metrics(self, file_path:str):
        data ={
            "query_history": self.query_history[-100:],
            "error_history": self.error_history[-50:],
            "summary": self.get_metrics_summary(hours = 1),
            "health": self.get_health_status(),
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Metrics exported to {file_path}")

_monitoring_instance = Optional[MonitoringSystem]=None


def get_monitoring()->MonitoringSystem:
    global _monitoring_instance
    if _monitoring_instance is None:
        _monitoring_instance = MonitoringSystem()
    return _monitoring_instance