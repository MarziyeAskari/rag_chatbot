# mlops/monitoring.py
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

query_counter = Counter("rag_queries_total", "Total queries", ["status"])
query_duration_hist = Histogram("rag_query_duration_seconds", "Query duration seconds")
vector_store_size_g = Gauge("rag_vector_store_size", "Documents in vector store")
system_cpu_g = Gauge("rag_system_cpu_percent", "CPU percent")
system_mem_g = Gauge("rag_system_memory_percent", "Memory percent")
error_counter = Counter("rag_errors_total", "Errors total", ["error_type"])


class MonitoringSystem:
    def __init__(self, metrics_port: int = 9090, log_file: str = "./logs/monitoring.log", start_server: bool = True):
        self.metrics_port = metrics_port
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.query_history: List[Dict] = []
        self.max_history = 1000

        if start_server:
            try:
                start_http_server(self.metrics_port)
                logger.info(f"Prometheus metrics server started on :{self.metrics_port}")
            except Exception as e:
                logger.warning(f"Metrics server not started (maybe already running): {e}")

    def record_query(self, question: str, answer: str, duration: float, success: bool = True, error: Optional[Exception] = None):
        status = "success" if success else "error"
        query_counter.labels(status=status).inc()
        query_duration_hist.observe(duration)

        if not success and error:
            error_counter.labels(error_type=type(error).__name__).inc()

        self.query_history.append({
            "ts": datetime.datetime.utcnow().isoformat(),
            "q_preview": question[:120],
            "answer_len": len(answer or ""),
            "duration": float(duration),
            "success": bool(success),
            "error": str(error) if error else None,
        })
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)

    def update_system_metrics(self) -> Dict[str, float]:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        system_cpu_g.set(cpu)
        system_mem_g.set(mem)
        return {"cpu_percent": cpu, "memory_percent": mem}

    def update_vector_store_size(self, size: int):
        vector_store_size_g.set(size)

    def health(self) -> Dict:
        sysm = self.update_system_metrics()
        # simple scoring
        score = 100
        if sysm["cpu_percent"] > 80:
            score -= 20
        if sysm["memory_percent"] > 80:
            score -= 20

        status = "healthy" if score >= 80 else "degraded" if score >= 50 else "unhealthy"
        return {"status": status, "score": max(0, score), "metrics": sysm, "ts": datetime.datetime.utcnow().isoformat()}

    def export(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"health": self.health(), "recent_queries": self.query_history[-100:]}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


_monitor: Optional[MonitoringSystem] = None


def get_monitoring() -> MonitoringSystem:
    global _monitor
    if _monitor is None:
        _monitor = MonitoringSystem()
    return _monitor
