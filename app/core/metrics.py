"""轻量运行指标：观察并发峰值、延迟与失败率。

仅统计进程内数据，重启归零。用于判断真实并发是否接近容量上限，
为「是否需要扩容 / 换推理层」提供数据依据。
"""

import threading
from dataclasses import dataclass, field


@dataclass
class Metrics:
    in_flight: int = 0            # 当前处理中的对话请求数
    peak_in_flight: int = 0       # 历史并发峰值（核心指标）
    total_requests: int = 0
    failed_requests: int = 0
    busy_rejections: int = 0      # 因并发已满被拒绝的请求
    staged_requests: int = 0      # 真正记录了分阶段耗时的请求数（用于算平均）

    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    total_generation_ms: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def enter(self) -> int:
        """登记请求进入，返回进入时刻的并发数（供日志使用）。"""
        with self._lock:
            self.in_flight += 1
            self.total_requests += 1
            current = self.in_flight
            if current > self.peak_in_flight:
                self.peak_in_flight = current
            return current

    def leave(self, latency_ms: float, ok: bool = True) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            self.total_latency_ms += latency_ms
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
            if not ok:
                self.failed_requests += 1

    def record_busy(self) -> None:
        with self._lock:
            self.busy_rejections += 1

    def record_stage(self, retrieval_ms: float, generation_ms: float) -> None:
        with self._lock:
            self.staged_requests += 1
            self.total_retrieval_ms += retrieval_ms
            self.total_generation_ms += generation_ms

    def snapshot(self) -> dict:
        with self._lock:
            total = self.total_requests
            staged = self.staged_requests
            return {
                "in_flight": self.in_flight,
                "peak_in_flight": self.peak_in_flight,
                "total_requests": total,
                "failed_requests": self.failed_requests,
                "busy_rejections": self.busy_rejections,
                "avg_latency_ms": round(self.total_latency_ms / total, 1) if total else 0,
                "max_latency_ms": round(self.max_latency_ms, 1),
                # 分母使用 staged_requests：只有成功走完流程的请求才有分阶段耗时
                "avg_retrieval_ms": round(self.total_retrieval_ms / staged, 1) if staged else 0,
                "avg_generation_ms": round(self.total_generation_ms / staged, 1) if staged else 0,
            }


# 进程级单例：所有路由与健康检查共享
metrics = Metrics()
