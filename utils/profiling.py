import time
import functools
import tracemalloc
from typing import Callable, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TIME PROFILING
# ============================================================================

def profile_time(func: Callable) -> Callable:
    """
    Decorator to profile execution time of a function.

    Usage:
        @profile_time
        def my_function():
            pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f"⏱️  {func.__name__}: {duration:.4f}s")
        return result

    return wrapper


def profile_time_detailed(func: Callable) -> Callable:
    """
    Decorator with more detailed timing info (includes call count).

    Usage:
        @profile_time_detailed
        def my_function():
            pass
    """
    call_count = 0
    total_time = 0.0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal call_count, total_time

        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start

        call_count += 1
        total_time += duration
        avg_time = total_time / call_count

        logger.info(
            f"⏱️  {func.__name__}: "
            f"{duration:.4f}s (call #{call_count}, avg: {avg_time:.4f}s)"
        )
        return result

    return wrapper


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("database query"):
            results = db.query(...)
    """

    def __init__(self, name: str = "block"):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        logger.info(f"⏱️  {self.name}: {duration:.4f}s")


# ============================================================================
# MEMORY PROFILING
# ============================================================================

def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.

    Usage:
        @profile_memory
        def my_function():
            pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(
            f"💾 {func.__name__}: "
            f"current={current / 1024 ** 2:.1f}MB, "
            f"peak={peak / 1024 ** 2:.1f}MB"
        )
        return result

    return wrapper


class MemoryTracker:
    """
    Context manager for tracking memory usage.

    Usage:
        with MemoryTracker("loading data"):
            data = load_large_dataset()
    """

    def __init__(self, name: str = "block"):
        self.name = name

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *args):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(
            f"💾 {self.name}: "
            f"current={current / 1024 ** 2:.1f}MB, "
            f"peak={peak / 1024 ** 2:.1f}MB"
        )


# ============================================================================
# COMBINED PROFILING
# ============================================================================

def profile_all(func: Callable) -> Callable:
    """
    Decorator to profile both time and memory.

    Usage:
        @profile_all
        def my_function():
            pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        duration = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(
            f"📊 {func.__name__}: "
            f"time={duration:.4f}s, "
            f"memory_peak={peak / 1024 ** 2:.1f}MB"
        )
        return result

    return wrapper


class Profiler:
    """
    Context manager for profiling both time and memory.

    Usage:
        with Profiler("entire pipeline"):
            process_data()
    """

    def __init__(self, name: str = "block"):
        self.name = name
        self.start = None

    def __enter__(self):
        tracemalloc.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(
            f"📊 {self.name}: "
            f"time={duration:.4f}s, "
            f"memory_peak={peak / 1024 ** 2:.1f}MB"
        )


# ============================================================================
# PIPELINE PROFILER (for multi-stage processes)
# ============================================================================

class PipelineProfiler:
    """
    Profile multiple stages of a pipeline.

    Usage:
        profiler = PipelineProfiler()

        with profiler.stage("load data"):
            data = load_data()

        with profiler.stage("preprocess"):
            processed = preprocess(data)

        with profiler.stage("model inference"):
            results = model.predict(processed)

        profiler.report()
    """

    def __init__(self):
        self.stages = []
        self.current_stage = None

    def stage(self, name: str):
        """Context manager for a pipeline stage."""
        return self._Stage(self, name)

    class _Stage:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            self.start_time = None

        def __enter__(self):
            tracemalloc.start()
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            duration = time.perf_counter() - self.start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.parent.stages.append({
                'name': self.name,
                'time': duration,
                'memory_peak': peak
            })

    def report(self):
        """Print summary of all stages."""
        total_time = sum(s['time'] for s in self.stages)

        logger.info("\n" + "=" * 70)
        logger.info("📊 PIPELINE PROFILE REPORT")
        logger.info("=" * 70)

        for stage in self.stages:
            pct = (stage['time'] / total_time * 100) if total_time > 0 else 0
            logger.info(
                f"{stage['name']:30s} | "
                f"{stage['time']:6.2f}s ({pct:5.1f}%) | "
                f"{stage['memory_peak'] / 1024 ** 2:6.1f}MB"
            )

        logger.info("=" * 70)
        logger.info(f"{'TOTAL':30s} | {total_time:6.2f}s")
        logger.info("=" * 70 + "\n")