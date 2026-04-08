import statistics
import torch


def get_l2_flush_buffer():
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    n_elements = l2_bytes // 4
    return torch.empty(n_elements, dtype=torch.int32, device="cuda")


def timed_call(fn, setup=None, flush_buf=None):
    if setup is not None:
        setup()
    if flush_buf is not None:
        flush_buf.zero_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def benchmark(fn, warmup_ms=100.0, rep_ms=1000.0, flush_l2=False, setup=None):
    if setup is not None:
        setup()
    fn()
    torch.cuda.synchronize()

    flush_buf = get_l2_flush_buffer() if flush_l2 else None

    estimate_ms = statistics.median(
        [timed_call(fn, setup, flush_buf) for _ in range(5)]
    )
    n_warmup = max(1, int(warmup_ms / estimate_ms))
    n_repeat = max(1, int(rep_ms / estimate_ms))

    for _ in range(n_warmup):
        timed_call(fn, setup, flush_buf)

    return [timed_call(fn, setup, flush_buf) for _ in range(n_repeat)]


def filter_outliers_iqr(timings, factor=1.5):
    q1, _, q3 = statistics.quantiles(timings)
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return [t for t in timings if lo <= t <= hi]


def compute_stats(timings):
    sorted_t = sorted(timings)
    n = len(timings)
    mean = sum(timings) / n
    med = sorted_t[n // 2]
    variance = sum((t - mean) ** 2 for t in sorted_t) / n
    return {
        "mean_ms": round(mean, 4),
        "median_ms": round(med, 4),
        "std_ms": round(variance**0.5, 4),
        "min_ms": round(sorted_t[0], 4),
        "max_ms": round(sorted_t[-1], 4),
        "p20_ms": round(sorted_t[int(n * 0.2)], 4),
        "p80_ms": round(sorted_t[int(n * 0.8)], 4),
        "n": n,
    }
