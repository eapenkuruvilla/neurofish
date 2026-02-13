import threading
import time

# Simulate TT access pattern
shared_dict = {}

def worker_dict(n, results, idx, use_shared):
    if use_shared:
        d = shared_dict
    else:
        d = {}  # Thread-local dict
    
    start = time.perf_counter()
    for i in range(n):
        key = (idx * 1000000) + i  # Unique keys per thread
        # Simulate TT lookup + store pattern
        val = d.get(key)
        if val is None:
            d[key] = (5, 100, 0, 12345)  # TTEntry-like tuple
    results[idx] = time.perf_counter() - start

N = 500000

# Test 1: Separate dicts (no contention)
print("=== Separate dicts (no contention) ===")
results_1 = [0]
worker_dict(N, results_1, 0, False)
print(f"1 thread: {results_1[0]:.3f}s")

results_2 = [0, 0]
t1 = threading.Thread(target=worker_dict, args=(N, results_2, 0, False))
t2 = threading.Thread(target=worker_dict, args=(N, results_2, 1, False))
t1.start(); t2.start()
t1.join(); t2.join()
print(f"2 threads: {max(results_2):.3f}s")
print(f"Efficiency: {results_1[0] / max(results_2) * 100:.1f}%")

# Test 2: Shared dict (contention)
print("\n=== Shared dict (TT-like contention) ===")
shared_dict.clear()
results_1 = [0]
worker_dict(N, results_1, 0, True)
print(f"1 thread: {results_1[0]:.3f}s")

shared_dict.clear()
results_2 = [0, 0]
t1 = threading.Thread(target=worker_dict, args=(N, results_2, 0, True))
t2 = threading.Thread(target=worker_dict, args=(N, results_2, 1, True))
t1.start(); t2.start()
t1.join(); t2.join()
print(f"2 threads: {max(results_2):.3f}s")
print(f"Efficiency: {results_1[0] / max(results_2) * 100:.1f}%")
