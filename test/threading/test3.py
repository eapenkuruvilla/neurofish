import time
import threading
import sys

# A CPU-intensive function
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def run_single_thread(n_times, n_value):
    start_time = time.time()
    for _ in range(n_times):
        fibonacci(n_value)
    end_time = time.time()
    return end_time - start_time

def run_two_threads(n_times, n_value):
    start_time = time.time()
    
    # Divide the work between two threads
    half_times = n_times // 2
    
    thread1 = threading.Thread(target=lambda: [fibonacci(n_value) for _ in range(half_times)])
    thread2 = threading.Thread(target=lambda: [fibonacci(n_value) for _ in range(half_times)])
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    end_time = time.time()
    return end_time - start_time

# Configuration
ITERATIONS = 20  # Number of times to calculate Fibonacci
FIB_N = 35       # The N value for Fibonacci (adjust for appropriate runtime)

print(f"Testing with {ITERATIONS} iterations of fibonacci({FIB_N})...")

# Check for 'free-threading build' status
if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():
    print("Running in free-threaded build (GIL disabled). True parallelism possible.")
else:
    print("Running in standard Python build (GIL enabled). Multi-threading will not be faster for CPU tasks.")

# Run tests
single_time = run_single_thread(ITERATIONS, FIB_N)
print(f"Single-threaded time: {single_time:.4f} seconds")

threaded_time = run_two_threads(ITERATIONS, FIB_N)
print(f"Two-threaded time:    {threaded_time:.4f} seconds")

# Compare
if threaded_time < single_time:
    print(f"\nTwo threads were faster by {(single_time - threaded_time):.4f} seconds.")
elif threaded_time > single_time:
    print(f"\nSingle thread was faster by {(threaded_time - single_time):.4f} seconds (likely due to GIL overhead).")
else:
    print("\nTimes are approximately equal.")


