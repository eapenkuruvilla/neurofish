import threading
import time
from libs.chess_cpp import Board

def worker(n, results, idx):
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    start = time.perf_counter()
    for _ in range(n):
        moves = board.legal_moves()
        for m in moves[:5]:
            board.push(m)
            board.pop()
    results[idx] = time.perf_counter() - start

N = 50000
results_1 = [0]
results_2 = [0, 0]

# Single thread
worker(N, results_1, 0)
print(f"1 thread: {results_1[0]:.3f}s")

# Two threads
t1 = threading.Thread(target=worker, args=(N, results_2, 0))
t2 = threading.Thread(target=worker, args=(N, results_2, 1))
t1.start(); t2.start()
t1.join(); t2.join()
print(f"2 threads: {max(results_2):.3f}s (ideal: {results_1[0]:.3f}s)")
print(f"Parallel efficiency: {results_1[0] / max(results_2) * 100:.1f}%")
