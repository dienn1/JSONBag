from typing import *
from numpy.typing import *
from tqdm.auto import tqdm
from multiprocessing.sharedctypes import RawArray
import multiprocessing as mp
import ctypes
import numpy as np

from distance_matrix_wrapper import DistanceMatrixWrapper


# Generate a flatten distance matrix
def _generate_distance_matrix(data: List[str], dist: Callable[[Any, Any], float]) -> NDArray:
    print("Generating Distance Matrix (flatten) ...")
    n = len(data)
    dist_matrix_f = np.zeros(n*(n+1) // 2, dtype=np.float32)
    index = 0
    for i in tqdm(range(n)):
        for j in range(i, n):
            dist_matrix_f[index] = dist(data[i], data[j]) if i != j else 0
            index += 1
    return dist_matrix_f


def generate_distance_matrix(data: List[str], dist: Callable[[Any, Any], float], num_workers=0) -> NDArray:
    if num_workers == 0:
        return _generate_distance_matrix(data, dist)
    n = len(data)
    array_size = n*(n+1) // 2
    shared_array = RawArray(ctypes.c_float, array_size)     # Shared array for multiprocessing
    dist_matrix_f = np.frombuffer(shared_array, dtype=np.float32, count=array_size)
    dist_matrix_f.fill(0)
    workers = list()
    chunk_size = array_size // num_workers
    chunk_remain = array_size % num_workers
    print("Generating Distance Matrix (flatten) with num_workers={} ...".format(num_workers))
    for i in range(num_workers - 1):
        start = i * chunk_size
        stop = (i + 1) * chunk_size
        workers.append(mp.Process(target=_compute_distance_matrix, args=(shared_array, start, stop, data, dist,)))
    print("Starting workers ...")
    for p in workers:
        p.start()
    print("Main process working ...")
    start = (num_workers - 1) * chunk_size
    stop = num_workers*chunk_size + chunk_remain
    dist_matrix = DistanceMatrixWrapper(dist_matrix_f)
    for i in tqdm(range(start, stop)):
        x, y = dist_matrix.convert_flatten_index(i)
        dist_matrix_f[i] = dist(data[x], data[y]) if x != y else 0
    print("Main process finished, joining workers ...")
    for p in workers:
        p.join()
    return dist_matrix_f


def _compute_distance_matrix(dist_matrix_shared: RawArray, start: int, stop: int, data: List[str], dist: Callable[[Any, Any], float]):
    dist_matrix_f = np.frombuffer(dist_matrix_shared, dtype=np.float32, count=len(dist_matrix_shared))
    dist_matrix = DistanceMatrixWrapper(dist_matrix_f)
    for i in range(start, stop):
        x, y = dist_matrix.convert_flatten_index(i)
        dist_matrix_f[i] = dist(data[x], data[y]) if x != y else 0


def _generate_cross_distance_matrix(data1: List[str], data2: List[str], dist: Callable[[Any, Any], float]) -> NDArray:
    print("Generating cross Distance Matrix...")
    dist_matrix = np.zeros((len(data1), len(data2)), dtype=np.float32)
    for i in tqdm(range(len(data1))):
        for j in range(len(data2)):
            dist_matrix[i, j] = dist(data1[i], data2[j])
    return dist_matrix


def generate_cross_distance_matrix(data1: List[str], data2: List[str], dist: Callable[[Any, Any], float], num_workers=0) -> NDArray:
    if num_workers == 0:
        return _generate_cross_distance_matrix(data1, data2, dist)
    array_size = len(data1) * len(data2)
    shared_array = RawArray(ctypes.c_float, array_size)  # Shared array for multiprocessing
    dist_matrix_f = np.frombuffer(shared_array, dtype=np.float32, count=array_size)
    dist_matrix_f.fill(69)
    workers = list()
    chunk_size = array_size // num_workers
    chunk_remain = array_size % num_workers
    print("Generating cross Distance Matrix with num_workers={} ...".format(num_workers))
    for i in range(num_workers - 1):
        start = i * chunk_size
        stop = (i + 1) * chunk_size
        # print(f"Processing chunk {start} to {stop-1}")
        workers.append(mp.Process(target=_compute_cross_distance_matrix, args=(shared_array, start, stop, data1, data2, dist, )))
    print("Starting workers ...")
    for p in workers:
        p.start()
    print("Main process working ...")
    start = (num_workers - 1) * chunk_size
    stop = num_workers*chunk_size + chunk_remain
    col = len(data2)
    for i in tqdm(range(start, stop)):
        x = i // col
        y = i % col
        dist_matrix_f[i] = dist(data1[x], data2[y])
    print("Main process finished, joining workers ...")
    for p in workers:
        p.join()
    return dist_matrix_f.reshape(len(data1), len(data2))


def _compute_cross_distance_matrix(dist_matrix_shared: RawArray, start: int, stop: int, data1: List[str], data2: List[str], dist: Callable[[Any, Any], float]):
    dist_matrix_f = np.frombuffer(dist_matrix_shared, dtype=np.float32, count=len(dist_matrix_shared))
    col = len(data2)
    for i in range(start, stop):
        x = i // col
        y = i % col
        dist_matrix_f[i] = dist(data1[x], data2[y])


def summarize_dist_matrix(dist_matrix_flat: NDArray, state_names: Union[List[str], None], ax=None, name=""):
    print("AVERAGE DISTANCE:", dist_matrix_flat.mean())

    dist_matrix_wrapper = DistanceMatrixWrapper(dist_matrix_flat)
    max_distance = dist_matrix_flat.max()
    max_i = dist_matrix_flat.argmax()
    # print(max_i)
    max_i0, max_i1 = dist_matrix_wrapper.convert_flatten_index(max_i)
    if state_names is not None:
        detail_str = f"between {max_i0} {state_names[max_i0]} and {max_i1} {state_names[max_i1]}"
    else:
        detail_str = ""
    print("MAX DISTANCE:", max_distance, detail_str)

    min_distance = dist_matrix_flat[dist_matrix_flat != 0].min()
    min_i = np.where(dist_matrix_flat == min_distance)
    # print(min_i)
    min_i0, min_i1 = dist_matrix_wrapper.convert_flatten_index(min_i[0][0])
    if state_names is not None:
        detail_str = f"between {min_i0} {state_names[min_i0]} and {min_i1} {state_names[min_i1]}"
    else:
        detail_str = ""
    print("MIN (non-zero) DISTANCE:", min_distance, detail_str)
    print("STANDARD DEVIATION: ", np.std(dist_matrix_flat[dist_matrix_flat != 0]))

    if ax is None:
        return
    ax.hist(dist_matrix_flat[dist_matrix_flat != 0], bins=50, color='tab:cyan', edgecolor='black')
    ax.set_title(name)