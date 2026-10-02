"""Ordered thread scheduling shared by the crop and recognition stages."""
from concurrent.futures import ThreadPoolExecutor, as_completed


def map_ordered(items, fn, workers: int, progress=None):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        results = []
        for index, item in enumerate(items, 1):
            results.append(fn(item))
            if progress:
                progress(index, len(items))
        return results

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        future_to_index = {
            executor.submit(fn, item): index
            for index, item in enumerate(items)
        }
        for done, future in enumerate(as_completed(future_to_index), 1):
            results[future_to_index[future]] = future.result()
            if progress:
                progress(done, len(items))
    return results
