"""Resume checkpoint downloads using bounded parallel HTTP byte ranges.

Writes disjoint ranges directly into one file. A small atomic sidecar tracks
completed ranges, so an interrupted sparse file is not mistaken for complete.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def download_checkpoint(url, path, workers=16, chunk_bytes=4 * 1024**2):
    path = Path(path)
    state_path = path.with_suffix(".ranges.json")
    complete = path.with_suffix(".download_complete")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
        size = int(response.headers["Content-Length"])
    if complete.exists():
        if not path.exists() or path.stat().st_size != size:
            raise RuntimeError(f"Completed checkpoint has incorrect size: {path}")
        return
    count = (size + chunk_bytes - 1) // chunk_bytes
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if (state["url"], state["size"]) != (url, size):
            raise RuntimeError("Download state does not match requested checkpoint")
        previous_chunk_bytes = state["chunk_bytes"]
        if previous_chunk_bytes != chunk_bytes:
            if previous_chunk_bytes % chunk_bytes:
                raise RuntimeError("New range size must divide the previous range size")
            expanded = []
            for index in state["completed"]:
                first = index * previous_chunk_bytes // chunk_bytes
                last = (min((index + 1) * previous_chunk_bytes, size) + chunk_bytes - 1) // chunk_bytes
                expanded.extend(range(first, last))
            state.update({"chunk_bytes": chunk_bytes, "completed": expanded})
        if not path.exists() or path.stat().st_size < max(
                (min((i + 1) * chunk_bytes, size) for i in state["completed"]), default=0):
            raise RuntimeError("Checkpoint is shorter than its completed-range state")
    else:
        # Existing files from curl are contiguous prefixes. Once a sidecar
        # exists, only its completed ranges determine resumability.
        prefix = path.stat().st_size if path.exists() else 0
        if prefix > size:
            raise RuntimeError("Existing checkpoint exceeds server size")
        state = {"url": url, "size": size, "chunk_bytes": chunk_bytes,
                 "completed": list(range(prefix // chunk_bytes))}
    lock = threading.Lock()

    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    save()
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    started = time.monotonic()
    initial_completed = len(state["completed"])

    def transfer(index):
        start, end = index * chunk_bytes, min(size, (index + 1) * chunk_bytes) - 1
        for attempt in range(8):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                               "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Server did not honor the exact requested range")
                    offset = start
                    while offset <= end:
                        data = response.read(min(1024**2, end - offset + 1))
                        if not data:
                            raise RuntimeError("Truncated checkpoint range")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(descriptor, view, offset)
                            if written <= 0:
                                raise RuntimeError("Failed to write checkpoint bytes")
                            offset += written
                            view = view[written:]
                with lock:
                    state["completed"].append(index)
                    completed_count = len(state["completed"])
                    if completed_count % 8 == 0 or completed_count == count:
                        # Commit ranges in batches: frequent fsync calls on
                        # network storage serialize otherwise parallel writes.
                        # Only persist completion after the bytes are durable.
                        os.fsync(descriptor)
                        save()
                        elapsed = max(time.monotonic() - started, .001)
                        rate = (completed_count - initial_completed) * chunk_bytes / elapsed / 1024**2
                        print(f"{path.name}: {completed_count}/{count} ranges; {rate:.1f} MiB/s", flush=True)
                return
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 10))

    try:
        pending = [i for i in range(count) if i not in set(state["completed"])]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(transfer, i) for i in pending]
            for job in as_completed(jobs):
                job.result()
        if len(set(state["completed"])) != count or path.stat().st_size != size:
            raise RuntimeError("Checkpoint download is incomplete")
        os.fsync(descriptor)
        save()
        complete.write_text(json.dumps({"url": url, "bytes": size}) + "\n")
    finally:
        os.close(descriptor)
