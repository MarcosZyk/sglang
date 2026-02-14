import os
from multiprocessing import shared_memory

for i in range(1000):
    try:
        name = f"/tmp/artesia_socket_{i}"
        os.remove(name)
    except FileNotFoundError:
        pass

    try:
        name = f"artesia-cpu-shm-{i}"
        shm = shared_memory.SharedMemory(name)
        shm.close()
        shm.unlink()
        print(f"Clean zombie SHM instance {name}", flush=True)
    except FileNotFoundError:
        pass
