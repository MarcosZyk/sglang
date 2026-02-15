# SGLang with Artesia

Stay in this dir for all operations.

## 1. How to Run

### 1.1. Before launch up

Clean residual shared memory or socket resources.

```shell
python clean_shared_memory.py
```

### 1.2. Launch up Artesia


Run ControlPlane

```shell
nohup bash launch_control_plane.sh >cp.log &
```

Check you got following log in cp.log before moving to next step

```
[2026-02-14 17:16:40] INFO launch_server.py:60 - Control-Plane is fired up and ready to roll!
```

Run DataPlane

```shell
nohup bash launch_data_plane.sh >dp.log &
```

Check you got following log in dp.log before moving to next step. Make sure all DataPlanes are ready.

```
...
[2026-02-14 17:16:50] INFO launch_server.py:94 - Data-Plane-[0] is fired up and ready to roll!
...
[2026-02-14 17:16:50] INFO launch_server.py:94 - Data-Plane-[1] is fired up and ready to roll!
```

### 1.3. Launch up SGLang

```shell
nohup bash launch_server.sh >sgl.log &
```

## 2. How to Stop

### 2.1. Stop SGLang

```shell
ps -x

# Find the following SGLang Process
# 76367 pts/2    Sl     0:13 python3 -m sglang.launch_server --mem-fraction-static 0.15 --model-path Qwen/Qwen3-8B --max-total-tokens 4096 --enable-artesia

kill -9 76367
```

### 2.2. Stop Artesia

```shell
ps -x

# Find the following DataPlane Process
# 76301 pts/2    Sl     0:00 /opt/nvidia/hpc_sdk/Linux_aarch64/26.1/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi/bin/.bin/mpirun -n 2 python -m artesia.data_plane.launch_server --device cuda --mem-size 10G --chunk-size 64M
kill 76301

# Find the following ControlPlane Process
# 76207 pts/2    Sl     0:05 python -m artesia.control_plane.launch_server --page-bytes-size 2M
kill 76207

# Clean residual resources if necessary
python clean_shared_memory.py
```








