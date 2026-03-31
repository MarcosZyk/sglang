# SGLang with Artesia

Stay in this dir for all operations.

## 1. How to Run

### 1.1. Before launch up

Clean residual shared memory or socket resources.

```shell
python clean_shared_memory.py
```

### 1.2. Launch up Artesia

Standalone suite

```shell
cd standalone_artesia
vim standalone_artesia_config.sh
nohup bash launch_artesia_standalone.sh &
```

This suite also provides:

```shell
cd standalone_artesia
bash launch_control_plane.sh
bash launch_data_plane.sh
```

Cross-node suite

On node 0:

```shell
cd crossnode_artesia
vim crossnode_artesia_config.sh
nohup bash launch_artesia_crossnode_head.sh &
```

On node 1:

```shell
cd crossnode_artesia
vim crossnode_artesia_config.sh
nohup bash launch_artesia_crossnode_worker.sh &
```

Each suite has its own single editable config file:

```shell
sglang/dev/scripts/standalone_artesia/standalone_artesia_config.sh
sglang/dev/scripts/crossnode_artesia/crossnode_artesia_config.sh
```

Check you got following log in `artesia-logs/control-plane.log` before moving to next step

```
[2026-02-14 17:16:40] INFO launch_server.py:60 - Control-Plane is fired up and ready to roll!
```

Check you got following log in each `artesia-logs/data-plane-rank-*.log` before moving to next step

```shell
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
# 76301 pts/2    Sl     0:00 python -m artesia.data_plane.launch_server --global-rank 0 ...
# 76302 pts/2    Sl     0:00 python -m artesia.data_plane.launch_server --global-rank 1 ...
kill 76301 76302

# Find the following ControlPlane Process
# 76207 pts/2    Sl     0:05 python -m artesia.control_plane.launch_server --page-bytes-size 2M
kill 76207

# Clean residual resources if necessary
python clean_shared_memory.py
```




