mpirun -n 2 python -m artesia.data_plane.launch_server \
    --device cuda \
    --mem-size 10G \
    --chunk-size 64M
