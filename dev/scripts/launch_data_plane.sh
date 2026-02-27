mpirun -n 2 python -m artesia.data_plane.launch_server \
    --device cuda \
    --mem-size 60G \
    --chunk-size 64M
