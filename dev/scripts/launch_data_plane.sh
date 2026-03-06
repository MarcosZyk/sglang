mpirun -n 2 \
    -x OMPI_COMM_WORLD_LOCAL_RANK \
    -x OMPI_COMM_WORLD_RANK \
    python -m artesia.data_plane.launch_server \
    --device cuda \
    --mem-size 60G \
    --chunk-size 256M
