NODE_RANK=$1
HOST_PORT=$2

#NUMA_0='0|2|4|6|8|10|12|14|16|18|20|22|24|26|28|30|32|34|36|38|40|42|44|46|48|50|52|54|56|58|60|62|64|66|68|70|72|74|76|78|80|82|84|86|88|90|92|94|96|98|100|102|104|106|108|110|112|114|116|118|120|122|124|126|128|130|132|134|136|138|140|142|144|146|148|150|152|154|156|158'
#NUMA_1='1|3|5|7|9|11|13|15|17|19|21|23|25|27|29|31|33|35|37|39|41|43|45|47|49|51|53|55|57|59|61|63|65|67|69|71|73|75|77|79|81|83|85|87|89|91|93|95|97|99|101|103|105|107|109|111|113|115|117|119|121|123|125|127|129|131|133|135|137|139|141|143|145|147|149|151|153|155|157|159'
#
#if [[ ${NODE_RANK} = 0 ]]; then
##  export SGLANG_CPU_OMP_THREADS_BIND=${NUMA_0}
#  export SGLANG_CPU_OMP_THREADS_BIND='0-159'
#else
#  export SGLANG_CPU_OMP_THREADS_BIND=${NUMA_1}
#fi
#
#echo ${SGLANG_CPU_OMP_THREADS_BIND}

export AMX_PARALLEL=1

python3 -m sglang.launch_server \
    --device cpu \
    --mem-fraction-static 0.048 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --tp 2 \
    --dist-init-addr 127.0.0.1:20000 \
    --host 0.0.0.0 --port $HOST_PORT \
    --nnodes 2 \
    --node-rank $NODE_RANK

#    --ep-size 2 \
