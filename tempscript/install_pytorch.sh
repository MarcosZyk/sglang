#!/bin/zsh

source /root/.zshrc
cd /root/pytorch_cpu
conda activate sglamx-dong1

# Install the following dependencies if you want to install pytorch_cpu in a new conda env, these dependencies are already included in "torch2.8+ucc"
conda install -y cmake ninja
pip install -r /root/pytorch_cpu/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install mkl-static mkl-include -i https://mirrors.aliyun.com/pypi/simple/

export USE_UCC=1
export USE_SYSTEM_UCC=1
export UCC_DIR=/opt/ucc/lib/cmake/ucc
export USE_CUDA=0
export CMAKE_PREFIX_PATH=/opt/ucc

python setup.py develop -i https://mirrors.aliyun.com/pypi/simple/