#!/bin/bash
set -e
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"

# Install python and dependencies to specified position
[ -f Miniconda3-latest-Linux-x86_64.sh ] || wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
[ -d miniconda ] || bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $script_dir/miniconda
$script_dir/miniconda/bin/conda create \
	    --prefix $script_dir/python-occlum -y \
	        python=3.10.0

$script_dir/python-occlum/bin/pip install torch-2.2.0+cpu-cp310-cp310-linux_x86_64.whl
$script_dir/python-occlum/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
