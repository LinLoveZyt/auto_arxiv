#!/bin/bash

# set -e: 脚本中的任何命令失败，立即退出。这是一个保证脚本健壮性的好习惯。
set -e

# --- 配置 ---
# 定义你的Conda环境名称
CONDA_ENV_NAME="auto_arvix"

# --- 激活Conda环境 ---
# Shell脚本无法直接使用'conda activate'，需要先'source' conda的配置文件。
# 这通常位于你的Miniconda或Anaconda安装目录下。
# 如果下面的路径不正确，请根据你的实际安装路径进行修改。
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误：找不到Conda配置文件。请检查脚本中的路径。"
    exit 1
fi

# 激活指定的环境
conda activate ${CONDA_ENV_NAME}
echo "✅ Conda环境 '${CONDA_ENV_NAME}' 已激活。"



# ▼▼▼ 新增环境变量 ▼▼▼
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 命令分发 ---
# $1 代表用户输入的第一个参数 (如 "server", "daily")
COMMAND=$1

# 使用 case 语句来处理不同的命令
case ${COMMAND} in
    "server")
        echo "🚀 正在启动FastAPI服务器..."
        # $@ 会将传递给脚本的所有其他参数 (如 --skip-daily-check) 传递给python命令
        python main.py run-server "${@:2}"
        ;;
    "daily")
        echo "📅 正在独立运行每日任务..."
        python main.py daily
        ;;
    "install")
        echo "📦 正在安装/更新依赖..."
        # 从Conda安装核心GPU库
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        conda install faiss-gpu -c pytorch -y
        # 从pip安装其余的库
        pip install -r requirements.txt
        echo "✅ 依赖安装完成。"
        ;;
    "test-gpu")
        echo "🔍 正在运行GPU环境验证脚本..."
        python verify_gpu.py
        ;;
    *)
        echo "用法: $0 {server|daily|install|test-gpu}"
        echo "  - server: 启动FastAPI服务器 (并自动运行每日任务)。"
        echo "  - daily:  仅运行每日任务，不启动服务器。"
        echo "  - install: 安装所有项目依赖。"
        echo "  - test-gpu: 运行GPU环境验证脚本。"
        exit 1
        ;;
esac