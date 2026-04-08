#!/bin/bash
# install_torch.sh
# PyTorch 2.9.0 + CUDA 12.8 wheel のインストール
#
# Usage:
#   bash install_torch.sh --cuda 12.8

if [ "$1" != "--cuda" ] || [ -z "$2" ]; then
    echo "Usage: bash install_torch.sh --cuda <CUDA_VERSION>"
    exit 1
fi

CUDA_VERSION=$2 # 12.8

if [ "$CUDA_VERSION" = "12.8" ]; then
    # torch==2.9.0+cu128 などがこれでinstallできる
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
else
    echo "Usage: bash install_torch.sh --cuda <CUDA_VERSION>"
    echo "CUDA_VERSION must be 12.8"
    exit 1
fi

pip install --no-deps torchao==0.14.1
# pip install --no-deps torchao==0.3.1 # this is not available due to the import error
pip install --no-deps torchmetrics==1.8.2
pip install --no-deps torchsummary==1.5.1
pip install --no-deps torchtune==0.6.1
pip install --no-deps torchviz==0.0.3
pip install --no-deps torchcodec==0.8.1
