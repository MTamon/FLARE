#!/bin/bash
# build_environment.sh
# FLARE 環境構築スクリプト
# CUDA バージョンを引数に取り、install_torch.sh を呼び出した後、
# 約120個の pip パッケージを固定バージョンで一括インストール
#
# Usage:
#   bash install/build_environment.sh --cuda 12.8

if [ "$1" != "--cuda" ] || [ -z "$2" ]; then
    echo "Usage: bash install/build_environment.sh --cuda <CUDA_VERSION>"
    exit 1
fi

bash install/python3.11.0/install_torch.sh --cuda $2

pip install --no-deps absl-py==2.3.1
pip install --no-deps accelerate==1.11.0
pip install --no-deps aiohappyeyeballs==2.6.1
pip install --no-deps aiohttp==3.13.2
pip install --no-deps aiosignal==1.4.0
pip install --no-deps annotated-types==0.7.0
pip install --no-deps antlr4-python3-runtime==4.9.3
pip install --no-deps anyio==4.11.0
pip install --no-deps attrs==25.4.0
pip install --no-deps audioread==3.1.0
pip install --no-deps blobfile==3.1.0
pip install --no-deps certifi==2025.10.5
pip install --no-deps cffi==2.0.0
pip install --no-deps charset-normalizer==3.4.4
pip install --no-deps click==8.3.0
pip install --no-deps cmpfilter==0.0.1
pip install --no-deps contourpy==1.3.3
pip install --no-deps cycler==0.12.1
pip install --no-deps Cython==0.29.35
pip install --no-deps datasets==4.4.1
pip install --no-deps decorator==5.2.1
pip install --no-deps dfcon==0.3.0
pip install --no-deps dill==0.4.0
pip install --no-deps dtaidistance==2.3.12
pip install --no-deps easy-video==0.0.4
pip install --no-deps evaluate==0.4.6
pip install --no-deps filelock==3.19.1
pip install --no-deps flatbuffers==25.9.23
pip install --no-deps fonttools==4.60.1
pip install --no-deps frozenlist==1.8.0
pip install --no-deps gitdb==4.0.12
pip install --no-deps GitPython==3.1.45
pip install --no-deps graphviz==0.21
pip install --no-deps h11==0.16.0
pip install --no-deps hf-xet==1.2.0
pip install --no-deps httpcore==1.0.9
pip install --no-deps httpx==0.28.1
pip install --no-deps huggingface-hub==0.36.0
pip install --no-deps hydra-core==1.3.2
pip install --no-deps idna==3.11
pip install --no-deps ImageIO==2.37.2
pip install --no-deps imageio-ffmpeg==0.6.0
# pip install --no-deps importlib_metadata==8.0.0 # unnecessary?
# pip install --no-deps importlib_resources==6.4.0 # unnecessary?
pip install --no-deps jax==0.7.1
pip install --no-deps jaxlib==0.7.1
pip install --no-deps jiwer==4.0.0
pip install --no-deps joblib==1.3.2
pip install --no-deps kagglehub==0.3.13
pip install --no-deps kiwisolver==1.4.9
pip install --no-deps lazy_loader==0.4
pip install --no-deps librosa==0.11.0
pip install --no-deps lightning-utilities==0.15.2
pip install --no-deps llvmlite==0.45.1
pip install --no-deps lxml==6.0.2
pip install --no-deps matplotlib==3.10.7
pip install --no-deps mediapipe==0.10.11
pip install --no-deps ml-dtypes==0.5.3
pip install --no-deps moviepy==2.2.1
pip install --no-deps msgpack==1.1.2
pip install --no-deps multidict==6.7.0
pip install --no-deps multiprocess==0.70.18
pip install --no-deps networkx==3.5
pip install --no-deps numba==0.62.1
pip install --no-deps numpy==2.2.6
pip install --no-deps omegaconf==2.3.0
pip install --no-deps opencv-contrib-python==4.11.0.86
pip install --no-deps opencv-python==4.12.0.88
pip install --no-deps opt_einsum==3.4.0
pip install --no-deps packaging==25.0
pip install --no-deps pandas==2.3.3
pip install --no-deps pillow==11.3.0
pip install --no-deps platformdirs==4.5.0
pip install --no-deps pooch==1.8.2
pip install --no-deps proglog==0.1.12
pip install --no-deps propcache==0.4.1
pip install --no-deps protobuf==3.20.3
pip install --no-deps psutil==7.1.3
pip install --no-deps pyarrow==22.0.0
pip install --no-deps pycparser==2.23
pip install --no-deps pycryptodomex==3.23.0
pip install --no-deps pydantic==2.12.4
pip install --no-deps pydantic_core==2.41.5
pip install --no-deps pyparsing==3.2.5
pip install --no-deps python-dateutil==2.9.0.post0
pip install --no-deps python-dotenv==1.2.1
pip install --no-deps pytorch-lightning==2.5.6
pip install --no-deps pytz==2025.2
pip install --no-deps pyworld==0.3.5
pip install --no-deps PyYAML==6.0.3
pip install --no-deps rapidfuzz==3.14.3
pip install --no-deps regex==2025.11.3
pip install --no-deps requests==2.32.5
pip install --no-deps safetensors==0.6.2
pip install --no-deps schedulefree==1.4.1
pip install --no-deps scikit-learn==1.7.2
pip install --no-deps scipy==1.16.3
pip install --no-deps sentencepiece==0.2.1
pip install --no-deps sentry-sdk==2.43.0
pip install --no-deps shellingham==1.5.4
pip install --no-deps six==1.17.0
pip install --no-deps smmap==5.0.2
pip install --no-deps sniffio==1.3.1
pip install --no-deps sounddevice==0.5.3
pip install --no-deps soundfile==0.13.1
pip install --no-deps soxr==1.0.0
pip install --no-deps sympy==1.14.0
pip install --no-deps threadpoolctl==3.6.0
pip install --no-deps tiktoken==0.12.0
pip install --no-deps tokenizers==0.22.1
pip install --no-deps toolpack==0.0.5
pip install --no-deps tqdm==4.66.3
pip install --no-deps transformers==4.57.1
pip install --no-deps typer-slim==0.20.0
pip install --no-deps typing-inspection==0.4.2
pip install --no-deps typing_extensions==4.15.0
pip install --no-deps tzdata==2025.2
pip install --no-deps urllib3==2.5.0
pip install --no-deps wandb==0.22.3
pip install --no-deps xxhash==3.6.0
pip install --no-deps yarl==1.22.0
pip install --no-deps zipp==3.23.0
