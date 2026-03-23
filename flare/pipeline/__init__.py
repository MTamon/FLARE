"""FLARE pipeline package.

リアルタイムパイプラインとバッチ処理パイプラインを提供する。
PipelineBufferによるステージ間データ受け渡し、
FrameDropPolicyによるフレームドロップ制御を含む。
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

# --- PipelineBuffer / FrameDropPolicy ---
try:
    from flare.pipeline.buffer import FrameDropPolicy as FrameDropPolicy
    from flare.pipeline.buffer import PipelineBuffer as PipelineBuffer

    __all__.extend(["PipelineBuffer", "FrameDropPolicy"])
except ImportError:
    pass

# --- BatchPipeline ---
try:
    from flare.pipeline.batch import BatchPipeline as BatchPipeline

    __all__.append("BatchPipeline")
except ImportError:
    pass

# --- RealtimePipeline ---
try:
    from flare.pipeline.realtime import RealtimePipeline as RealtimePipeline

    __all__.append("RealtimePipeline")
except ImportError:
    pass
