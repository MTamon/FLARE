"""FLARE pipeline package.

リアルタイムパイプラインとバッチ処理パイプラインを提供する。
PipelineBufferによるステージ間データ受け渡し、
FrameDropPolicyによるフレームドロップ制御を含む。
"""

from __future__ import annotations

from typing import List

__all__: List[str] = ["PipelineBuffer", "FrameDropPolicy", "BatchPipeline", "RealtimePipeline"]
