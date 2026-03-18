"""カスタム例外 + ErrorPolicy (Section 8.5)

3 段階のエラー処理ポリシー:
  SKIP  — 顔未検出、特徴抽出失敗 → 当該フレームをスキップし前フレーム結果を保持
  RETRY — GPU OOM、一時的 I/O エラー → 指数バックオフで最大 3 回リトライ
  ABORT — モデルロード失敗、設定不正、致命的エラー → パイプラインを安全に停止
"""

from __future__ import annotations

import enum
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


# ---------------------------------------------------------------------------
# ErrorPolicy enum
# ---------------------------------------------------------------------------

class ErrorPolicy(enum.Enum):
    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------

class LHGToolkitError(Exception):
    """lhg_toolkit の基底例外。"""


class FaceNotDetectedError(LHGToolkitError):
    """顔が検出されなかった場合。"""


class FeatureExtractionError(LHGToolkitError):
    """特徴量抽出に失敗した場合。"""


class ModelLoadError(LHGToolkitError):
    """モデルのロードに失敗した場合。"""


class ConfigError(LHGToolkitError):
    """設定ファイルのバリデーションエラー。"""


class RendererNotInitializedError(LHGToolkitError):
    """setup() が呼ばれる前に render() が呼ばれた場合。"""


# ---------------------------------------------------------------------------
# PipelineErrorHandler (Section 8.5 コード例準拠)
# ---------------------------------------------------------------------------

class PipelineErrorHandler:
    """パイプライン内で発生した例外を ErrorPolicy に分類し、
    RETRY 時には指数バックオフを実行する。

    リアルタイムモードでは処理継続を最優先、
    バッチモードでは処理精度を優先する。
    """

    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_BACKOFF_BASE: float = 0.5  # 秒

    def classify(self, error: Exception) -> ErrorPolicy:
        """例外を ErrorPolicy に分類する。"""
        import torch  # lazy import — torch 未インストール時でも errors.py を読み込めるように

        if isinstance(error, (FaceNotDetectedError, FeatureExtractionError)):
            return ErrorPolicy.SKIP

        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return ErrorPolicy.RETRY

        if isinstance(error, (IOError, OSError)):
            return ErrorPolicy.RETRY

        if isinstance(error, (ModelLoadError, ConfigError)):
            return ErrorPolicy.ABORT

        # 未知の例外 — リアルタイムでは継続優先
        logger.error(f"Unexpected error: {error}")
        return ErrorPolicy.SKIP

    def execute_with_policy(
        self,
        fn: Callable[..., T],
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        **kwargs: Any,
    ) -> Optional[T]:
        """fn を実行し、失敗時は ErrorPolicy に従って処理する。

        Returns:
            fn の戻り値。SKIP の場合は None。

        Raises:
            Exception: ABORT の場合は元の例外を再送出する。
        """
        retries = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                policy = self.classify(exc)
                ctx_str = f" context={context}" if context else ""

                if policy is ErrorPolicy.SKIP:
                    logger.warning(
                        f"SKIP: {type(exc).__name__}: {exc}{ctx_str}"
                    )
                    return None

                if policy is ErrorPolicy.RETRY:
                    retries += 1
                    if retries > max_retries:
                        logger.warning(
                            f"RETRY exhausted ({max_retries}), "
                            f"falling back to SKIP: {exc}{ctx_str}"
                        )
                        return None
                    wait = backoff_base * (2 ** (retries - 1))
                    logger.info(
                        f"RETRY {retries}/{max_retries} after {wait:.2f}s: "
                        f"{type(exc).__name__}: {exc}{ctx_str}"
                    )
                    time.sleep(wait)
                    continue

                # ABORT
                logger.critical(
                    f"ABORT: {type(exc).__name__}: {exc}{ctx_str}"
                )
                raise