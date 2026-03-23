"""FLAREエラーハンドリングポリシー。

仕様書8.5節「エラーハンドリングポリシー」に基づき、3段階のエラー処理ポリシー
（SKIP / RETRY / ABORT）、カスタム例外階層、パイプラインエラーハンドラ、
および指数バックオフリトライデコレータを提供する。

エラー処理方針:
    リアルタイムモードでは処理継続を最優先とし、バッチモードでは処理精度を優先する。

    ========= ========================== ==========================================
    レベル     対象                        動作
    ========= ========================== ==========================================
    SKIP      顔未検出、特徴抽出失敗       フレームスキップ、前フレーム結果を保持
    RETRY     GPU OOM、一時的I/Oエラー     指数バックオフで最大3回リトライ
    ABORT     モデルロード失敗、設定不正    パイプラインを安全に停止
    ========= ========================== ==========================================

Example:
    >>> from flare.utils.errors import PipelineErrorHandler, ErrorPolicy
    >>> handler = PipelineErrorHandler()
    >>> policy = handler.handle(FaceNotDetectedError("no face"), {})
    >>> policy  # ErrorPolicy.SKIP
"""

from __future__ import annotations

import functools
import time
from enum import Enum
from typing import Any, Callable, Dict, Tuple, Type, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# カスタム例外階層
# =============================================================================


class FLAREBaseError(Exception):
    """FLAREの全例外の基底クラス。

    全てのFLARE固有例外はこのクラスを継承する。
    標準のExceptionを直接継承し、FLARE由来の例外を一括で
    キャッチできるようにする。

    Example:
        >>> try:
        ...     raise FaceNotDetectedError("顔が検出されません")
        ... except FLAREBaseError as e:
        ...     print(f"FLAREエラー: {e}")
    """


class FaceNotDetectedError(FLAREBaseError):
    """顔検出失敗エラー。

    face_detect.pyによる顔検出で顔が見つからなかった場合に送出される。
    PipelineErrorHandlerではSKIPポリシーとして処理され、
    当該フレームをスキップして前フレームの結果を保持する。
    """


class ModelLoadError(FLAREBaseError):
    """モデルロード失敗エラー。

    Extractor, Renderer, LHGモデルのチェックポイントファイルの
    読み込みに失敗した場合に送出される。
    PipelineErrorHandlerではABORTポリシーとして処理され、
    パイプラインを安全に停止する。
    """


class ConfigError(FLAREBaseError):
    """設定ファイル関連のエラー。

    YAML設定ファイルの読み込み失敗、パース失敗、
    pydanticバリデーション失敗時に送出される。
    PipelineErrorHandlerではABORTポリシーとして処理される。
    """


class PipelineError(FLAREBaseError):
    """パイプライン実行時エラー。

    パイプラインの初期化・実行中に発生する一般的なエラー。
    スレッド間通信の障害やパイプライン状態の不整合等に使用する。
    """


class ConverterError(FLAREBaseError):
    """パラメータ変換エラー。

    converters/モジュールでのパラメータ変換中に発生するエラー。
    入力パラメータの形式不正や変換処理の失敗時に送出される。
    """


class BufferOverflowError(FLAREBaseError):
    """バッファオーバーフローエラー。

    PipelineBufferの容量超過時に送出される。
    通常はdrop_oldestポリシーで自動処理されるが、
    blockポリシー使用時のタイムアウトで発生する可能性がある。
    """


# =============================================================================
# ErrorPolicy
# =============================================================================


class ErrorPolicy(Enum):
    """エラー処理ポリシーの3段階。

    パイプラインのエラーハンドラが返すポリシーにより、
    パイプラインの後続動作が決定される。

    Attributes:
        SKIP: 当該フレームをスキップし処理を継続する。
        RETRY: 指数バックオフでリトライする。
        ABORT: パイプラインを安全に停止する。
    """

    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


# =============================================================================
# PipelineErrorHandler
# =============================================================================


class PipelineErrorHandler:
    """パイプラインエラーハンドラ。

    例外の種類に応じて適切なErrorPolicyを判定し、対応するログを出力する。
    リアルタイムモードでは処理継続を最優先とし、未知の例外に対しても
    SKIPポリシーを返す。

    Example:
        >>> handler = PipelineErrorHandler()
        >>> try:
        ...     result = extractor.extract(image)
        ... except Exception as e:
        ...     policy = handler.handle(e, {"frame_index": 42})
        ...     if policy == ErrorPolicy.SKIP:
        ...         continue
    """

    def handle(self, error: Exception, context: Dict[str, Any]) -> ErrorPolicy:
        """例外を受け取り、適切なErrorPolicyを返す。

        例外の型に基づいてログ出力とポリシー判定を行う。
        GPU OOMの場合はCUDAキャッシュのクリアも実行する。

        Args:
            error: 発生した例外インスタンス。
            context: エラー発生時のコンテキスト情報。
                例: ``{"frame_index": 42, "module": "extractor"}``

        Returns:
            判定されたエラー処理ポリシー。

        Example:
            >>> policy = handler.handle(
            ...     FaceNotDetectedError("no face"),
            ...     {"frame_index": 100}
            ... )
            >>> policy  # ErrorPolicy.SKIP
        """
        if isinstance(error, FaceNotDetectedError):
            logger.warning(
                "顔検出失敗: フレームをスキップします | "
                "error={} | context={}",
                error,
                context,
            )
            return ErrorPolicy.SKIP

        if _is_cuda_oom(error):
            logger.error(
                "GPU OOM: CUDAキャッシュをクリアしてリトライします | "
                "error={} | context={}",
                error,
                context,
            )
            _safe_cuda_empty_cache()
            return ErrorPolicy.RETRY

        if isinstance(error, (ModelLoadError, ConfigError)):
            logger.critical(
                "致命的エラー: パイプラインを停止します | "
                "error={} | context={}",
                error,
                context,
            )
            return ErrorPolicy.ABORT

        # その他の例外: リアルタイムでは継続優先
        logger.opt(exception=error).error(
            "予期しないエラー: フレームをスキップします | "
            "error={} | context={}",
            error,
            context,
        )
        return ErrorPolicy.SKIP


# =============================================================================
# retry_with_backoff デコレータ
# =============================================================================


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """指数バックオフリトライデコレータ。

    指定された例外が発生した場合に、指数バックオフ
    （``delay = base_delay * 2^attempt``）でリトライする。
    ``max_retries`` 回失敗後は例外をそのまま再送出する。

    Args:
        max_retries: 最大リトライ回数。デフォルト3回。
        base_delay: 基本遅延時間（秒）。デフォルト0.5秒。
            実際の遅延は ``base_delay * 2^attempt`` となる。
        exceptions: リトライ対象の例外型のタプル。
            デフォルトは全例外。

    Returns:
        デコレータ関数。

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=0.1,
        ...                     exceptions=(IOError, TimeoutError))
        ... def fetch_data():
        ...     # 一時的に失敗する可能性のある処理
        ...     ...
    """

    def decorator(func: F) -> F:
        """対象関数をリトライロジックでラップする。

        Args:
            func: ラップ対象の関数。

        Returns:
            リトライロジック付きの関数。
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """リトライロジックを実行する。

            Args:
                *args: 対象関数の位置引数。
                **kwargs: 対象関数のキーワード引数。

            Returns:
                対象関数の戻り値。

            Raises:
                Exception: max_retries回リトライ後も失敗した場合、
                    最後の例外を再送出する。
            """
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc

                    if attempt >= max_retries:
                        logger.error(
                            "リトライ上限到達: {} | 関数={} | 試行={}/{}",
                            exc,
                            func.__name__,
                            attempt + 1,
                            max_retries + 1,
                        )
                        raise

                    delay: float = base_delay * (2**attempt)
                    logger.warning(
                        "リトライ: {} | 関数={} | 試行={}/{} | "
                        "次回遅延={:.2f}秒",
                        exc,
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)

            # 型チェッカー向けガード（ロジック上は到達しない）
            if last_exception is not None:  # pragma: no cover
                raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# 内部ヘルパー
# =============================================================================


def _is_cuda_oom(error: Exception) -> bool:
    """例外がCUDA Out of Memoryエラーかどうかを判定する。

    torch.cuda.OutOfMemoryErrorが利用可能な場合はisinstanceで判定し、
    利用不可の場合（CPU環境等）はFalseを返す。

    Args:
        error: 判定対象の例外。

    Returns:
        CUDA OOMエラーの場合True。
    """
    try:
        import torch

        return isinstance(error, torch.cuda.OutOfMemoryError)
    except (ImportError, AttributeError):
        return False


def _safe_cuda_empty_cache() -> None:
    """CUDAキャッシュを安全にクリアする。

    torch.cuda.empty_cache()を呼び出す。
    torchが利用不可の場合は何もしない。
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass
