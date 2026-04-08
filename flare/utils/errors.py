"""エラーハンドリングモジュール。

パイプライン実行時のエラーを3段階のポリシー（SKIP / RETRY / ABORT）で処理する。
リアルタイムモードでは処理継続を最優先とし、バッチモードでは処理精度を優先する。

仕様書§8.5に基づくエラー分類:
    - SKIP: 顔未検出、特徴抽出失敗 → 当該フレームをスキップし前フレーム結果を保持。
    - RETRY: GPU OOM、一時的I/Oエラー → 指数バックオフで最大3回リトライ。
    - ABORT: モデルロード失敗、設定不正、致命的エラー → パイプラインを安全に停止。

Example:
    PipelineErrorHandlerの使用::

        handler = PipelineErrorHandler()
        try:
            result = extractor.extract(image)
        except Exception as e:
            policy = handler.handle(e, {"frame_index": 42})
            if policy == ErrorPolicy.SKIP:
                result = previous_result
            elif policy == ErrorPolicy.RETRY:
                ...
            elif policy == ErrorPolicy.ABORT:
                raise SystemExit(1)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
from loguru import logger


class ErrorPolicy(Enum):
    """エラー発生時の処理ポリシー。

    Attributes:
        SKIP: 当該フレームをスキップし前フレーム結果を保持する。
            ログにWARNINGを記録する。
        RETRY: 指数バックオフで最大3回リトライする。
            失敗時はSKIPへフォールバックする。
        ABORT: パイプラインを安全に停止する。
            チェックポイントを保存後に終了する。
    """

    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


class FaceNotDetectedError(Exception):
    """顔検出に失敗した場合に送出される例外。

    face_detect.pyが入力フレームから顔領域を検出できなかった場合に使用する。
    パイプラインではSKIPポリシーで処理され、前フレームの結果が保持される。
    """


class ModelLoadError(Exception):
    """モデルのロードに失敗した場合に送出される例外。

    事前学習済みモデルファイルの読み込み、重みの復元、またはモデルの初期化に
    失敗した場合に使用する。パイプラインではABORTポリシーで処理される。
    """


class ConfigError(Exception):
    """設定の読み込みまたはバリデーションに失敗した場合に送出される例外。

    YAML設定ファイルの構文エラー、pydanticバリデーション失敗、または
    必須設定項目の欠落に使用する。パイプラインではABORTポリシーで処理される。
    """


class RendererNotInitializedError(Exception):
    """Rendererが未初期化の状態でrender()が呼ばれた場合に送出される例外。

    BaseRenderer.setup()が完了する前にrender()を呼び出した場合に使用する。
    パイプラインではABORTポリシーで処理される。
    """


class PipelineErrorHandler:
    """パイプラインのエラーハンドラ。

    発生した例外の種類に応じて適切なErrorPolicyを返す。
    リアルタイムモードでは処理継続を最優先とし、未知のエラーに対しても
    SKIPポリシーを返して処理を継続する。

    仕様書§8.5の分岐ロジック:
        - ``FaceNotDetectedError`` → SKIP
        - ``torch.cuda.OutOfMemoryError`` → RETRY（empty_cache実行後）
        - ``ModelLoadError`` / ``ConfigError`` → ABORT
        - その他の例外 → SKIP（リアルタイムでは継続優先）
    """

    def handle(self, error: Exception, context: dict[str, Any]) -> ErrorPolicy:
        """例外の種類に応じたErrorPolicyを返す。

        GPU OOMの場合はtorch.cuda.empty_cache()を実行してからRETRYを返す。
        全てのエラーはloguruで適切なレベルのログに記録される。

        Args:
            error: 発生した例外インスタンス。
            context: エラーの文脈情報を格納する辞書。
                例: ``{"frame_index": 42, "module": "extractor"}``。
                ログ出力に使用される。

        Returns:
            エラーに対する処理ポリシー。

        Example:
            ::

                handler = PipelineErrorHandler()
                policy = handler.handle(
                    FaceNotDetectedError("no face in frame"),
                    {"frame_index": 100},
                )
                assert policy == ErrorPolicy.SKIP
        """
        if isinstance(error, FaceNotDetectedError):
            logger.warning(
                "Face not detected, skipping frame. context={}", context
            )
            return ErrorPolicy.SKIP

        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            logger.error(
                "GPU OOM detected, cache cleared, will retry. context={}",
                context,
            )
            return ErrorPolicy.RETRY

        if isinstance(error, (ModelLoadError, ConfigError)):
            logger.critical(
                "Fatal error: {}, aborting pipeline. context={}",
                error,
                context,
            )
            return ErrorPolicy.ABORT

        logger.error("Unexpected error: {}, skipping frame. context={}", error, context)
        return ErrorPolicy.SKIP
