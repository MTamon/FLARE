"""FLARE: Facial Landmark Analysis & Rendering Engine.

LHG研究における特徴量抽出（エンコード）とフォトリアルレンダリング（デコード）の
統合ツール。BFMベース（Route A）とFLAMEベース（Route B）の2つのパイプラインを
サポートし、リアルタイムモードと前処理（バッチ）モードの両方に対応する。

Modules:
    config: YAML設定ファイルの読み込みとpydanticバリデーション。
    extractors: 3DMMパラメータ抽出の基底クラスと実装。
    renderers: フォトリアルレンダリングの基底クラスと実装。
    model_interface: LHGモデルインターフェースの基底クラス。
    converters: パラメータ形式変換（DECA→FLAME等）。
    pipeline: リアルタイム・バッチ処理パイプライン。
    utils: 顔検出、動画I/O、ロギング等のユーティリティ。
"""

from __future__ import annotations

__version__ = "2.2.0"
"""str: FLAREパッケージのバージョン文字列。仕様書v2.2に対応。"""
