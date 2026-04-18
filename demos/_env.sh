#!/usr/bin/env bash
# demos/_env.sh — FLARE デモ共通環境設定
#
# run_demo*.sh から source される。エクスポートした変数はラッパーのサブシェル内
# にのみ存在し、ユーザのインタラクティブシェルには影響を与えない。
#
# --- なぜ EGL ベンダーを固定するか ---
#
# MediaPipe の FaceLandmarker (TFLite GPU delegate) は検出器構築時に EGL
# コンテキストを作成する。Linux では EGL/GLX の dispatch は libglvnd が担当し、
# /usr/share/glvnd/egl_vendor.d/ 内の JSON を参照してベンダードライバを選ぶ。
# デフォルトでは MESA が優先されるが、ML マシンに MESA DRI ドライバがない場合
#
#   libEGL warning: MESA-LOADER: failed to open radeonsi / swrast
#   GPU support is not available: Unable to initialize EGL
#
# というエラーが出て MediaPipe は CPU XNNPACK に暗黙フォールバックする。
# Webcam デモが ~100 FPS でなく ~20 FPS になるのはこれが原因。
#
# __EGL_VENDOR_LIBRARY_FILENAMES で NVIDIA JSON を直接指定することで
# MESA プローブを完全にスキップし GPU delegate を確実に開く。
# システム全体の設定は変更不要 — ラッパーが存在する間だけ有効。
#
# --- WSL2 での注意 ---
#
# WSL2 上では NVIDIA EGL ライブラリが利用できないため GPU delegate は
# 機能しない。各デモスクリプトが起動時に警告を表示し、CPU に自動フォールバック
# する。

_NVIDIA_EGL_JSON_CANDIDATES=(
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    "/etc/glvnd/egl_vendor.d/10_nvidia.json"
    "/usr/local/share/glvnd/egl_vendor.d/10_nvidia.json"
)

_nv_json=""
for _c in "${_NVIDIA_EGL_JSON_CANDIDATES[@]}"; do
    if [ -f "$_c" ]; then
        _nv_json="$_c"
        break
    fi
done

if [ -n "$_nv_json" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$_nv_json"
    export __GLX_VENDOR_LIBRARY_NAME="nvidia"
    echo "[demos/_env.sh] EGL vendor を NVIDIA に固定: $_nv_json"
else
    echo "[demos/_env.sh] WARN: NVIDIA EGL vendor JSON が見つかりません:" >&2
    for _c in "${_NVIDIA_EGL_JSON_CANDIDATES[@]}"; do
        echo "                - $_c" >&2
    done
    echo "[demos/_env.sh] WARN: MediaPipe GPU delegate は CPU にフォールバックします。" >&2
fi

# Python バイトコードキャッシュを抑制 (短命なデモ実行向け)
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# PYTHONPATH に FLARE ルートを追加
_HERE_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT_ABS="$(cd "$_HERE_ABS/.." && pwd)"
export PYTHONPATH="${_ROOT_ABS}${PYTHONPATH:+:$PYTHONPATH}"

unset _c _nv_json _NVIDIA_EGL_JSON_CANDIDATES _HERE_ABS _ROOT_ABS
