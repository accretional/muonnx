#!/usr/bin/env python3
"""quantize_static_qdq.py — static, calibrated int8 quantization in QDQ format.

Why QDQ + static (vs the dynamic QOperator path in quantize_int8.py):
  * QDQ (QuantizeLinear/DequantizeLinear pairs) is the format the onnxruntime
    OpenVINO EP actually consumes — the QOperator/MatMulInteger graph from
    dynamic quant makes the OV EP throw "Output names mismatch between OpenVINO
    and ONNX". QDQ fixes that.
  * Static calibration bakes activation scales in, removing the per-call
    DynamicQuantizeLinear overhead that blunts dynamic int8 on CPU.

Built for the whisper encoder: its input is `input_features` [1,80,3000] (a
30 s log-mel window), so calibration = real audio -> whisper log-mel -> run the
fp32 graph to record activation ranges. U8S8 (QUInt8 activations x QInt8 weights)
is the AVX-512 VNNI combination; MatMul-only (Conv would become unsupported ops).

    quantize_static_qdq.py --input enc.onnx --output enc.qdq.onnx \
        --calib-dir DIR [--calib-dir DIR2 ...] [--max-samples 64]

NOTE: calibration audio is supplied via --calib-dir at runtime ONLY; this script
hardcodes no paths and prints no filenames (counts only).
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import librosa
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, CalibrationMethod, QuantType, QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
N_SAMPLES = 30 * SR     # 30 s window
N_FRAMES = 3000
_MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)


def log_mel(path):
    """Whisper log-mel features [1, 80, 3000] from an audio file (pad/trim 30 s)."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, window="hann", center=True)
    power = np.abs(stft[..., :-1]) ** 2
    mel = _MEL @ power
    logspec = np.log10(np.maximum(mel, 1e-10))
    logspec = np.maximum(logspec, logspec.max() - 8.0)
    logspec = (logspec + 4.0) / 4.0
    return logspec[np.newaxis, :, :N_FRAMES].astype(np.float32)


def gather(dirs, max_samples):
    """Deterministic strided sample of .wav files across the given dirs."""
    files = []
    for d in dirs:
        for root, _, names in os.walk(d):
            files += [os.path.join(root, n) for n in names if n.lower().endswith(".wav")]
    files.sort()
    if not files:
        sys.exit("no .wav files found in --calib-dir(s)")
    if len(files) > max_samples:
        step = len(files) / max_samples
        files = [files[int(i * step)] for i in range(max_samples)]
    return files


class Reader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.files, self.input_name, self.i = files, input_name, 0

    def get_next(self):
        while self.i < len(self.files):
            path = self.files[self.i]; self.i += 1
            try:
                return {self.input_name: log_mel(path)}
            except Exception:
                continue  # skip unreadable clips; no filename printed
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib-dir", action="append", required=True, dest="calib_dirs")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--input-name", default="input_features")
    args = ap.parse_args()

    files = gather(args.calib_dirs, args.max_samples)
    print(f"calibration: {len(files)} clips from {len(args.calib_dirs)} dir(s)")

    with tempfile.TemporaryDirectory() as td:
        pre = os.path.join(td, "pre.onnx")
        print("pre-processing (shape inference + optimize)…")
        quant_pre_process(args.input, pre, skip_symbolic_shape=False)
        sz = os.path.getsize(args.input) / 1e6
        print(f"quantize_static QDQ U8S8 MatMul-only: {sz:.0f} MB -> {args.output}")
        quantize_static(
            pre, args.output, Reader(files, args.input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul"],
            calibrate_method=CalibrationMethod.MinMax,
        )
    out = os.path.getsize(args.output) / 1e6
    print(f"done: {out:.0f} MB ({sz / out:.1f}x smaller)")


if __name__ == "__main__":
    main()
