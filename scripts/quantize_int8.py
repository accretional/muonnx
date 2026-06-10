#!/usr/bin/env python3
"""quantize_int8.py — dynamic int8 quantization of an ONNX model.

Produces a U8S8 (uint8 activations x int8 weights) graph: the combination Intel's
AVX-512 VNNI (VPDPBUSD) accelerates, and which onnxruntime's CPU EP (MLAS) and the
OpenVINO EP both run with VNNI on capable Intel hosts. Dynamic quantization needs
no calibration data — weights are quantized at conversion, activations at runtime —
so it's the cheap, lossless-to-set-up path; it helps most on large matmul-bound
graphs (e.g. a transformer encoder) and can be neutral/negative on many tiny
autoregressive matmuls (a KV-cache decoder), so quantize per-part and measure.

    quantize_int8.py <input.onnx> <output.onnx> [--no-per-channel]

Weights ~4x smaller (also shrinks cold-start read). I/O tensor names/shapes are
unchanged, so a quantized part is a drop-in for its fp32 file.
"""
import argparse
import os
import sys

from onnxruntime.quantization import quantize_dynamic, QuantType


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--no-per-channel", action="store_true",
                    help="disable per-channel weight quant (per-channel is more accurate)")
    ap.add_argument("--ops", default="MatMul",
                    help="comma-separated op types to quantize (default MatMul: the "
                         "attention/FFN linears — the dominant cost + VNNI target). We "
                         "deliberately EXCLUDE Conv: dynamic quant emits ConvInteger, which "
                         "the CPU EP doesn't implement, and the conv stem is tiny anyway.")
    args = ap.parse_args()

    sz = os.path.getsize(args.input) / 1e6
    ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    print(f"quantize_dynamic: {args.input} ({sz:.0f} MB) -> {args.output}  ops={ops}")
    quantize_dynamic(
        args.input,
        args.output,
        weight_type=QuantType.QInt8,           # int8 weights; activations -> uint8 (U8S8 -> VNNI)
        per_channel=not args.no_per_channel,
        op_types_to_quantize=ops,
    )
    outsz = os.path.getsize(args.output) / 1e6
    print(f"done: {outsz:.0f} MB ({sz / outsz:.1f}x smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
