//go:build !muonnx_fat

package muonnx

// embed_slim.go is the default build: no embedded ORT dylib. Init discovers the
// shared library on disk (ONNXRUNTIME_LIB / system dirs / ./third_party). Build
// with -tags muonnx_fat (after scripts/prep_embed.sh stages the dylib into
// src/muonnx/ort/) to embed it for a fully self-contained binary.
var ortLib []byte
