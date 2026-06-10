//go:build muonnx_fat && darwin

package muonnx

import _ "embed"

// embed_fat_darwin.go embeds the macOS ORT dylib into the binary. Requires
// scripts/prep_embed.sh to have staged src/muonnx/ort/libonnxruntime.dylib first.
//
//go:embed ort/libonnxruntime.dylib
var ortLib []byte
