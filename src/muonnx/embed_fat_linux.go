//go:build muonnx_fat && linux

package muonnx

import _ "embed"

// embed_fat_linux.go embeds the Linux ORT shared object into the binary. Requires
// scripts/prep_embed.sh to have staged src/muonnx/ort/libonnxruntime.so first.
//
//go:embed ort/libonnxruntime.so
var ortLib []byte
