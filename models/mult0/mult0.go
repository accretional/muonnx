// Package mult0 is the mult_0 test model (Y = X * 0). It embeds its weights and
// declares them to the runtime with the one-line Load convention — no per-package
// Load boilerplate. Regenerate the weights with `go run ./tools/genmodels`.
package mult0

import (
	"embed"

	muonnx "github.com/accretional/muonnx/src/muonnx"
)

//go:embed onnx
var weights embed.FS

// Load loads the mult_0 weights (no deps). Pass it to model.New.
var Load = muonnx.Load(weights, "mult_0")
