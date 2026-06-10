// Package add1 is the add_1 test model (Y = X + 1). See package mult0 for the
// Load convention; regenerate weights with `go run ./tools/genmodels`.
package add1

import (
	"embed"

	muonnx "github.com/accretional/muonnx/src/muonnx"
)

//go:embed onnx
var weights embed.FS

// Load loads the add_1 weights (no deps). Pass it to model.New.
var Load = muonnx.Load(weights, "add_1")
