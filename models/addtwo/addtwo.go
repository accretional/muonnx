// Package addtwo is the add_two test model (Y = A + B) — a genuinely multi-input
// graph used to exercise model.Session. Regenerate weights with
// `go run ./tools/genmodels`.
package addtwo

import (
	"embed"

	muonnx "github.com/accretional/muonnx/src/muonnx"
)

//go:embed onnx
var weights embed.FS

// Load loads the add_two weights (no deps). Pass it to model.NewSession.
var Load = muonnx.Load(weights, "add_two")
