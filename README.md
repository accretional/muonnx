# muonnx

Golang AI Inference, Training, Fine-Tuning, and Weight Server for ONNX machine learning models — optimized for CPU inference on Intel Xeon amd64, CPU/NPU Apple Silicon arm64, and webGPU via transformers.js.

## Spine

```
main → server.InferenceServer → service.Service → model.Model → onnx dep (weights)
```

`main` starts a server with its gRPC services; the server builds each service's
models (loading weights once, deps-first), warms them, then serves. Models declare
their weights with one line and resolve them from a central `/onnx` dir or their
own `go:embed`. See [docs/runtime.md](docs/runtime.md).

## Quickstart

```sh
go run ./tools/genmodels                                   # author the test models (pure Go)
ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib go run ./cmd/eval
# -> PASS: random input -> all 1s through muonnx server (mult_0 -> add_1)
```

A model package is just:

```go
package mult0

import (
    "embed"
    muonnx "github.com/accretional/muonnx/src/muonnx"
)

//go:embed onnx
var weights embed.FS

var Load = muonnx.Load(weights, "mult_0")
```

## Layout

- `src/muonnx` — runtime core (environment, weight registry + `Load`, build graph, catalog, ORT lifecycle) + `builder`/`model`/`service`/`server`.
- `proto/` — `ONNXRuntimeConfig`, `ONNXEnvironmentConfig`, `ONNXWeightSource`, the weight server, and a generic `Inference` service.
- `models/` — model packages (each embeds its weights).
- `tools/` — the ONNX schema + the pure-Go model generator.
- `cmd/eval` — end-to-end acceptance test.

Requires Go 1.26 and an ONNX Runtime shared library (binding pinned to v1.22.0 / ORT 1.22).
