# muonnx

Torch-free ONNX inference runtime + weight server for Go. muonnx loads ONNX
models, builds them into live [ONNX Runtime](https://onnxruntime.ai) sessions,
and serves them over gRPC — with a small, opinionated structure for how models
declare their weights, get placed on execution providers (CPU / CoreML), and are
brought up. Targets CPU inference on Intel/AMD64, CPU+GPU on Apple Silicon, and
serving weights to WebGPU clients (transformers.js).

No Python at runtime. No cgo beyond the ONNX Runtime binding. Models are ordinary
Go packages.

## The spine

```
main → server.InferenceServer → service.Service → model.{Model,Session,Multipart} → weights
```

`main` constructs a server with the gRPC services it wants to host. The server
builds each service's models (resolving + opening their ONNX graphs, deps-first,
in parallel), **warms** them, and only then serves. A service *owns* its models; a
model resolves its weights through muonnx and builds them into ORT sessions.

## Quickstart

```sh
go run ./tools/genmodels                                    # author the test models (pure Go)
ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib go run ./cmd/eval
# PASS: random input -> all 1s through muonnx server (mult_0 -> add_1)
# PASS: multi-input model.Session (add_two)
# PASS: weight server ListModels + streaming Fetch
```

`cmd/eval` is the end-to-end acceptance test and the best worked example: it
builds an `InferenceServer`, hosts a service composing two models, drives a real
gRPC round-trip, exercises `model.Session`, and pulls weights from the weight
server.

Requires **Go 1.26** and an ONNX Runtime shared library. The binding is pinned to
`github.com/yalue/onnxruntime_go v1.22.0` (ORT 1.22 C API); `muonnx.Init("")`
discovers the library via `$ONNXRUNTIME_LIB`, common system dirs, or
`./third_party`, or you can embed it (`-tags muonnx_fat`).

## Concepts

### Weights: the `Load` convention

A model package embeds its small graphs and/or resolves big ones from a central
directory, with one line:

```go
package mult0

import (
    "embed"
    muonnx "github.com/accretional/muonnx/src/muonnx"
)

//go:embed onnx
var weights embed.FS

// Resolves onnx/mult_0.onnx from this embed, or the /onnx override dir.
var Load = muonnx.Load(weights, "mult_0")
```

`muonnx.Load(embedFS, name, deps...)` returns a shared, dedup'd `LoadFunc`.
Resolution is **precision-first, then location**:

- **Precision** — fp32 at `<name>.onnx`, fp16 at `fp16/<name>.onnx`. The
  environment's inference precision picks the order (fp32 default — onnxruntime's
  CPU EP has no native fp16 kernels, so fp32 is faster on CPU; fp16 wins on
  GPU/WebGPU and is the default *serving* precision).
- **Location** — the central weight source (`ONNXWeightSource.path`, default
  `/onnx`) overrides a package's embed. A model too big to embed simply ships in
  the weight source; a small one travels inside the binary. `go:embed` and `/onnx`
  use the same layout, so the same model resolves either way.

### The build graph

Everything that loads is a `Node`:

```go
type Node interface { Deps() []Node; Build() error }
func Build(roots ...Node) (*Built, error)   // deps-first, once each; *Built.Close() releases all
```

`Build` walks the graph (a service's models are its deps), builds each node once,
and returns a `*Built` whose `Close()` releases every loaded resource in reverse
order. Catalog registration and lifecycle are handled here, not in `main`.

### Three model tiers

Pick the smallest one that fits your graph:

| | use it for | run API |
|---|---|---|
| `model.Model` | one graph, single float32 in/out | `Run(data, shape) → (data, shape)` |
| `model.Session` | one graph, **multiple named I/O, mixed dtypes** | `Run(map[string]Tensor) → map[string]Tensor` |
| `model.Multipart` | **several graphs as one logical model** | `Session(part)` → raw ORT for hot loops |

- **`Session`** auto-discovers I/O at build and handles ORT tensor create/destroy
  + output-dtype detection for you (`model.F32/I64/I32` builders). Reach for it
  whenever a single model has more than one input/output or non-float tensors.
- **`Multipart`** bundles conjoined graphs — e.g. an encoder + a KV-cache decoder
  — built and closed as a unit, each with its **own execution provider** (a
  static encoder on CoreML while the dynamic decoder stays on CPU is a one-liner).
  Parts load **concurrently** (cold start ≈ the slowest part, not the sum). It
  hands back the live sessions; you drive the model-specific inference loop.

### Services, warm, and the server

```go
type Service interface { Node; Register(grpc.ServiceRegistrar) }
type Warmer  interface { Warm(context.Context) error }   // optional
```

A `Service` is a `Node` that registers a gRPC handler. If it implements `Warmer`,
the server runs a throwaway inference before opening the port — so the first real
request is never cold. `server.InferenceServer` ties it together:

```go
srv := server.New(mysvc).
    Config(cfg).                                   // ONNXRuntimeConfig (precision, /onnx path)
    Options(grpc.MaxRecvMsgSize(64 << 20))
defer srv.Stop()
srv.Serve(lis)                                     // build → warm (readiness gate) → serve
```

### Weight server

`weightserver.New()` is a ready-made `Service` implementing the `ONNXRuntime`
gRPC API: `ListModels` streams the catalog (models self-register at build), and
`Fetch` streams a model's bytes at a requested precision — so a browser/WebGPU
client can pull the fp16 variant the server itself doesn't run.

### Execution providers (CoreML)

`SetEnvironmentConfig` sets the EP preference; `Multipart.Part.Providers` and
`model.NewSession(load, providers...)` set it per model. CoreML is appended on
darwin with `RequireStaticInputShapes=1` (its MLProgram backend aborts on dynamic
dims, so dynamic nodes fall back to CPU instead of crashing) and a **durable,
on-volume compile cache** with a pinned `$TMPDIR` (ORT 1.22 keys CoreML cache
reuse on `$TMPDIR`; pinning makes a fresh process reuse the compiled model instead
of recompiling — see `cacheBase()` / `$MUONNX_CACHE`).

Measured on a whisper-small encoder (static fp16, Apple Silicon), per
`MLComputeUnits`:

| backend | encoder/run | vs ORT CPU |
|---|---|---|
| ORT CPU EP | 1320 ms | 1.0× |
| CoreML `CPUOnly` | 738 ms | 1.8× |
| CoreML `CPUAndNeuralEngine` (ANE) | 703 ms | 1.9× |
| CoreML `CPUAndGPU` | 526 ms | 2.5× |
| CoreML `ALL` (default) | 491 ms | 2.7× |

Takeaway: for transformer encoders the win is **CoreML's MLProgram graph
optimization + the GPU**, not the ANE/NPU (which barely moves it — the ANE doesn't
take attention-heavy graphs well). `ALL` is the right default. Override with
`$MUONNX_COREML_UNITS`.

## Writing a model + service

A model package: the `Load` one-liner above, then `model.New(Load)` /
`model.NewSession(Load)` / `model.NewMultipart(name, parts)` in your service.

A service:

```go
type mysvc struct {
    pb.UnimplementedFooServer
    m *model.Session
}
func New() *mysvc                                   { return &mysvc{m: model.NewSession(foo.Load)} }
func (s *mysvc) Deps() []muonnx.Node               { return []muonnx.Node{s.m} } // server builds it
func (s *mysvc) Build() error                      { return nil }
func (s *mysvc) Register(r grpc.ServiceRegistrar)  { pb.RegisterFooServer(r, s) }
func (s *mysvc) Warm(context.Context) error        { _, err := s.m.Run(sample); return err }
func (s *mysvc) Foo(ctx, req) (*pb.Resp, error)    { out, _ := s.m.Run(...); return ... }
```

## Configuration (`proto/`)

- **`ONNXRuntimeConfig`** — `weight_source` (`/onnx` path), `inference_precision`,
  `serving_precision`, model catalog. Applied via `server.Config` / `SetConfig`.
- **`ONNXEnvironmentConfig`** — host knobs: `execution_providers`, intra/inter-op
  threads. Applied via `SetEnvironmentConfig`.
- **`ONNXRuntime` / `Inference`** — the weight-server and a generic tensor
  inference gRPC surface.

## Layout

```
src/muonnx            core: environment, weight registry + Load, build graph, catalog, ORT + EP/session options
src/muonnx/builder    resolved path → live ORT session (auto-discovers I/O)
src/muonnx/model      Model · Session · Multipart
src/muonnx/service    Service + Warmer interfaces
src/muonnx/server     InferenceServer (build → warm → serve)
src/muonnx/weightserver  ONNXRuntime weight service (ListModels, streaming Fetch)
proto/                config + gRPC protos (muonnxpb)
models/               example/test model packages (mult0, add1, addtwo)
tools/                ONNX protobuf schema + pure-Go model generator
cmd/eval              end-to-end acceptance test / worked example
docs/runtime.md       deeper design notes
```

## Self-contained binary

`scripts/prep_embed.sh` stages the ORT dylib; `go build -tags muonnx_fat` embeds
it so the binary runs with no ONNX Runtime on disk.

## Status

Build/vet/gofmt clean; `cmd/eval` green. Used in
[accretional/muonnx-demo](https://github.com/accretional/muonnx-demo) (VAD +
Whisper transcription). Training / fine-tuning are roadmap, not yet implemented.
