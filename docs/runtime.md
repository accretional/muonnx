# muonnx runtime

A torch-free ONNX inference runtime + weight server for Go. The spine is

```
main → server.InferenceServer → service.Service → model.Model → onnx dep (weights)
```

`main` starts a server with the gRPC services it wants to host; the server builds
each service's dependency graph (models → weights, each loaded once), warms every
model-backed service, and only then serves.

## Packages

| Path | Role |
|---|---|
| `src/muonnx` | core: environment, weight registry + `Load`, build graph, catalog, ORT lifecycle |
| `src/muonnx/builder` | builds resolved weights into a live ORT session (auto-discovers I/O) |
| `src/muonnx/model` | `Model` build-graph node: Build (resolve→build session→catalog), Run |
| `src/muonnx/service` | `Service` (Node + gRPC `Register`) and `Warmer` interfaces |
| `src/muonnx/server` | `InferenceServer`: build → register → warm → serve |
| `proto/` | `ONNXRuntimeConfig`, `ONNXEnvironmentConfig`, `ONNXWeightSource`, weight server, generic `Inference` |
| `models/<pkg>` | a model package: embeds its weights + `var Load = muonnx.Load(...)` |
| `tools/genmodels` | authors the test ONNX models in pure Go from the ONNX protobuf schema |
| `cmd/eval` | acceptance test: random input → all 1s through the server (mult_0 → add_1) |

## The load lifecycle

```
init() Register ─▶ Load ─▶ model.Build ─▶ service.Warm ─▶ all ready ─▶ Serve
       (callback)  (acquire,  (live ORT     (warming inv.)  (barrier)
                    dedup'd)    session)
```

- **Register / Load** (`weight_registry.go`). A model package declares its weights
  with one line — `var Load = muonnx.Load(weights, "name", dep1.Load, …)`. `Load`
  returns a `LoadFunc` bound to a shared `Dep`; calling it loads sub-deps then
  resolves its own weights once, shared across all holders of that name.
- **Resolve** (precision-first, then location). Layout, identical in the central
  `/onnx` weight source and a package's embed:
  - fp32 → `<name>.onnx` (`/onnx/name.onnx`, embed `onnx/name.onnx`)
  - fp16 → `fp16/<name>.onnx` (`/onnx/fp16/name.onnx`, embed `onnx/fp16/name.onnx`)
  `/onnx` overrides the embed for a given precision; if a quant isn't embedded,
  `/onnx` is the necessary source. Serving precision is independent of inference
  (`ResolveServing`) — a WebGPU client can fetch an fp16 we embed but don't run.
- **Build** (`model.Build` via `builder.Build`). Turns the resolved file into a
  live `ort.DynamicAdvancedSession`, auto-discovering input/output names. (This is
  "building the model at runtime" — there is no `OpenSession`.)
- **Warm / ready** (`service.Warmer`, `server.Warm`). A service backing RPCs with
  a model gives it a throwaway inference; the server warms all such services
  concurrently and waits before serving. That wait is readiness.

## Config

`ONNXRuntimeConfig` (per server): `weight_source.path` (the `/onnx` dir; default
`/onnx`), `inference_precision`, `serving_precision`, model catalog. Apply it with
`server.New(...).Config(cfg)` — the server reads it once, provisioned at the same
scope as `/onnx`. `ONNXEnvironmentConfig` holds host knobs (ORT lib, threads, EP
preference).

## Run the acceptance test

```sh
go run ./tools/genmodels                 # author models/{mult0,add1}/onnx/*.onnx
ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib go run ./cmd/eval
# -> PASS: random input -> all 1s through muonnx server (mult_0 -> add_1)
```

`muonnx.Init("")` also discovers the ORT lib via `ONNXRUNTIME_LIB`, system dirs,
or `./third_party/onnxruntime-*`.

## Self-contained (fat) binary

`muonnx.Init` resolves the ORT shared library: explicit path > on-disk discovery
(`ONNXRUNTIME_LIB`, system dirs, `./third_party`) > **embedded dylib**. Build the
self-contained variant with:

```sh
ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib ./scripts/prep_embed.sh  # stage dylib + models
go build -tags muonnx_fat ./cmd/eval                                    # embeds it (~+32 MB)
./eval                                                                  # runs with no ORT on disk
```

The default (slim) build embeds nothing and discovers the lib at runtime. The
`muonnx_fat` tag pulls in `embed_fat_{darwin,linux}.go` (per-platform `//go:embed`
of `src/muonnx/ort/lib...`); `Init` materializes it to a temp file when nothing is
found on disk, and `Shutdown` removes it. Large *models* follow the same idea: too
big to embed → ship in the `/onnx` weight source, where `Resolve` finds them.

## Weight server

`weightserver.New()` is a Service implementing `ONNXRuntime`: `ListModels` streams
the catalog (self-registered by models at Build), and `Fetch` streams a model's
weights at the requested precision (UNSPECIFIED → serving precision), resolved
wherever the variant lives — so a WebGPU client can fetch an fp16 this host embeds
but doesn't run. Compose it on the same server: `server.New(infSvc, weightserver.New())`.
