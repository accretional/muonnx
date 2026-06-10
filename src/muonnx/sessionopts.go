package muonnx

import (
	"log"
	"os"
	"path/filepath"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

// sessionopts.go translates the environment's execution-provider + threading
// policy into ORT session options, the single place EP selection happens. Both
// builder.Build and any package driving ORT sessions directly (e.g. a multi-input
// model) call SessionOptions so every session in the process honors the same
// policy.

// SessionOptions builds ORT session options from the environment's
// execution-provider policy. See SessionOptionsFor.
func SessionOptions() (*ort.SessionOptions, error) {
	return SessionOptionsFor(Environment().ExecutionProviders...)
}

// SessionOptionsFor builds ORT session options for an explicit execution-provider
// preference, ignoring the environment's EP policy but still applying its
// threading settings. Use it to place individual sessions on different providers
// — e.g. a static-shape encoder on "coreml" while a dynamic decoder stays on
// "cpu". Returns nil for the plain-CPU default (nil = ORT defaults; CPU is always
// the implicit fallback). The caller owns the returned options and must Destroy
// them after creating its session(s).
//
// CoreML is appended when "coreml" is requested and the host is darwin;
// unsupported nodes fall back to CPU within the session, and non-darwin coreml
// requests are skipped. Note (measured on whisper-small's encoder): the speedup
// is mostly CoreML's MLProgram graph optimization + the GPU — the ANE/NPU
// contributes little for transformer encoders (CPUAndNeuralEngine ≈ CPUOnly,
// well behind CPUAndGPU). Default ALL lets CoreML use the GPU.
//
// OpenVINO is appended when "openvino" is requested and the host is linux. It
// targets Intel CPU/GPU/NPU (x86) and uses AVX-512/VNNI kernels — the lever for
// int8/VNNI inference on Intel servers (e.g. Cloud Run). It is BEST-EFFORT: the
// stock onnxruntime release lacks the OpenVINO EP, so AppendExecutionProviderOpenVINO
// fails unless the loaded libonnxruntime was built with OpenVINO (Intel's
// onnxruntime-openvino) and the OpenVINO runtime libs are present; on failure we
// log and fall back to CPU (OV is a preference, CPU the implicit fallback). Tune
// via MUONNX_OPENVINO_DEVICE (default CPU), MUONNX_OPENVINO_PRECISION (e.g. FP32/
// FP16/ACCURACY; omitted → OV default), and the intra-op thread count.
func SessionOptionsFor(providers ...string) (*ort.SessionOptions, error) {
	e := Environment()
	coreml := e.OS == "darwin" && hasProvider(providers, "coreml")
	openvino := e.OS == "linux" && hasProvider(providers, "openvino")
	cuda := e.OS == "linux" && hasProvider(providers, "cuda")
	gLevel, gSet := graphOptLevel()
	if !coreml && !openvino && !cuda && !gSet && e.IntraOpThreads == 0 && e.InterOpThreads == 0 {
		return nil, nil // nothing to configure: use ORT's CPU defaults
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if e.IntraOpThreads > 0 {
		if err := opts.SetIntraOpNumThreads(e.IntraOpThreads); err != nil {
			opts.Destroy()
			return nil, err
		}
	}
	if e.InterOpThreads > 0 {
		if err := opts.SetInterOpNumThreads(e.InterOpThreads); err != nil {
			opts.Destroy()
			return nil, err
		}
	}
	if gSet {
		// Lower graph-optimization levels cut session-creation time at a possible
		// small steady-state cost — the lever for cold start (e.g. the t3-llm TTS
		// model's ~90 s fp16 graph-opt replay). Default (unset) keeps ORT's
		// ENABLE_ALL. Set via MUONNX_GRAPH_OPT=disable|basic|extended|all.
		if err := opts.SetGraphOptimizationLevel(gLevel); err != nil {
			opts.Destroy()
			return nil, err
		}
	}
	if coreml {
		// MLProgram is the modern CoreML format; ALL lets CoreML place ops on
		// the ANE/GPU/CPU as it sees fit. RequireStaticInputShapes=1 is essential:
		// CoreML's MLProgram backend aborts (hard native SIGABRT) on unbounded/
		// dynamic dimensions, so we tell the EP to only take static-shape subgraphs
		// and leave dynamic ones on CPU. Models exported with fixed shapes get the
		// most acceleration; fully-dynamic graphs (e.g. Whisper as exported) fall
		// back to CPU rather than crashing.
		//
		// ModelCacheDirectory persists the compiled .mlmodelc across processes so
		// only the first run pays the compile. CRUCIAL: ORT 1.22's CoreML cache
		// reuse keys on $TMPDIR consistency — if $TMPDIR differs between launches
		// (it varies by login context, and is unset in many shells) the cache
		// misses and recompiles every time. So we pin a stable $TMPDIR co-located
		// with the cache. Both live under cacheBase() — on the weights volume, not
		// $HOME — and are treated like derived weight artifacts.
		// MLComputeUnits selects which silicon CoreML may use: ALL (ANE+GPU+CPU,
		// the default — CoreML picks per op), CPUAndNeuralEngine (ANE/NPU),
		// CPUAndGPU, or CPUOnly. Override via MUONNX_COREML_UNITS to pin/measure.
		units := os.Getenv("MUONNX_COREML_UNITS")
		if units == "" {
			units = "ALL"
		}
		coremlOpts := map[string]string{
			"ModelFormat":              "MLProgram",
			"MLComputeUnits":           units,
			"RequireStaticInputShapes": "1",
		}
		base := cacheBase()
		if td := ensureDir(filepath.Join(base, "tmp")); td != "" {
			// NOTE: this redirects the WHOLE process's $TMPDIR (every later
			// os.TempDir / os.CreateTemp, including other libraries). It is
			// deliberate and required: ORT 1.22's CoreML cache reuse keys on
			// $TMPDIR, so a stable value is the only way to make the cross-process
			// compile cache hit. We always set it (rather than only-when-unset) so
			// reuse is reliable regardless of the launching shell's $TMPDIR, which
			// varies by login context. Scope it with $MUONNX_CACHE if undesired.
			os.Setenv("TMPDIR", td)
		}
		if dir := ensureDir(filepath.Join(base, "coreml")); dir != "" {
			coremlOpts["ModelCacheDirectory"] = dir
		}
		if err := opts.AppendExecutionProviderCoreMLV2(coremlOpts); err != nil {
			opts.Destroy()
			return nil, err
		}
	}
	if openvino {
		// Best-effort: a stock libonnxruntime has no OpenVINO EP, so this errors
		// unless an OpenVINO-enabled build is loaded. Don't fail the load — log and
		// leave the session on CPU (OV is a preference; CPU is the fallback).
		if err := opts.AppendExecutionProviderOpenVINO(openVINOOptions(e)); err != nil {
			log.Printf("muonnx: OpenVINO EP unavailable (%v); using CPU. "+
				"Needs an onnxruntime built with OpenVINO + the OpenVINO runtime libs.", err)
		}
	}
	if cuda {
		// CUDA (NVIDIA) EP. Best-effort: a stock CPU-only libonnxruntime lacks it,
		// so this errors unless a CUDA-enabled ORT build + the CUDA/cuDNN runtime are
		// present (the GPU image); on failure we log and fall back to CPU. Unlike
		// CoreML it handles dynamic shapes, so it's the target for the TTS
		// transformers (t3-llm AR loop, s3gen) and fp16 inference on GPU.
		if co, cerr := ort.NewCUDAProviderOptions(); cerr != nil {
			log.Printf("muonnx: CUDA provider options unavailable (%v); using CPU.", cerr)
		} else {
			if m := cudaOptions(); len(m) > 0 {
				_ = co.Update(m)
			}
			if err := opts.AppendExecutionProviderCUDA(co); err != nil {
				log.Printf("muonnx: CUDA EP unavailable (%v); using CPU. "+
					"Needs an onnxruntime built with CUDA + the CUDA/cuDNN runtime libs.", err)
			}
			co.Destroy()
		}
	}
	return opts, nil
}

// cudaOptions builds the CUDA EP provider-option map. device_id selects the GPU
// (default 0; MUONNX_CUDA_DEVICE); gpu_mem_limit caps the arena in bytes
// (MUONNX_CUDA_MEM_LIMIT — matters on 8–16 GB cards); cudnn_conv_algo_search
// (EXHAUSTIVE|HEURISTIC|DEFAULT via MUONNX_CUDA_CONV_ALGO) trades first-run
// autotune for steady-state speed.
func cudaOptions() map[string]string {
	o := map[string]string{}
	dev := os.Getenv("MUONNX_CUDA_DEVICE")
	if dev == "" {
		dev = "0"
	}
	o["device_id"] = dev
	if v := os.Getenv("MUONNX_CUDA_MEM_LIMIT"); v != "" {
		o["gpu_mem_limit"] = v
	}
	if v := os.Getenv("MUONNX_CUDA_CONV_ALGO"); v != "" {
		o["cudnn_conv_algo_search"] = v
	}
	return o
}

// openVINOOptions builds the OpenVINO EP provider-option map. device_type selects
// the Intel target (CPU/GPU/NPU/AUTO; default CPU); precision is optional (OV
// picks ACCURACY/FP32 if unset); num_of_threads reuses the intra-op thread policy
// so one knob (MUONNX_INTRA_OP_THREADS) bounds both EPs; cache_dir persists OV's
// compiled blobs across loads (like the CoreML model cache), co-located with the
// other derived artifacts under cacheBase().
func openVINOOptions(e ONNXRuntimeEnvironment) map[string]string {
	o := map[string]string{}
	dev := os.Getenv("MUONNX_OPENVINO_DEVICE")
	if dev == "" {
		dev = "CPU"
	}
	o["device_type"] = dev
	if p := os.Getenv("MUONNX_OPENVINO_PRECISION"); p != "" {
		o["precision"] = p
	}
	if e.IntraOpThreads > 0 {
		o["num_of_threads"] = strconv.Itoa(e.IntraOpThreads)
	}
	if dir := ensureDir(filepath.Join(cacheBase(), "openvino")); dir != "" {
		o["cache_dir"] = dir
	}
	return o
}

// CoreMLCacheDir is where compiled CoreML models are persisted across runs (the
// cold-cache check uses it). See cacheBase for the location policy.
func CoreMLCacheDir() string { return ensureDir(filepath.Join(cacheBase(), "coreml")) }

// cacheBase is muonnx's on-disk cache root for derived artifacts (compiled CoreML
// models, the pinned temp dir). It is deliberately NOT under $HOME — it lives
// next to the weights (often a large volume) and is treated like a weight
// artifact. Resolution:
//  1. $MUONNX_CACHE (explicit override)
//  2. <weight source>/.muonnx-cache  — co-located with weights
//  3. <dir of the executable>/.muonnx-cache  — local to the binary
//  4. <temp>/muonnx-cache  — last resort
func cacheBase() string {
	if d := os.Getenv("MUONNX_CACHE"); d != "" {
		return absOr(d)
	}
	// Co-locate with an explicitly-configured weight source. We skip the default
	// "/onnx" sentinel: it's commonly a read-only mount or absent on dev hosts, so
	// falling through to the executable dir is more reliable there. (ensureDir
	// degrades gracefully either way if the chosen dir isn't writable.)
	if ws := Environment().WeightSourcePath; ws != "" && ws != DefaultWeightSourcePath {
		return filepath.Join(absOr(ws), ".muonnx-cache")
	}
	if exe, err := os.Executable(); err == nil {
		return filepath.Join(filepath.Dir(exe), ".muonnx-cache")
	}
	return filepath.Join(os.TempDir(), "muonnx-cache")
}

func ensureDir(d string) string {
	if os.MkdirAll(d, 0o755) != nil {
		return ""
	}
	return d
}

func absOr(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		return a
	}
	return p
}

// graphOptLevel reads MUONNX_GRAPH_OPT (disable|basic|extended|all) into an ORT
// graph-optimization level. Returns (level, true) only when set, so the default
// stays ORT's ENABLE_ALL (highest optimization, slowest load).
func graphOptLevel() (ort.GraphOptimizationLevel, bool) {
	switch os.Getenv("MUONNX_GRAPH_OPT") {
	case "disable", "none":
		return ort.GraphOptimizationLevelDisableAll, true
	case "basic":
		return ort.GraphOptimizationLevelEnableBasic, true
	case "extended":
		return ort.GraphOptimizationLevelEnableExtended, true
	case "all":
		return ort.GraphOptimizationLevelEnableAll, true
	}
	return 0, false
}

func hasProvider(eps []string, want string) bool {
	for _, p := range eps {
		if p == want {
			return true
		}
	}
	return false
}
