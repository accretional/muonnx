package muonnx

import (
	"os"
	"path/filepath"

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
// CoreML (Apple ANE/GPU) is appended when "coreml" is requested and the host is
// darwin; unsupported nodes still fall back to CPU within the session, and
// non-darwin coreml requests are skipped.
func SessionOptionsFor(providers ...string) (*ort.SessionOptions, error) {
	e := Environment()
	coreml := e.OS == "darwin" && hasProvider(providers, "coreml")
	if !coreml && e.IntraOpThreads == 0 && e.InterOpThreads == 0 {
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
		coremlOpts := map[string]string{
			"ModelFormat":              "MLProgram",
			"MLComputeUnits":           "ALL",
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
	return opts, nil
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

func hasProvider(eps []string, want string) bool {
	for _, p := range eps {
		if p == want {
			return true
		}
	}
	return false
}
