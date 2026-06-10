// Package muonnx is a torch-free ONNX inference runtime + weight server for Go.
// Its spine is server → service → model → onnx (dep): the server starts gRPC
// services, services own models, models load their weights through this package
// and build them into live ORT sessions.
//
// The load decision hinges on the ONNXRuntimeEnvironment (host capabilities +
// config), computed at init and refined by the server's ONNXRuntimeConfig. See
// weight_registry.go for resolution (/onnx override vs package embed) and
// runtime.go for the build graph + ORT lifecycle.
package muonnx

import (
	"runtime"
	"sync"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
)

// DefaultWeightSourcePath is the central directory a deployment provisions
// weights into (ONNXWeightSource.path). Files here override a package's embedded
// copy, and for a model too big to embed they are the only — necessary — source.
const DefaultWeightSourcePath = "/onnx"

// ONNXRuntimeEnvironment is the resolved runtime context every weight load
// consults: host capabilities plus the active ONNXRuntimeConfig policy. Read it
// via Environment(); it changes only when the server calls SetConfig at startup.
type ONNXRuntimeEnvironment struct {
	OS        string   // runtime.GOOS
	Arch      string   // runtime.GOARCH
	Providers []string // ORT execution providers plausibly available on this host

	// WeightSourcePath is the central override dir (ONNXWeightSource.path);
	// /onnx by default.
	WeightSourcePath string

	// Precision policy. InferencePrec drives in-process loads (FP32 default —
	// CPU has no native fp16 kernels); ServingPrec drives weights handed to
	// external clients (FP16 default — WebGPU).
	InferencePrec Precision
	ServingPrec   Precision

	// Execution-provider + threading policy (from ONNXEnvironmentConfig). These
	// drive SessionOptions; default is plain CPU. ExecutionProviders is a
	// preference order, e.g. ["coreml","cpu"]; unknown/unavailable EPs are
	// skipped and CPU is always the implicit fallback.
	ExecutionProviders []string
	IntraOpThreads     int
	InterOpThreads     int
}

var (
	envMu sync.RWMutex
	env   ONNXRuntimeEnvironment

	// EP/threading policy is held separately from the ONNXRuntimeConfig-derived
	// env so SetConfig and SetEnvironmentConfig can be called in either order
	// without clobbering each other. Default: plain CPU.
	execProviders = []string{"cpu"}
	intraOp       int
	interOp       int
)

// Environment returns a copy of the current resolved environment (config-derived
// fields merged with the execution-provider/threading policy).
func Environment() ONNXRuntimeEnvironment {
	envMu.RLock()
	defer envMu.RUnlock()
	e := env
	e.ExecutionProviders = append([]string(nil), execProviders...)
	e.IntraOpThreads = intraOp
	e.InterOpThreads = interOp
	return e
}

// SetConfig injects the server's ONNXRuntimeConfig and recomputes the
// environment (precision policy + /onnx path). Call once at startup before any
// Build/Resolve. nil keeps host defaults. Does not touch execution-provider
// policy (see SetEnvironmentConfig).
func SetConfig(cfg *pb.ONNXRuntimeConfig) {
	envMu.Lock()
	defer envMu.Unlock()
	env = detectEnvironment(cfg)
}

// SetEnvironmentConfig applies host/runtime knobs from an ONNXEnvironmentConfig:
// the execution-provider preference order and intra/inter-op thread counts. Call
// before any Build. nil is a no-op; zero/empty fields keep current values.
func SetEnvironmentConfig(cfg *pb.ONNXEnvironmentConfig) {
	if cfg == nil {
		return
	}
	envMu.Lock()
	defer envMu.Unlock()
	if eps := cfg.GetExecutionProviders(); len(eps) > 0 {
		execProviders = append([]string(nil), eps...)
	}
	if n := cfg.GetIntraOpThreads(); n > 0 {
		intraOp = int(n)
	}
	if n := cfg.GetInterOpThreads(); n > 0 {
		interOp = int(n)
	}
}

// detectEnvironment builds the environment from the host, then overlays cfg.
// Pure + side-effect-free so init() can call it.
func detectEnvironment(cfg *pb.ONNXRuntimeConfig) ONNXRuntimeEnvironment {
	e := ONNXRuntimeEnvironment{
		OS:               runtime.GOOS,
		Arch:             runtime.GOARCH,
		Providers:        detectProviders(),
		WeightSourcePath: DefaultWeightSourcePath,
		InferencePrec:    FP32,
		ServingPrec:      FP16,
	}
	if cfg != nil {
		e.InferencePrec = precisionFromProto(cfg.GetInferencePrecision(), FP32)
		e.ServingPrec = precisionFromProto(cfg.GetServingPrecision(), FP16)
		if ws := cfg.GetWeightSource(); ws != nil && ws.GetPath() != "" {
			e.WeightSourcePath = ws.GetPath()
		}
	}
	return e
}

// detectProviders reports ORT execution providers plausibly available on this
// host (advisory; CPU always, CoreML on Apple).
func detectProviders() []string {
	p := []string{"cpu"}
	if runtime.GOOS == "darwin" {
		p = append(p, "coreml")
	}
	return p
}
