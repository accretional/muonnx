package muonnx

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
)

// weight_registry.go is the acquire half of the load lifecycle. A model package
// declares its weights at init with the one-liner
//
//	var Load = muonnx.Load(weights, "mult_0", dep1.Load, dep2.Load)
//
// — "ask onnx for a callback to load." Load returns a LoadFunc bound to a shared
// Dep handle. Calling it loads the dep's sub-deps then resolves its own weights
// exactly once; every holder of the same name shares that result (dedup —
// "already loaded by something else"). Resolution prefers the central /onnx
// override dir over the package embed, at the Environment's precision. Turning
// the resolved file into a live ORT session is the builder's job (Build), kept
// separate on purpose.

// Precision selects an ONNX weight variant. Engine-facing (decoupled from the
// proto enum). Auto means "use the Environment default".
type Precision int

const (
	Auto Precision = iota
	FP32
	FP16
)

func (p Precision) String() string {
	switch p {
	case FP16:
		return "fp16"
	case FP32:
		return "fp32"
	default:
		return "auto"
	}
}

func precisionFromProto(p pb.Precision, dflt Precision) Precision {
	switch p {
	case pb.Precision_PRECISION_FP16:
		return FP16
	case pb.Precision_PRECISION_FP32:
		return FP32
	default:
		return dflt
	}
}

// Dep is a lazily-loaded, process-shared handle to one dependency's ONNX file.
type Dep struct {
	name  string
	embed fs.FS
	deps  []LoadFunc
	once  sync.Once
	path  string
	clean func()
	err   error
}

// LoadFunc loads a Dep (its sub-deps first, then itself) and returns the shared
// handle. Idempotent and safe to call from multiple holders.
type LoadFunc func() (*Dep, error)

var (
	regMu    sync.Mutex
	registry = map[string]*Dep{}
)

// Load registers name's weights (embed) and its dep load funcs, returning a
// LoadFunc. Two callers naming the same dep share one handle (one load). pkgEmbed
// is the caller's `//go:embed onnx/*.onnx` FS (nil = resolve from /onnx only).
func Load(pkgEmbed fs.FS, name string, deps ...LoadFunc) LoadFunc {
	regMu.Lock()
	d, ok := registry[name]
	if !ok {
		d = &Dep{name: name, embed: pkgEmbed, deps: deps}
		registry[name] = d
	} else {
		if d.embed == nil {
			d.embed = pkgEmbed
		}
		if len(d.deps) == 0 {
			d.deps = deps
		}
	}
	regMu.Unlock()
	return d.Load
}

// Load resolves sub-deps then this dep's weights once (shared), returning the
// handle. Path() is valid afterward.
func (d *Dep) Load() (*Dep, error) {
	for _, dl := range d.deps {
		if _, err := dl(); err != nil {
			return nil, fmt.Errorf("muonnx: dep of %q: %w", d.name, err)
		}
	}
	d.once.Do(func() {
		d.path, d.clean, d.err = ResolveAt(d.embed, d.name, Environment().InferencePrec)
	})
	if d.err != nil {
		return nil, d.err
	}
	return d, nil
}

// Path returns the resolved on-disk weight path (valid after Load).
func (d *Dep) Path() string { return d.path }

// Name returns the dependency name.
func (d *Dep) Name() string { return d.name }

// ResolveModel resolves a registered model by name at the given precision,
// returning a path + cleanup. Auto uses the serving precision. Used by the weight
// server's Fetch to hand external clients a variant (often fp16) independent of
// what this host loads for inference. Errors if name was never Registered.
func ResolveModel(name string, prec Precision) (string, func(), error) {
	regMu.Lock()
	d, ok := registry[name]
	regMu.Unlock()
	if !ok {
		return "", func() {}, fmt.Errorf("muonnx: model %q not registered", name)
	}
	if prec == Auto {
		prec = Environment().ServingPrec
	}
	return ResolveAt(d.embed, name, prec)
}

// ReleaseWeights removes every materialized temp weight file. Call at shutdown.
func ReleaseWeights() {
	regMu.Lock()
	defer regMu.Unlock()
	for _, d := range registry {
		if d.clean != nil {
			d.clean()
			d.clean = nil
		}
	}
}

// --- resolution ------------------------------------------------------------
//
// Layout convention, identical in the central /onnx dir and a package embed:
//
//	fp32 -> "<dep>.onnx"        (/onnx/mult_0.onnx,  embed onnx/mult_0.onnx)
//	fp16 -> "fp16/<dep>.onnx"   (/onnx/fp16/mult_0.onnx, embed onnx/fp16/mult_0.onnx)
//
// Resolve cares about precision first (Environment), then location (/onnx wins
// over embed; if a quant isn't embedded, /onnx is the necessary source).

// ResolveAt resolves dep at an explicit precision, returning a path + cleanup.
func ResolveAt(pkgEmbed fs.FS, dep string, prec Precision) (path string, cleanup func(), err error) {
	noop := func() {}
	e := Environment()
	order := variantOrder(dep, prec)
	for _, rel := range order {
		if e.WeightSourcePath != "" {
			p := filepath.Join(e.WeightSourcePath, rel)
			if st, statErr := os.Stat(p); statErr == nil && !st.IsDir() {
				return p, noop, nil
			}
		}
		if pkgEmbed != nil {
			if data, rdErr := fs.ReadFile(pkgEmbed, "onnx/"+rel); rdErr == nil {
				return materialize(dep, data)
			}
		}
	}
	return "", noop, fmt.Errorf("muonnx: dep %q (%s) not found in %q or embedded onnx/ (tried %v)",
		dep, prec, e.WeightSourcePath, order)
}

// Resolve resolves dep at the Environment's inference precision.
func Resolve(pkgEmbed fs.FS, dep string) (string, func(), error) {
	return ResolveAt(pkgEmbed, dep, Environment().InferencePrec)
}

// ResolveServing resolves dep at the Environment's serving precision — what the
// weight server hands external clients (fp16 by default), independent of what
// this host loads for inference.
func ResolveServing(pkgEmbed fs.FS, dep string) (string, func(), error) {
	return ResolveAt(pkgEmbed, dep, Environment().ServingPrec)
}

// Available reports whether dep is resolvable at inference precision without
// materializing anything.
func Available(pkgEmbed fs.FS, dep string) bool {
	e := Environment()
	for _, rel := range variantOrder(dep, e.InferencePrec) {
		if e.WeightSourcePath != "" {
			if st, err := os.Stat(filepath.Join(e.WeightSourcePath, rel)); err == nil && !st.IsDir() {
				return true
			}
		}
		if pkgEmbed != nil {
			if _, err := fs.Stat(pkgEmbed, "onnx/"+rel); err == nil {
				return true
			}
		}
	}
	return false
}

func variantOrder(dep string, p Precision) []string {
	fp32 := dep + ".onnx"
	fp16 := "fp16/" + dep + ".onnx"
	if p == FP16 {
		return []string{fp16, fp32}
	}
	return []string{fp32, fp16}
}

func materialize(dep string, data []byte) (string, func(), error) {
	noop := func() {}
	tmp, err := os.CreateTemp("", "muonnx-"+filepath.Base(dep)+"-*.onnx")
	if err != nil {
		return "", noop, fmt.Errorf("muonnx: temp for %q: %w", dep, err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", noop, fmt.Errorf("muonnx: write temp for %q: %w", dep, err)
	}
	tmp.Close()
	name := tmp.Name()
	return name, func() { os.Remove(name) }, nil
}
