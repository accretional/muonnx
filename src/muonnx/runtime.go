package muonnx

import (
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
)

// runtime.go holds the build graph (the server→service→model→dep spine), the
// model catalog (owned here, not in main), and the ORT runtime lifecycle.

// --- build graph -----------------------------------------------------------

// Node is a build-graph node: Build() constructs it (after its Deps), called once
// per Build. A node that holds resources also implements Close (collected into
// the returned *Built). Models and services are Nodes.
type Node interface {
	Deps() []Node
	Build() error
}

// Built is the lifecycle handle for one Build: the loaded nodes that need
// releasing, closed in reverse order.
type Built struct {
	closers []func() error
}

// Close releases every loaded node in reverse build order, returning the first
// error (the rest still run).
func (b *Built) Close() error {
	var first error
	for i := len(b.closers) - 1; i >= 0; i-- {
		if err := b.closers[i](); err != nil && first == nil {
			first = err
		}
	}
	b.closers = nil
	return first
}

func closerFor(n Node) func() error {
	switch c := n.(type) {
	case interface{ Close() error }:
		return c.Close
	case interface{ Close() }:
		return func() error { c.Close(); return nil }
	default:
		return nil
	}
}

// Build builds each root and all transitive Deps exactly once (dedup by
// identity, deps before dependents) and returns a *Built whose Close releases
// them. On error it closes whatever was already built. Assumes a DAG.
func Build(roots ...Node) (*Built, error) {
	b := &Built{}
	done := map[Node]bool{}
	var visit func(Node) error
	visit = func(n Node) error {
		if done[n] {
			return nil
		}
		for _, d := range n.Deps() {
			if err := visit(d); err != nil {
				return err
			}
		}
		if err := n.Build(); err != nil {
			return err
		}
		done[n] = true
		if c := closerFor(n); c != nil {
			b.closers = append(b.closers, c)
		}
		return nil
	}
	for _, r := range roots {
		if err := visit(r); err != nil {
			_ = b.Close()
			return nil, err
		}
	}
	return b, nil
}

// --- catalog ---------------------------------------------------------------

var (
	catMu   sync.Mutex
	catalog []*pb.ONNXModel
)

// RegisterModel records a model in the runtime catalog. A model calls this from
// Build() so the catalog reflects what actually loaded; dedups by name.
func RegisterModel(m *pb.ONNXModel) {
	if m == nil || m.GetName() == "" {
		return
	}
	catMu.Lock()
	defer catMu.Unlock()
	for _, e := range catalog {
		if e.GetName() == m.GetName() {
			return
		}
	}
	catalog = append(catalog, m)
}

// Models returns a copy of the registered catalog. The ONNXRuntime weight
// service serves this.
func Models() []*pb.ONNXModel {
	catMu.Lock()
	defer catMu.Unlock()
	out := make([]*pb.ONNXModel, len(catalog))
	copy(out, catalog)
	return out
}

// ResetCatalog clears the catalog (tests).
func ResetCatalog() {
	catMu.Lock()
	defer catMu.Unlock()
	catalog = nil
}

// --- ORT runtime lifecycle -------------------------------------------------

var (
	ortOnce    sync.Once
	ortErr     error
	ortMatTemp string // materialized embedded dylib, removed on Shutdown
)

// Init initializes the shared ONNX Runtime environment. Resolution order for the
// shared library: explicit libPath > discovery (ONNXRUNTIME_LIB, system dirs,
// ./third_party) > embedded dylib (fat build) materialized to a temp file.
// Idempotent — call once at startup before any model Build.
func Init(libPath string) error {
	ortOnce.Do(func() {
		if libPath == "" {
			libPath = discoverLibrary()
		}
		if libPath == "" && HasEmbeddedDylib() {
			p, err := materializeDylib()
			if err != nil {
				ortErr = err
				return
			}
			ortMatTemp = p
			libPath = p
		}
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// Shutdown tears down the ORT environment and removes any materialized embedded
// dylib. Call once at process exit, after all sessions are closed.
func Shutdown() error {
	err := ort.DestroyEnvironment()
	if ortMatTemp != "" {
		os.Remove(ortMatTemp)
		ortMatTemp = ""
	}
	return err
}
