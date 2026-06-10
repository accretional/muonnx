package model

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
	muonnx "github.com/accretional/muonnx/src/muonnx"
)

// session.go is the single-graph, multi-IO model wrapper — the middle ground
// between Model (one graph, single float32 in/out, the convenience case) and
// Multipart (several graphs you drive with raw ORT). A Session runs ONE graph
// with any number of named inputs/outputs and mixed dtypes (e.g. int64 ids +
// float32 tensors), creating and destroying the ORT tensors for you. Reach for
// raw ORT (via builder/Multipart) only for hot loops that reuse tensors across
// calls; for everything else Session removes the []ort.Value boilerplate.

// Tensor is a typed dense tensor passed to or returned from a Session. Exactly
// one data field is non-nil; its length must equal the product of Shape.
type Tensor struct {
	Shape []int64
	F32   []float32
	I64   []int64
	I32   []int32
}

// F32/I64/I32 build a Tensor of the given element type.
func F32(shape []int64, data []float32) Tensor { return Tensor{Shape: shape, F32: data} }
func I64(shape []int64, data []int64) Tensor   { return Tensor{Shape: shape, I64: data} }
func I32(shape []int64, data []int32) Tensor   { return Tensor{Shape: shape, I32: data} }

// Session is a muonnx.Node wrapping one built ONNX graph with an ergonomic
// multi-IO Run. I/O names are auto-discovered at Build.
type Session struct {
	load      muonnx.LoadFunc
	providers []string

	mu   sync.Mutex
	sess *ort.DynamicAdvancedSession
	in   []string
	out  []string
	name string
}

// NewSession declares a single-graph model from a weight loader. providers is the
// execution-provider preference (empty = environment default). Nothing loads
// until Build.
func NewSession(load muonnx.LoadFunc, providers ...string) *Session {
	return &Session{load: load, providers: providers}
}

// Deps: like Model, a Session pulls its weights via its Load handle, not the
// build graph.
func (s *Session) Deps() []muonnx.Node { return nil }

// Build resolves the weights, opens the session with its execution provider,
// discovers I/O names, and registers the model in the catalog.
func (s *Session) Build() error {
	s.mu.Lock()
	built := s.sess != nil
	s.mu.Unlock()
	if built {
		return nil // idempotent: matches the build graph's build-once semantics
	}
	d, err := s.load()
	if err != nil {
		return err
	}
	providers := s.providers
	if len(providers) == 0 {
		providers = muonnx.Environment().ExecutionProviders
	}
	opts, err := muonnx.SessionOptionsFor(providers...)
	if err != nil {
		return fmt.Errorf("muonnx/model: session options: %w", err)
	}
	if opts != nil {
		defer opts.Destroy()
	}
	inInfo, outInfo, err := ort.GetInputOutputInfo(d.Path())
	if err != nil {
		return fmt.Errorf("muonnx/model: input/output info for %s: %w", d.Name(), err)
	}
	in, out := names(inInfo), names(outInfo)
	sess, err := ort.NewDynamicAdvancedSession(d.Path(), in, out, opts)
	if err != nil {
		return fmt.Errorf("muonnx/model: build session %s: %w", d.Name(), err)
	}
	s.mu.Lock()
	s.sess, s.in, s.out, s.name = sess, in, out, d.Name()
	s.mu.Unlock()
	muonnx.RegisterModel(&pb.ONNXModel{Name: d.Name(), Uri: d.Path()})
	return nil
}

// Inputs and Outputs return the graph's tensor names (valid after Build).
func (s *Session) Inputs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.in...)
}

func (s *Session) Outputs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.out...)
}

// Run executes the graph: inputs maps every input name to a Tensor; the result
// maps every output name to a Tensor (data copied out, ORT values released).
// Safe for concurrent use (serialized internally).
func (s *Session) Run(inputs map[string]Tensor) (map[string]Tensor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sess == nil {
		return nil, fmt.Errorf("muonnx/model: session %q not built", s.name)
	}

	var toDestroy []ort.Value
	defer func() {
		for _, v := range toDestroy {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	ins := make([]ort.Value, len(s.in))
	for i, name := range s.in {
		t, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("muonnx/model: session %q missing input %q", s.name, name)
		}
		v, err := newValue(t)
		if err != nil {
			return nil, fmt.Errorf("muonnx/model: input %q: %w", name, err)
		}
		ins[i], toDestroy = v, append(toDestroy, v)
	}

	outs := make([]ort.Value, len(s.out)) // nil entries => ORT auto-allocates
	runErr := s.sess.Run(ins, outs)
	// Collect every allocated output for cleanup BEFORE checking the error or
	// reading: ORT may have allocated some outputs even when Run (or its internal
	// value conversion) ultimately errors, and a readValue failure below must not
	// strand the remaining outputs.
	for _, v := range outs {
		if v != nil {
			toDestroy = append(toDestroy, v)
		}
	}
	if runErr != nil {
		return nil, fmt.Errorf("muonnx/model: session %q run: %w", s.name, runErr)
	}

	res := make(map[string]Tensor, len(s.out))
	for i, name := range s.out {
		t, err := readValue(outs[i])
		if err != nil {
			return nil, fmt.Errorf("muonnx/model: output %q: %w", name, err)
		}
		res[name] = t
	}
	return res, nil
}

// Close releases the session.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sess != nil {
		_ = s.sess.Destroy()
		s.sess = nil
	}
	return nil
}

func newValue(t Tensor) (ort.Value, error) {
	sh := ort.NewShape(t.Shape...)
	switch {
	case t.F32 != nil:
		return ort.NewTensor(sh, t.F32)
	case t.I64 != nil:
		return ort.NewTensor(sh, t.I64)
	case t.I32 != nil:
		return ort.NewTensor(sh, t.I32)
	default:
		return nil, fmt.Errorf("empty tensor (no F32/I64/I32 data)")
	}
}

func readValue(v ort.Value) (Tensor, error) {
	switch x := v.(type) {
	case *ort.Tensor[float32]:
		return Tensor{Shape: cloneShape(x.GetShape()), F32: append([]float32(nil), x.GetData()...)}, nil
	case *ort.Tensor[int64]:
		return Tensor{Shape: cloneShape(x.GetShape()), I64: append([]int64(nil), x.GetData()...)}, nil
	case *ort.Tensor[int32]:
		return Tensor{Shape: cloneShape(x.GetShape()), I32: append([]int32(nil), x.GetData()...)}, nil
	default:
		return Tensor{}, fmt.Errorf("unsupported output type %T", v)
	}
}

func cloneShape(s ort.Shape) []int64 { return append([]int64(nil), []int64(s)...) }
