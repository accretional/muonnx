// Package model wraps one ONNX model as a build-graph node: Build resolves its
// weights (via the LoadFunc from muonnx.Load), builds them into a live session
// (auto I/O via the builder), and self-registers in the catalog. Run is a
// single-input/single-output float convenience used by simple services like
// cmd/eval; richer models call Session() and drive ORT directly.
package model

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
	muonnx "github.com/accretional/muonnx/src/muonnx"
	"github.com/accretional/muonnx/src/muonnx/builder"
)

// Model is a muonnx.Node holding a built ONNX session.
type Model struct {
	load muonnx.LoadFunc

	mu      sync.Mutex
	sess    *ort.DynamicAdvancedSession
	inputs  []string
	outputs []string
	name    string
}

// New creates a model node from a LoadFunc (typically a package's exported Load,
// e.g. model.New(mult0.Load)). The weights aren't touched until Build.
func New(load muonnx.LoadFunc) *Model { return &Model{load: load} }

// Deps reports no graph deps: a model's weight sub-deps are pulled by its
// LoadFunc, not the build graph. (Services list model nodes in their Deps.)
func (m *Model) Deps() []muonnx.Node { return nil }

// Build resolves the weights (and their sub-deps) and builds the live session,
// then registers the model in the catalog. Called once by muonnx.Build.
func (m *Model) Build() error {
	d, err := m.load()
	if err != nil {
		return err
	}
	sess, in, out, err := builder.Build(d.Path())
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.sess, m.inputs, m.outputs, m.name = sess, in, out, d.Name()
	m.mu.Unlock()
	muonnx.RegisterModel(&pb.ONNXModel{Name: d.Name(), Uri: d.Path()})
	return nil
}

// Close releases the ORT session.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sess != nil {
		_ = m.sess.Destroy()
		m.sess = nil
	}
	return nil
}

// Name returns the model's dep name (valid after Build).
func (m *Model) Name() string { return m.name }

// Run executes a single-input/single-output model on a flat row-major float
// tensor of the given shape, returning the output data + shape. Errors if the
// model isn't single-in/single-out. Safe for concurrent use.
func (m *Model) Run(data []float32, shape []int64) ([]float32, []int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sess == nil {
		return nil, nil, fmt.Errorf("muonnx/model %q: not built", m.name)
	}
	if len(m.inputs) != 1 || len(m.outputs) != 1 {
		return nil, nil, fmt.Errorf("muonnx/model %q: Run needs single in/out, has %d/%d",
			m.name, len(m.inputs), len(m.outputs))
	}
	in, err := ort.NewTensor(ort.NewShape(shape...), data)
	if err != nil {
		return nil, nil, fmt.Errorf("muonnx/model %q: input tensor: %w", m.name, err)
	}
	defer in.Destroy()

	outs := []ort.Value{nil} // nil => ORT auto-allocates the output
	if err := m.sess.Run([]ort.Value{in}, outs); err != nil {
		return nil, nil, fmt.Errorf("muonnx/model %q: run: %w", m.name, err)
	}
	out, ok := outs[0].(*ort.Tensor[float32])
	if !ok {
		if outs[0] != nil {
			outs[0].Destroy()
		}
		return nil, nil, fmt.Errorf("muonnx/model %q: non-float32 output", m.name)
	}
	defer out.Destroy()

	src := out.GetData()
	cp := make([]float32, len(src))
	copy(cp, src)
	return cp, []int64(out.GetShape()), nil
}
