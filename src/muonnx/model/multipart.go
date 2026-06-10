package model

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
	muonnx "github.com/accretional/muonnx/src/muonnx"
)

// multipart.go handles conjoined models: a single logical model made of several
// ONNX graphs that are built and managed together but orchestrated by the caller
// (e.g. Whisper = encoder + decoder-prefill + decoder-with-past sharing a KV
// cache). muonnx.model.Model covers the single-graph, single-in/out case; a
// Multipart covers everything beyond it.
//
// Multipart owns what is generic — resolving each part's weights, building each
// as its own ORT session with its OWN execution provider (so a static encoder can
// run on CoreML while a dynamic decoder stays on CPU), lifecycle, and catalog
// registration as one model — and hands back the live sessions (with discovered
// or declared I/O) for the caller to drive. The inference loop across parts is
// model-specific and stays with the caller.

// Part is one ONNX graph within a multipart model.
type Part struct {
	// Load resolves this part's weights (a muonnx.Load handle).
	Load muonnx.LoadFunc
	// Providers is this part's execution-provider preference (e.g. ["coreml"]).
	// nil/empty uses the environment default. This is the per-graph placement that
	// makes "encoder on ANE, decoder on CPU" a one-liner.
	Providers []string
	// Inputs/Outputs pin the session's I/O names + order. Leave nil to
	// auto-discover from the model (fine for simple graphs; pin them when order
	// matters, e.g. a KV-cache decoder's many tensors).
	Inputs  []string
	Outputs []string
}

// Multipart is a muonnx.Node bundling several Parts as one logical model.
type Multipart struct {
	name  string
	parts map[string]Part

	mu       sync.Mutex
	sessions map[string]*ort.DynamicAdvancedSession
	io       map[string][2][]string // part -> {inputs, outputs}
}

// NewMultipart declares a multipart model. name is the catalog name; parts maps a
// caller-chosen key (e.g. "encoder") to its Part. Nothing loads until Build.
func NewMultipart(name string, parts map[string]Part) *Multipart {
	return &Multipart{name: name, parts: parts}
}

// Deps: a Multipart resolves its parts' weights via their Load handles, not the
// build graph, so it has no graph deps (like Model).
func (m *Multipart) Deps() []muonnx.Node { return nil }

// Build resolves and opens every part as its own ORT session (each with its own
// execution provider), then registers the model in the catalog. Called once by
// muonnx.Build / the server.
func (m *Multipart) Build() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sessions != nil {
		return nil // idempotent: matches the build graph's build-once semantics
	}
	m.sessions = make(map[string]*ort.DynamicAdvancedSession, len(m.parts))
	m.io = make(map[string][2][]string, len(m.parts))
	for key, p := range m.parts {
		sess, in, out, err := buildPart(p)
		if err != nil {
			m.closeLocked()
			return fmt.Errorf("muonnx/model: multipart %q part %q: %w", m.name, key, err)
		}
		m.sessions[key] = sess
		m.io[key] = [2][]string{in, out}
	}
	muonnx.RegisterModel(&pb.ONNXModel{Name: m.name})
	return nil
}

func buildPart(p Part) (*ort.DynamicAdvancedSession, []string, []string, error) {
	d, err := p.Load()
	if err != nil {
		return nil, nil, nil, err
	}
	path := d.Path()
	in, out := p.Inputs, p.Outputs
	if len(in) == 0 || len(out) == 0 {
		inInfo, outInfo, err := ort.GetInputOutputInfo(path)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("input/output info: %w", err)
		}
		in, out = names(inInfo), names(outInfo)
	}
	providers := p.Providers
	if len(providers) == 0 {
		providers = muonnx.Environment().ExecutionProviders
	}
	opts, err := muonnx.SessionOptionsFor(providers...)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("session options: %w", err)
	}
	if opts != nil {
		defer opts.Destroy()
	}
	sess, err := ort.NewDynamicAdvancedSession(path, in, out, opts)
	if err != nil {
		return nil, nil, nil, err
	}
	return sess, in, out, nil
}

// Session returns the built session for a part (nil if unknown / not built).
func (m *Multipart) Session(part string) *ort.DynamicAdvancedSession {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sessions[part]
}

// Has reports whether the model includes this part. Build is all-or-nothing, so
// after a successful Build every declared part is present; optionality is handled
// by the caller choosing which parts to declare (e.g. include a KV-cache graph
// only when muonnx.Available reports it). An unknown key returns false.
func (m *Multipart) Has(part string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sessions[part] != nil
}

// IO returns a part's input and output tensor names (as built/discovered).
func (m *Multipart) IO(part string) (inputs, outputs []string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if io, ok := m.io[part]; ok {
		return io[0], io[1]
	}
	return nil, nil
}

// Close releases every part's session.
func (m *Multipart) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closeLocked()
	return nil
}

func (m *Multipart) closeLocked() {
	for k, s := range m.sessions {
		if s != nil {
			_ = s.Destroy()
		}
		delete(m.sessions, k)
	}
}

func names(infos []ort.InputOutputInfo) []string {
	out := make([]string, len(infos))
	for i, info := range infos {
		out[i] = info.Name
	}
	return out
}
