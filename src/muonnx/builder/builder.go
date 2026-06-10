// Package builder turns resolved ONNX weights into a live, runnable model at
// runtime — i.e. it builds the model. (This is what an "OpenSession" helper
// would have done, named for what it actually is.) It auto-discovers the model's
// input/output names from the file so model packages never hardcode I/O lists.
package builder

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	muonnx "github.com/accretional/muonnx/src/muonnx"
)

// Build constructs a live ORT session from an on-disk ONNX file, returning the
// session plus the discovered input/output names. Session options (execution
// provider, threading) come from the muonnx environment. The caller must have
// initialized the ORT runtime (muonnx.Init) first. Close the session when done.
func Build(path string) (sess *ort.DynamicAdvancedSession, inputs, outputs []string, err error) {
	inInfo, outInfo, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("muonnx/builder: input/output info for %s: %w", path, err)
	}
	inputs = names(inInfo)
	outputs = names(outInfo)
	opts, err := muonnx.SessionOptions()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("muonnx/builder: session options: %w", err)
	}
	if opts != nil {
		defer opts.Destroy()
	}
	sess, err = ort.NewDynamicAdvancedSession(path, inputs, outputs, opts)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("muonnx/builder: build session for %s: %w", path, err)
	}
	return sess, inputs, outputs, nil
}

func names(infos []ort.InputOutputInfo) []string {
	out := make([]string, len(infos))
	for i, info := range infos {
		out[i] = info.Name
	}
	return out
}
