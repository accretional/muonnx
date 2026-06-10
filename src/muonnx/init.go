package muonnx

// init.go computes the ONNXRuntimeEnvironment from the host at package load, so
// any model that resolves weights before the server injects a config still sees
// sane defaults (fp32 inference, /onnx weight source). The server refines this
// via SetConfig at startup. Note: this does NOT initialize the ORT C runtime —
// that needs a shared-library path and is done explicitly via Init().
func init() {
	SetConfig(nil)
}
