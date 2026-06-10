package muonnx

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
)

// ort.go discovers the ONNX Runtime shared library so Init("") works without an
// explicit path. Order: ONNXRUNTIME_LIB env, common system install dirs, then a
// third_party/onnxruntime-*/lib checkout under the working directory. Embedding
// the dylib into the binary (the fully self-contained build) is a later addition
// — see scripts/prep_embed.sh and the docs.
func discoverLibrary() string {
	if p := os.Getenv("ONNXRUNTIME_LIB"); p != "" {
		return p
	}
	name := "libonnxruntime.so"
	if runtime.GOOS == "darwin" {
		name = "libonnxruntime.dylib"
	}
	candidates := []string{
		filepath.Join("/opt/homebrew/lib", name),
		filepath.Join("/usr/local/lib", name),
	}
	if m, _ := filepath.Glob(filepath.Join("third_party", "onnxruntime-*", "lib", name)); len(m) > 0 {
		candidates = append(candidates, m...)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// HasEmbeddedDylib reports whether this binary was built with -tags muonnx_fat
// and carries the ORT shared library.
func HasEmbeddedDylib() bool { return len(ortLib) > 0 }

func ortLibExt() string {
	if runtime.GOOS == "darwin" {
		return ".dylib"
	}
	return ".so"
}

// materializeDylib writes the embedded ORT shared library to a temp file and
// returns its path (executable). Used as the last-resort fallback in Init when
// no library is found on disk. Caller removes the file (Shutdown does).
func materializeDylib() (string, error) {
	if len(ortLib) == 0 {
		return "", errors.New("muonnx: no embedded ORT dylib (build with -tags muonnx_fat)")
	}
	tmp, err := os.CreateTemp("", "libonnxruntime-*"+ortLibExt())
	if err != nil {
		return "", err
	}
	if _, err := tmp.Write(ortLib); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", err
	}
	tmp.Close()
	if err := os.Chmod(tmp.Name(), 0o755); err != nil {
		os.Remove(tmp.Name())
		return "", err
	}
	return tmp.Name(), nil
}
