// Package weightserver is the default implementation of the ONNXRuntime weight
// service: it advertises the runtime catalog (ListModels) and streams model
// weights to external clients (Fetch) — e.g. transformers.js on WebGPU pulling an
// fp16 variant. It is a muonnx Service node with no model deps, so it composes on
// the same InferenceServer as inference services.
package weightserver

import (
	"fmt"
	"io"
	"os"

	"google.golang.org/grpc"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
	muonnx "github.com/accretional/muonnx/src/muonnx"
)

// chunkSize bounds each streamed Fetch message; weights of any size stream fine.
const chunkSize = 1 << 20 // 1 MiB

// Server implements muonnxpb.ONNXRuntimeServer and the muonnx Service contract.
type Server struct {
	pb.UnimplementedONNXRuntimeServer
}

// New constructs the weight server.
func New() *Server { return &Server{} }

// Deps / Build: the weight server holds no models — it serves the catalog and
// resolves bytes on demand — so there is nothing to build.
func (s *Server) Deps() []muonnx.Node { return nil }
func (s *Server) Build() error        { return nil }

// Register installs the ONNXRuntime handler.
func (s *Server) Register(r grpc.ServiceRegistrar) { pb.RegisterONNXRuntimeServer(r, s) }

// ListModels streams the runtime catalog (populated by models at Build time).
func (s *Server) ListModels(_ *pb.ONNXRequest, stream pb.ONNXRuntime_ListModelsServer) error {
	for _, m := range muonnx.Models() {
		if err := stream.Send(m); err != nil {
			return err
		}
	}
	return nil
}

// Fetch streams the requested model's weights at the requested precision
// (UNSPECIFIED -> serving precision). The variant is resolved wherever it lives
// (/onnx override or package embed), so a client can fetch an fp16 this host
// embeds but doesn't run for its own inference.
func (s *Server) Fetch(req *pb.ONNXModel, stream pb.ONNXRuntime_FetchServer) error {
	if req.GetName() == "" {
		return fmt.Errorf("muonnx/weightserver: ONNXModel.name required")
	}
	path, cleanup, err := muonnx.ResolveModel(req.GetName(), precFromProto(req.GetPrecision()))
	if err != nil {
		return err
	}
	defer cleanup()

	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("muonnx/weightserver: open %s: %w", path, err)
	}
	defer f.Close()

	buf := make([]byte, chunkSize)
	for {
		n, rerr := f.Read(buf)
		if n > 0 {
			if serr := stream.Send(&pb.ONNXChunk{Content: buf[:n]}); serr != nil {
				return serr
			}
		}
		if rerr == io.EOF {
			return nil
		}
		if rerr != nil {
			return fmt.Errorf("muonnx/weightserver: read %s: %w", path, rerr)
		}
	}
}

func precFromProto(p pb.Precision) muonnx.Precision {
	switch p {
	case pb.Precision_PRECISION_FP16:
		return muonnx.FP16
	case pb.Precision_PRECISION_FP32:
		return muonnx.FP32
	default:
		return muonnx.Auto
	}
}
