// Command eval validates the muonnx runtime end to end: it builds an
// InferenceServer hosting one gRPC service whose Run composes two models —
// mult_0 then add_1 — so any random input must come back as all 1s. It exercises
// the whole spine: server → service → model → onnx dep, plus the Load convention,
// auto I/O discovery, the warm/readiness gate, and a real gRPC round-trip.
//
//	ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib go run ./cmd/eval
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/accretional/muonnx/models/add1"
	"github.com/accretional/muonnx/models/mult0"
	pb "github.com/accretional/muonnx/proto/muonnxpb"
	muonnx "github.com/accretional/muonnx/src/muonnx"
	"github.com/accretional/muonnx/src/muonnx/model"
	"github.com/accretional/muonnx/src/muonnx/server"
	"github.com/accretional/muonnx/src/muonnx/weightserver"
)

// evalService implements the Inference gRPC service. It owns two models and runs
// Run(x) = add_1(mult_0(x)). Its Deps are the two model nodes (the server builds
// them); Build itself is a no-op since the models carry the state.
type evalService struct {
	pb.UnimplementedInferenceServer
	mul *model.Model
	add *model.Model
}

func newEvalService() *evalService {
	return &evalService{mul: model.New(mult0.Load), add: model.New(add1.Load)}
}

func (e *evalService) Deps() []muonnx.Node              { return []muonnx.Node{e.mul, e.add} }
func (e *evalService) Build() error                     { return nil }
func (e *evalService) Register(r grpc.ServiceRegistrar) { pb.RegisterInferenceServer(r, e) }

// Warm satisfies service.Warmer: a throwaway compose so the first real RPC isn't
// cold; the server holds serving until it returns.
func (e *evalService) Warm(context.Context) error {
	_, err := e.compose([]float32{0}, []int64{1})
	return err
}

func (e *evalService) compose(data []float32, shape []int64) ([]float32, error) {
	zeros, _, err := e.mul.Run(data, shape)
	if err != nil {
		return nil, fmt.Errorf("mult_0: %w", err)
	}
	ones, _, err := e.add.Run(zeros, shape)
	if err != nil {
		return nil, fmt.Errorf("add_1: %w", err)
	}
	return ones, nil
}

func (e *evalService) Run(_ context.Context, in *pb.Tensor) (*pb.Tensor, error) {
	out, err := e.compose(in.GetData(), in.GetShape())
	if err != nil {
		return nil, err
	}
	return &pb.Tensor{Shape: in.GetShape(), Data: out}, nil
}

func main() {
	ortLib := flag.String("onnxruntime-lib", os.Getenv("ONNXRUNTIME_LIB"),
		"libonnxruntime path (or set ONNXRUNTIME_LIB)")
	flag.Parse()

	if err := muonnx.Init(*ortLib); err != nil {
		log.Fatalf("eval: ORT init (set -onnxruntime-lib / ONNXRUNTIME_LIB): %v", err)
	}
	defer func() { _ = muonnx.Shutdown() }()

	srv := server.New(newEvalService(), weightserver.New())
	defer func() { _ = srv.Stop() }()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("eval: listen: %v", err)
	}
	go func() {
		if err := srv.Serve(lis); err != nil {
			log.Printf("eval: serve ended: %v", err)
		}
	}()

	conn, err := grpc.NewClient(lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("eval: dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewInferenceClient(conn)

	const n = 16
	in := &pb.Tensor{Shape: []int64{n}, Data: make([]float32, n)}
	for i := range in.Data {
		in.Data[i] = rand.Float32()*1000 - 500 // arbitrary
	}

	resp, err := client.Run(context.Background(), in)
	if err != nil {
		log.Fatalf("eval: Run RPC: %v", err)
	}

	ok := len(resp.GetData()) == n
	for _, v := range resp.GetData() {
		if v < 0.9999 || v > 1.0001 {
			ok = false
		}
	}
	fmt.Printf("catalog: %v\n", modelNames())
	fmt.Printf("input  (first 4): %v ...\n", in.Data[:4])
	fmt.Printf("output         : %v\n", resp.GetData())
	if !ok {
		log.Fatalf("FAIL: expected all 1s, got %v", resp.GetData())
	}
	fmt.Println("PASS: random input -> all 1s through muonnx server (mult_0 -> add_1)")

	// --- weight server: ListModels + streaming Fetch ----------------------
	wc := pb.NewONNXRuntimeClient(conn)
	listed, err := listModels(wc)
	if err != nil {
		log.Fatalf("eval: ListModels: %v", err)
	}
	nbytes, err := fetchModel(wc, "mult_0")
	if err != nil {
		log.Fatalf("eval: Fetch: %v", err)
	}
	fmt.Printf("weight server: ListModels=%v, Fetch(mult_0)=%d bytes\n", listed, nbytes)
	if len(listed) != 2 || nbytes == 0 {
		log.Fatalf("FAIL: weight server returned models=%v bytes=%d", listed, nbytes)
	}
	fmt.Println("PASS: weight server ListModels + streaming Fetch")
}

func listModels(wc pb.ONNXRuntimeClient) ([]string, error) {
	stream, err := wc.ListModels(context.Background(), &pb.ONNXRequest{})
	if err != nil {
		return nil, err
	}
	var out []string
	for {
		m, err := stream.Recv()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, err
		}
		out = append(out, m.GetName())
	}
}

func fetchModel(wc pb.ONNXRuntimeClient, name string) (int, error) {
	stream, err := wc.Fetch(context.Background(), &pb.ONNXModel{Name: name})
	if err != nil {
		return 0, err
	}
	total := 0
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			return total, nil
		}
		if err != nil {
			return 0, err
		}
		total += len(chunk.GetContent())
	}
}

func modelNames() []string {
	ms := muonnx.Models()
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.GetName()
	}
	return out
}
