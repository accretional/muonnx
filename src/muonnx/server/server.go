// Package server is the top of the build graph: main starts an InferenceServer
// with the gRPC services it wants to host; the server builds their dependency
// graph (models → onnx deps, each once), registers their handlers, warms every
// model-backed service to readiness, and serves. It owns the loaded-resource
// lifecycle.
//
//	srv := server.New(eval.Service).Config(cfg).Options(grpc.MaxRecvMsgSize(64 << 20))
//	defer srv.Stop()
//	srv.Serve(lis)
package server

import (
	"context"
	"net"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pb "github.com/accretional/muonnx/proto/muonnxpb"
	muonnx "github.com/accretional/muonnx/src/muonnx"
	"github.com/accretional/muonnx/src/muonnx/service"
)

// InferenceServer hosts a set of muonnx services on one gRPC server.
type InferenceServer struct {
	opts  []grpc.ServerOption
	svcs  []service.Service
	srv   *grpc.Server
	built *muonnx.Built
}

// New creates a server for the given services.
func New(svcs ...service.Service) *InferenceServer {
	return &InferenceServer{svcs: svcs}
}

// Config applies an ONNXRuntimeConfig to the runtime Environment (precision +
// /onnx weight source) for every subsequent Resolve. Provisioned at the same
// scope as /onnx; the server reads it once. nil is ignored. Chains.
func (s *InferenceServer) Config(cfg *pb.ONNXRuntimeConfig) *InferenceServer {
	if cfg != nil {
		muonnx.SetConfig(cfg)
	}
	return s
}

// Options appends gRPC server options (interceptors, message-size caps). Must
// precede the first GRPC()/Start()/Serve(). Chains.
func (s *InferenceServer) Options(opts ...grpc.ServerOption) *InferenceServer {
	s.opts = append(s.opts, opts...)
	return s
}

// Add appends services to host. Chains.
func (s *InferenceServer) Add(svcs ...service.Service) *InferenceServer {
	s.svcs = append(s.svcs, svcs...)
	return s
}

// GRPC returns the underlying gRPC server, created lazily with the configured
// Options. Use it to register services not yet modeled as graph nodes.
func (s *InferenceServer) GRPC() *grpc.Server {
	if s.srv == nil {
		s.srv = grpc.NewServer(s.opts...)
	}
	return s.srv
}

// Start builds every service's dependency graph (models → onnx deps, each once)
// and registers their handlers, then enables reflection. Idempotent.
func (s *InferenceServer) Start() error {
	if s.built != nil {
		return nil
	}
	nodes := make([]muonnx.Node, len(s.svcs))
	for i, svc := range s.svcs {
		nodes[i] = svc
	}
	built, err := muonnx.Build(nodes...)
	if err != nil {
		return err
	}
	s.built = built
	for _, svc := range s.svcs {
		svc.Register(s.GRPC())
	}
	reflection.Register(s.srv)
	return nil
}

// Warm runs every hosted Warmer concurrently and returns once all are done (or
// with the first error). This is the readiness gate — Serve calls it before
// accepting RPCs.
func (s *InferenceServer) Warm(ctx context.Context) error {
	var wg sync.WaitGroup
	errs := make(chan error, len(s.svcs))
	for _, svc := range s.svcs {
		w, ok := svc.(service.Warmer)
		if !ok {
			continue
		}
		wg.Add(1)
		go func(w service.Warmer) {
			defer wg.Done()
			if err := w.Warm(ctx); err != nil {
				errs <- err
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// Serve builds + registers (if needed), warms to readiness, then blocks serving
// on lis until Stop. RPCs are accepted only once all services are warm.
func (s *InferenceServer) Serve(lis net.Listener) error {
	if err := s.Start(); err != nil {
		return err
	}
	if err := s.Warm(context.Background()); err != nil {
		return err
	}
	return s.srv.Serve(lis)
}

// Stop gracefully stops serving, releases every loaded model in reverse build
// order, and removes materialized temp weight files.
func (s *InferenceServer) Stop() error {
	if s.srv != nil {
		s.srv.GracefulStop()
	}
	defer muonnx.ReleaseWeights()
	if s.built != nil {
		err := s.built.Close()
		s.built = nil
		return err
	}
	return nil
}
