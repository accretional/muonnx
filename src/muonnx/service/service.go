// Package service defines what a muonnx gRPC service is: a build-graph node that
// also registers a gRPC handler, and that may warm its models before the server
// serves. Concrete services live in their own packages (or in cmd/), list their
// model nodes in Deps(), read them via typed getters in Build(), and bind their
// proto server in Register().
package service

import (
	"context"

	"google.golang.org/grpc"

	muonnx "github.com/accretional/muonnx/src/muonnx"
)

// Service is a muonnx.Node that registers a gRPC handler once built.
type Service interface {
	muonnx.Node
	Register(grpc.ServiceRegistrar)
}

// Warmer is an optional Service capability: a service backing RPCs with a model
// gives it a warming invocation (a throwaway inference) so the first real RPC
// isn't cold. The server warms every Warmer and waits for all before serving —
// that wait is readiness. Services with no model don't implement it.
type Warmer interface {
	Warm(context.Context) error
}
