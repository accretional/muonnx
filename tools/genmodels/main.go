// Command genmodels authors muonnx's tiny test ONNX models in pure Go, by
// constructing onnx.ModelProto values and marshaling them — no Python, fully
// reproducible with `go run ./tools/genmodels`.
//
// It writes two models, each a single elementwise op over a dynamic-length
// float vector against a scalar constant:
//
//	mult_0 : Y = X * 0   (-> all zeros, any input)
//	add_1  : Y = X + 1   (-> input + 1)
//
// Chained (mult_0 then add_1), any random input becomes all ones — the property
// cmd/eval validates end-to-end through the muonnx server. Files land inside the
// model packages' embed dirs so `//go:embed onnx/*.onnx` picks them up:
//
//	models/mult0/onnx/mult_0.onnx
//	models/add1/onnx/add_1.onnx
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"google.golang.org/protobuf/proto"

	onnxpb "github.com/accretional/muonnx/tools/onnxschema/onnxpb"
)

const (
	irVersion  = 8  // ONNX IR; ORT 1.27 accepts well beyond this
	opsetVer   = 13 // Mul/Add are stable far below this
	floatDType = int32(onnxpb.TensorProto_FLOAT)
)

func main() {
	out := map[string]*onnxpb.ModelProto{
		"models/mult0/onnx/mult_0.onnx":   binaryScalarModel("Mul", "zero", 0),
		"models/add1/onnx/add_1.onnx":     binaryScalarModel("Add", "one", 1),
		"models/addtwo/onnx/add_two.onnx": twoInputModel("Add"), // Y = A + B (multi-IO Session demo)
	}
	for path, m := range out {
		if err := writeModel(path, m); err != nil {
			log.Fatalf("genmodels: %v", err)
		}
		fmt.Printf("wrote %s\n", path)
	}
}

// binaryScalarModel builds Y = op(X, <scalar const>) over a 1-D float vector of
// dynamic length N. The constant is a graph initializer (baked in), so X is the
// model's only input. Scalar broadcasts over X under ONNX numpy broadcasting.
func binaryScalarModel(opType, constName string, constVal float32) *onnxpb.ModelProto {
	vec := func(name string) *onnxpb.ValueInfoProto {
		return &onnxpb.ValueInfoProto{
			Name: proto.String(name),
			Type: &onnxpb.TypeProto{Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: proto.Int32(floatDType),
					Shape: &onnxpb.TensorShapeProto{Dim: []*onnxpb.TensorShapeProto_Dimension{
						{Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}},
					}},
				},
			}},
		}
	}
	scalar := &onnxpb.TensorProto{
		Name:      proto.String(constName),
		DataType:  proto.Int32(floatDType),
		Dims:      nil, // rank-0 scalar
		FloatData: []float32{constVal},
	}
	node := &onnxpb.NodeProto{
		Name:   proto.String(opType + "_node"),
		OpType: proto.String(opType),
		Input:  []string{"X", constName},
		Output: []string{"Y"},
	}
	return &onnxpb.ModelProto{
		IrVersion:    proto.Int64(irVersion),
		ProducerName: proto.String("muonnx-genmodels"),
		OpsetImport:  []*onnxpb.OperatorSetIdProto{{Domain: proto.String(""), Version: proto.Int64(opsetVer)}},
		Graph: &onnxpb.GraphProto{
			Name:        proto.String(opType + "_graph"),
			Node:        []*onnxpb.NodeProto{node},
			Initializer: []*onnxpb.TensorProto{scalar},
			Input:       []*onnxpb.ValueInfoProto{vec("X")},
			Output:      []*onnxpb.ValueInfoProto{vec("Y")},
		},
	}
}

// twoInputModel builds Y = op(A, B) over two 1-D float vectors of dynamic length
// N — a genuinely multi-input graph for exercising model.Session.
func twoInputModel(opType string) *onnxpb.ModelProto {
	vec := func(name string) *onnxpb.ValueInfoProto {
		return &onnxpb.ValueInfoProto{
			Name: proto.String(name),
			Type: &onnxpb.TypeProto{Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: proto.Int32(floatDType),
					Shape: &onnxpb.TensorShapeProto{Dim: []*onnxpb.TensorShapeProto_Dimension{
						{Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}},
					}},
				},
			}},
		}
	}
	node := &onnxpb.NodeProto{
		Name:   proto.String(opType + "_node"),
		OpType: proto.String(opType),
		Input:  []string{"A", "B"},
		Output: []string{"Y"},
	}
	return &onnxpb.ModelProto{
		IrVersion:    proto.Int64(irVersion),
		ProducerName: proto.String("muonnx-genmodels"),
		OpsetImport:  []*onnxpb.OperatorSetIdProto{{Domain: proto.String(""), Version: proto.Int64(opsetVer)}},
		Graph: &onnxpb.GraphProto{
			Name:   proto.String(opType + "_two_graph"),
			Node:   []*onnxpb.NodeProto{node},
			Input:  []*onnxpb.ValueInfoProto{vec("A"), vec("B")},
			Output: []*onnxpb.ValueInfoProto{vec("Y")},
		},
	}
}

func writeModel(path string, m *onnxpb.ModelProto) error {
	data, err := proto.Marshal(m)
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
