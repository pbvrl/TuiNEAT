#!/usr/bin/env bash

: '
(Re)Generate the grpc code from the .proto file.

Requires these programs:
    protobuf
    protoc-gen-go
    protoc-gen-go-grpc
and these Python libraries:
    grpcio
    grpcio-tools

Run from the project root:
(master) ~/p/tuineat > training_visualizer/generate_from_proto.sh

It should generate:
.
├── training_visualizer
│   └── grpc_api
│       ├── schema_grpc.pb.go
│       ├── schema.pb.go
│       └── schema.proto       <- From this file
│  
├── src
│   └── extra_modules
│       └── visualization
│           └── grpc_api
│               ├── schema_pb2_grpc.py
│               ├── schema_pb2.py
│               └── schema.proto
...
'

# Go
protoc --go_out=. --go-grpc_out=. training_visualizer/grpc_api/schema.proto
# Python
## https://github.com/protocolbuffers/protobuf/issues/1491#issuecomment-263879052
## To avoid import path issues:
mkdir -p src/extra_modules/training_visualizer/grpc_api
cp training_visualizer/grpc_api/schema.proto src/extra_modules/training_visualizer/grpc_api/schema.proto
python -m grpc_tools.protoc --proto_path=. --grpc_python_out=. --python_out=.  src/extra_modules/training_visualizer/grpc_api/schema.proto
