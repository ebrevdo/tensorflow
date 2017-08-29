package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

cuda_py_test(
    name = "list_ops_test",
    size = "small",
    srcs = ["list_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/contrib/tensor_list/python/ops:list_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
    ],
)
