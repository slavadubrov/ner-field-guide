"""Small real ONNX graphs exercise package validation and captured dependencies."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("onnx"), "install onnx extra")
class ONNXContracts(unittest.TestCase):
    def test_topological_sort_includes_subgraph_capture(self):
        import onnx
        from onnx import TensorProto as T
        from onnx import helper as h

        from ner_demo.onnx_check import sort_graph

        branch = h.make_graph(
            [h.make_node("Identity", ["captured"], ["out"])],
            "branch",
            [],
            [h.make_tensor_value_info("out", T.FLOAT, [1])],
        )
        graph = h.make_graph(
            [
                h.make_node(
                    "If",
                    ["condition"],
                    ["result"],
                    then_branch=branch,
                    else_branch=branch,
                ),
                h.make_node("Identity", ["input"], ["captured"]),
            ],
            "outer",
            [
                h.make_tensor_value_info("condition", T.BOOL, []),
                h.make_tensor_value_info("input", T.FLOAT, [1]),
            ],
            [h.make_tensor_value_info("result", T.FLOAT, [1])],
        )
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 19)])
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(model)
        sort_graph(model.graph)
        onnx.checker.check_model(model)
        self.assertEqual(model.graph.node[0].op_type, "Identity")

    def test_missing_external_weights(self):
        import onnx
        from onnx import TensorProto as T
        from onnx import helper as h

        from ner_demo.onnx_check import validate_package

        tensor = onnx.TensorProto(
            name="weight", data_type=T.FLOAT, dims=[1], data_location=T.EXTERNAL
        )
        entry = tensor.external_data.add()
        entry.key, entry.value = "location", "absent.bin"
        model = h.make_model(
            h.make_graph(
                [],
                "external",
                [],
                [h.make_tensor_value_info("weight", T.FLOAT, [1])],
                [tensor],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in [
                "gliner_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ]:
                (path / name).write_text("{}")
            (path / "model.onnx").write_bytes(model.SerializeToString())
            with self.assertRaisesRegex(ValueError, "external weights"):
                validate_package(path)
