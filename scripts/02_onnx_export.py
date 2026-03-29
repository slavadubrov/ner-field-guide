"""Export GLiNER to ONNX + INT8 quantization."""
from gliner import GLiNER
import os

model = GLiNER.from_pretrained("urchade/gliner-medium-v2.1")
model.export_to_onnx("gliner.onnx")
size_mb = os.path.getsize("gliner.onnx") / (1024 * 1024)
print(f"ONNX model: {size_mb:.1f} MB")
print("Run: uv run python scripts/02_onnx_export.py")