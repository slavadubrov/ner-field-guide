"""Export GLiNER to ONNX and an INT8-quantized variant."""

import shutil
from pathlib import Path

from gliner import GLiNER
from onnxruntime.quantization import QuantType, quantize_dynamic

MODEL_ID = "urchade/gliner_medium-v2.1"
ARTIFACT_DIR = Path("artifacts")
FP32_DIR = ARTIFACT_DIR / "gliner_medium.onnx"
FP32_MODEL = FP32_DIR / "model.onnx"
INT8_PATH = ARTIFACT_DIR / "gliner_medium_int8.onnx"


def sizeof_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    if FP32_DIR.exists():
        shutil.rmtree(FP32_DIR)
    print(f"Loading {MODEL_ID} ...")
    model = GLiNER.from_pretrained(MODEL_ID)

    print(f"Exporting to {FP32_DIR} (directory containing model.onnx)")
    model.export_to_onnx(str(FP32_DIR))
    if not FP32_MODEL.exists():
        raise FileNotFoundError(f"{FP32_MODEL} was not created by GLiNER export")
    print(f"→ Saved {sizeof_mb(FP32_MODEL):.1f} MB model file")

    print("Quantizing weights to INT8 ...")
    quantize_dynamic(
        model_input=str(FP32_MODEL),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"→ Saved {sizeof_mb(INT8_PATH):.1f} MB at {INT8_PATH}")

    print("Done. Load the INT8 model with onnxruntime for deployment tests.")


if __name__ == "__main__":
    main()
