"""Export the RF-DETR keypoint model for the browser build.

Writes web/models/kp336_{fp32,fp16}.onnx (gitignored) and splits each into
equal chunks under GitHub's 50 MiB warning size, plus a manifest_<variant>.json
the page uses to download, verify and reassemble them:

    uv run --with "onnx>=1.17" --with onnxscript --with onnxruntime --with sympy \
        --with packaging python tools/export_web_model.py     # export + convert + split
    uv run python tools/export_web_model.py --split-only     # re-split existing files

Two variants, measured in Chromium on an M5 Max with onnxruntime-web 1.29
(web/dev/pipeline.html; details in docs/WEB_HOSTING.md):
  fp16  WebGPU on GPUs with shader-f16: ~25 ms/frame, 72 MB. Matches fp32 to
        within 0.01 detection score. Hangs on the CPU backend, so GPU only.
  fp32  GPUs without shader-f16 (~29 ms) and the CPU fallback (WASM), 143 MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "web" / "models"
RESOLUTION = 336
MAX_CHUNK = 48 * 1024 * 1024  # GitHub warns above 50 MiB per file


def name(variant: str) -> str:
    return f"kp{RESOLUTION}_{variant}.onnx"


def export_fp32(dest: Path) -> None:
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview(device="cpu", resolution=RESOLUTION)
    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp, shape=(RESOLUTION, RESOLUTION), batch_size=1)
        found = sorted(Path(tmp).rglob("*.onnx"), key=lambda p: p.stat().st_size)
        if not found:
            raise SystemExit("export produced no .onnx file")
        shutil.copy(found[-1], dest)
    print(f"exported {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def convert_fp16(src: Path, dest: Path) -> None:
    # onnxconverter_common's converter breaks this graph (mixed-type Conv and
    # attention Casts); ORT's transformer converter handles it. Inputs and
    # outputs stay fp32 so the page code is the same for both variants.
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(dest))
    print(f"converted {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def split(variant: str) -> None:
    src = MODELS / name(variant)
    for old in MODELS.glob(f"{name(variant)}.part*"):
        old.unlink()
    data = src.read_bytes()
    count = -(-len(data) // MAX_CHUNK)
    chunk = -(-len(data) // count)  # equal-sized chunks
    parts = []
    for i in range(0, len(data), chunk):
        part = MODELS / f"{name(variant)}.part{i // chunk}"
        part.write_bytes(data[i:i + chunk])
        parts.append(part.name)
    manifest = {
        "name": name(variant),
        "variant": variant,
        "resolution": RESOLUTION,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "parts": parts,
        "inputs": {"input": [1, 3, RESOLUTION, RESOLUTION]},
        "outputs": ["dets", "labels", "keypoints"],
        "license": "RF-DETR keypoint preview weights, Apache-2.0, Roboflow",
    }
    (MODELS / f"manifest_{variant}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{variant}: {len(parts)} chunks, sha256 {manifest['sha256'][:12]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-only", action="store_true", help="reuse the existing .onnx files")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    fp32, fp16 = MODELS / name("fp32"), MODELS / name("fp16")
    if not args.split_only or not fp32.exists():
        export_fp32(fp32)
    if not args.split_only or not fp16.exists():
        convert_fp16(fp32, fp16)
    for variant in ("fp32", "fp16"):
        split(variant)


if __name__ == "__main__":
    main()
