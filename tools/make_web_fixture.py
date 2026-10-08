"""Build the browser decode test fixture from a public sample image.

Runs the exported ONNX model on the image, then decodes its raw outputs with
rf-detr's own PostProcess. tests in web/tests check that the JavaScript decoder
reproduces those poses exactly. Uses Roboflow's public people-walking sample,
so no personal data ends up in the repo.

    uv run --with onnxruntime python tools/make_web_fixture.py IMAGE
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main(image: str) -> None:
    from rfdetr import RFDETRKeypointPreview

    frame = cv2.resize(cv2.imread(image), (1280, 720))
    x = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (336, 336), interpolation=cv2.INTER_LINEAR)
    x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    sess = ort.InferenceSession(str(ROOT / "web/models/kp336_fp32.onnx"))
    dets, labels, kps = sess.run(None, {"input": x})

    model = RFDETRKeypointPreview(device="cpu", resolution=336)
    pp = model.model.postprocess
    out = pp({"pred_logits": torch.from_numpy(labels), "pred_boxes": torch.from_numpy(dets),
              "pred_keypoints": torch.from_numpy(kps)}, target_sizes=torch.tensor([[720, 1280]]))[0]
    keep = (out["scores"] > 0.5) & (out["labels"] == 1)
    expected = [
        {"score": float(s), "box": [float(v) for v in b],
         "keypoints": [[float(v) for v in k] for k in kp]}
        for s, b, kp in zip(out["scores"][keep], out["boxes"][keep], out["keypoints"][keep])
    ]
    fixture = {
        "size": [1280, 720], "threshold": 0.5,
        "trace_alpha": pp.trace_alpha, "num_select": pp.num_select,
        "dets": dets.ravel().round(6).tolist(), "dets_shape": list(dets.shape),
        "labels": labels.ravel().round(6).tolist(), "labels_shape": list(labels.shape),
        "keypoints": kps.ravel().round(6).tolist(), "keypoints_shape": list(kps.shape),
        "expected": expected,
    }
    path = ROOT / "web/tests/fixtures/decode_people_walking.json"
    path.write_text(json.dumps(fixture))
    print(f"{len(expected)} people expected; wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
