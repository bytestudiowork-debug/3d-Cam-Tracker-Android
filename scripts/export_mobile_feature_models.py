#!/usr/bin/env python3
import hashlib
import importlib
import os
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path(os.environ.get("RUNNER_TEMP", ROOT / ".model-export"))
SOURCE = WORK / "feature-source"
WEIGHTS = WORK / "weights"
ASSETS = ROOT / "app" / "src" / "main" / "assets" / "ai"
SOURCE_REVISION = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
EXTRACTOR_URL = (
    "https://raw.githubusercontent.com/verlab/accelerated_features/"
    + SOURCE_REVISION + "/weights/xfeat.pt"
)
EXTRACTOR_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
MATCHER_URL = (
    "https://huggingface.co/image-matching-models/xfeat/"
    "resolve/d1dd5fb/xfeat-lighterglue.pt"
)
MATCHER_SHA256 = "766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8"


def run(*args):
    subprocess.run(args, check=True)


def download(url, target, expected_sha):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "TrackStudio-Model-Builder"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != expected_sha:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Model checksum mismatch: expected {expected_sha}, received {actual}")


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    WORK.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        run("git", "clone", "--filter=blob:none", "https://github.com/verlab/accelerated_features.git", str(SOURCE))
    run("git", "-C", str(SOURCE), "fetch", "--depth=1", "origin", SOURCE_REVISION)
    run("git", "-C", str(SOURCE), "checkout", "--detach", SOURCE_REVISION)
    sys.path.insert(0, str(SOURCE))

    extractor_weights = WEIGHTS / "extractor.pt"
    matcher_weights = WEIGHTS / "matcher.pt"
    download(EXTRACTOR_URL, extractor_weights, EXTRACTOR_SHA256)
    download(MATCHER_URL, matcher_weights, MATCHER_SHA256)

    from modules.model import XFeatModel
    from modules.lighterglue import LighterGlue

    class ExtractorExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            descriptors, logits, reliability = self.model(image)
            return descriptors, logits, reliability

    extractor = XFeatModel().cpu().eval()
    extractor.load_state_dict(torch.load(extractor_weights, map_location="cpu"), strict=True)
    extractor_path = ASSETS / "extractor.onnx"
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        ExtractorExport(extractor),
        (torch.rand(1, 3, 480, 640),),
        str(extractor_path),
        input_names=["image"],
        output_names=["descriptors", "keypoint_logits", "reliability"],
        dynamic_axes={
            "image": {2: "image_height", 3: "image_width"},
            "descriptors": {2: "feature_height", 3: "feature_width"},
            "keypoint_logits": {2: "feature_height", 3: "feature_width"},
            "reliability": {2: "feature_height", 3: "feature_width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(extractor_path))
    extractor_session = ort.InferenceSession(str(extractor_path), providers=["CPUExecutionProvider"])
    for height, width in ((480, 640), (256, 448)):
        image = np.random.rand(1, 3, height, width).astype(np.float32)
        output = extractor_session.run(None, {"image": image})
        with torch.no_grad():
            reference = extractor(torch.from_numpy(image))
        for actual, expected in zip(output, reference):
            np.testing.assert_allclose(actual, expected.cpu().numpy(), rtol=2e-3, atol=2e-4)
        if output[0].shape != (1, 64, height // 8, width // 8):
            raise RuntimeError(f"Unexpected descriptor map shape: {output[0].shape}")
        if output[1].shape != (1, 65, height // 8, width // 8):
            raise RuntimeError(f"Unexpected keypoint map shape: {output[1].shape}")
        if output[2].shape != (1, 1, height // 8, width // 8):
            raise RuntimeError(f"Unexpected reliability map shape: {output[2].shape}")

    matcher_module = importlib.import_module("modules.lighterglue")
    matcher_module.LighterGlue.default_conf_xfeat["flash"] = False
    lighter = LighterGlue(str(matcher_weights))
    matcher = lighter.net.cpu().eval()
    matcher.conf.width_confidence = -1
    matcher.conf.depth_confidence = -1

    class MatcherExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, keypoints0, descriptors0, image_size0,
                    keypoints1, descriptors1, image_size1):
            output = self.model({
                "image0": {
                    "keypoints": keypoints0,
                    "descriptors": descriptors0,
                    "image_size": image_size0,
                },
                "image1": {
                    "keypoints": keypoints1,
                    "descriptors": descriptors1,
                    "image_size": image_size1,
                },
            })
            return output["log_assignment"]

    wrapper = MatcherExport(matcher).eval()
    sample = (
        torch.rand(1, 128, 2) * torch.tensor([640.0, 480.0]),
        torch.nn.functional.normalize(torch.rand(1, 128, 64), dim=-1),
        torch.tensor([[640.0, 480.0]]),
        torch.rand(1, 112, 2) * torch.tensor([640.0, 480.0]),
        torch.nn.functional.normalize(torch.rand(1, 112, 64), dim=-1),
        torch.tensor([[640.0, 480.0]]),
    )
    matcher_path = ASSETS / "matcher.onnx"
    torch.onnx.export(
        wrapper,
        sample,
        str(matcher_path),
        input_names=[
            "keypoints0", "descriptors0", "image_size0",
            "keypoints1", "descriptors1", "image_size1",
        ],
        output_names=["log_assignment"],
        dynamic_axes={
            "keypoints0": {1: "count0"},
            "descriptors0": {1: "count0"},
            "keypoints1": {1: "count1"},
            "descriptors1": {1: "count1"},
            "log_assignment": {1: "count0_with_dustbin", 2: "count1_with_dustbin"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(matcher_path))
    matcher_session = ort.InferenceSession(str(matcher_path), providers=["CPUExecutionProvider"])
    for count0, count1 in ((128, 112), (72, 96)):
        values = {
            "keypoints0": (torch.rand(1, count0, 2) * torch.tensor([640.0, 480.0])).numpy(),
            "descriptors0": torch.nn.functional.normalize(torch.rand(1, count0, 64), dim=-1).numpy(),
            "image_size0": np.array([[640.0, 480.0]], dtype=np.float32),
            "keypoints1": (torch.rand(1, count1, 2) * torch.tensor([640.0, 480.0])).numpy(),
            "descriptors1": torch.nn.functional.normalize(torch.rand(1, count1, 64), dim=-1).numpy(),
            "image_size1": np.array([[640.0, 480.0]], dtype=np.float32),
        }
        with torch.no_grad():
            reference = wrapper(
                torch.from_numpy(values["keypoints0"]),
                torch.from_numpy(values["descriptors0"]),
                torch.from_numpy(values["image_size0"]),
                torch.from_numpy(values["keypoints1"]),
                torch.from_numpy(values["descriptors1"]),
                torch.from_numpy(values["image_size1"]),
            ).cpu().numpy()
        assignment = matcher_session.run(["log_assignment"], values)[0]
        np.testing.assert_allclose(assignment, reference, rtol=3e-3, atol=3e-4)
        if assignment.shape != (1, count0 + 1, count1 + 1):
            raise RuntimeError(f"Unexpected assignment shape: {assignment.shape}")

    license_path = SOURCE / "LICENSE"
    if license_path.exists():
        (ASSETS / "THIRD_PARTY_LICENSES.txt").write_text(
            "The bundled model source and weights are distributed under Apache-2.0.\n"
            "Original source revision: " + SOURCE_REVISION + "\n"
            "Extractor weight SHA-256: " + EXTRACTOR_SHA256 + "\n"
            "Matcher weight SHA-256: " + MATCHER_SHA256 + "\n\n"
            + license_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    print("Generated and validated the two CPU inference assets.")
    for path in (extractor_path, matcher_path):
        print(f"{path.name}: {path.stat().st_size} bytes, sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
