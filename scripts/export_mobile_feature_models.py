#!/usr/bin/env python3
import hashlib
import importlib
import os
import pathlib
import subprocess
import sys
import types
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
    import torch.nn.functional as F

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

    class InstanceNormExport(nn.Module):
        def forward(self, image):
            mean = image.mean(dim=(2, 3), keepdim=True)
            variance = (image - mean).square().mean(dim=(2, 3), keepdim=True)
            return (image - mean) / torch.sqrt(variance + 1e-5)

    class ExportableFeatureModel(XFeatModel):
        def _unfold2d(self, image, ws=2):
            batch, channels, height, width = image.shape
            image = image.reshape(
                batch, channels, height // ws, ws, width // ws, ws
            )
            image = image.permute(0, 1, 3, 5, 2, 4)
            return image.reshape(
                batch, channels * ws * ws, height // ws, width // ws
            )

    class ExtractorExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            descriptors, logits, reliability = self.model(image)
            return descriptors, logits, reliability

    weights = torch.load(extractor_weights, map_location="cpu")
    extractor = ExportableFeatureModel().cpu().eval()
    extractor.load_state_dict(weights, strict=True)
    extractor.norm = InstanceNormExport()
    reference_extractor = XFeatModel().cpu().eval()
    reference_extractor.load_state_dict(weights, strict=True)
    parity_input = torch.rand(1, 3, 480, 640)
    with torch.no_grad():
        reference_outputs = reference_extractor(parity_input)
        export_outputs = extractor(parity_input)
    for reference_output, export_output in zip(reference_outputs, export_outputs):
        torch.testing.assert_close(export_output, reference_output, rtol=1e-5, atol=1e-5)

    extractor_path = ASSETS / "extractor.onnx"
    ASSETS.mkdir(parents=True, exist_ok=True)
    extractor_export = ExtractorExport(extractor).cpu().eval()
    torch.onnx.export(
        extractor_export,
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
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
    )
    onnx.checker.check_model(str(extractor_path))
    extractor_session = ort.InferenceSession(str(extractor_path), providers=["CPUExecutionProvider"])
    for height, width in ((480, 640), (256, 448)):
        image = np.random.rand(1, 3, height, width).astype(np.float32)
        output_names = ["descriptors", "keypoint_logits", "reliability"]
        output = extractor_session.run(output_names, {"image": image})
        with torch.no_grad():
            reference = extractor(torch.from_numpy(image))
        for output_name, actual, expected in zip(output_names, output, reference):
            print(
                output_name,
                "max_abs_diff=",
                float(np.max(np.abs(actual - expected.cpu().numpy()))),
            )
            np.testing.assert_allclose(actual, expected.cpu().numpy(), rtol=2e-3, atol=2e-4)
        if output[0].shape != (1, 64, height // 8, width // 8):
            raise RuntimeError(f"Unexpected descriptor map shape: {output[0].shape}")
        if output[1].shape != (1, 65, height // 8, width // 8):
            raise RuntimeError(f"Unexpected keypoint map shape: {output[1].shape}")
        if output[2].shape != (1, 1, height // 8, width // 8):
            raise RuntimeError(f"Unexpected reliability map shape: {output[2].shape}")

    matcher_module = importlib.import_module("modules.lighterglue")
    lightglue_module = importlib.import_module("kornia.feature.lightglue")
    matcher_module.LighterGlue.default_conf_xfeat["flash"] = False

    def export_rotate_half(value):
        original_shape = tuple(value.shape)
        half = original_shape[-1] // 2
        value = value.reshape(*original_shape[:-1], half, 2)
        first, second = value.split(1, dim=-1)
        first = first.reshape(*original_shape[:-1], half)
        second = second.reshape(*original_shape[:-1], half)
        return torch.stack((-second, first), dim=-1).reshape(original_shape)

    def export_sigmoid_log_double_softmax(similarity, certainty0, certainty1):
        batch, count0, count1 = similarity.shape
        certainties = (
            F.logsigmoid(certainty0)
            + F.logsigmoid(certainty1).permute(0, 2, 1)
        )
        scores0 = F.log_softmax(similarity, dim=2)
        scores1 = F.log_softmax(
            similarity.permute(0, 2, 1).contiguous(), dim=2
        ).permute(0, 2, 1)
        scores = similarity.new_full((batch, count0 + 1, count1 + 1), 0)
        scores[:, :count0, :count1] = scores0 + scores1 + certainties
        scores[:, :-1, -1] = F.logsigmoid(-certainty0.reshape(batch, count0))
        scores[:, -1, :-1] = F.logsigmoid(-certainty1.reshape(batch, count1))
        return scores

    def export_self_block_forward(self, value, encoding, mask=None):
        batch, count, _ = value.shape
        qkv = self.Wqkv(value).reshape(
            batch, count, self.num_heads, self.head_dim, 3
        )
        query, key, content = qkv.split(1, dim=-1)
        query = query.reshape(batch, count, self.num_heads, self.head_dim)
        key = key.reshape(batch, count, self.num_heads, self.head_dim)
        content = content.reshape(batch, count, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        content = content.permute(0, 2, 1, 3)
        query = lightglue_module.apply_cached_rotary_emb(encoding, query)
        key = lightglue_module.apply_cached_rotary_emb(encoding, key)
        similarity = torch.matmul(
            query, key.permute(0, 1, 3, 2)
        ) * (query.shape[-1] ** -0.5)
        if mask is not None:
            similarity = similarity.masked_fill(~mask, -float("inf"))
        attention = F.softmax(similarity, dim=3)
        context = torch.matmul(attention, content)
        if mask is not None:
            context = context.nan_to_num()
        message = self.out_proj(
            context.permute(0, 2, 1, 3).reshape(batch, count, self.embed_dim)
        )
        return value + self.ffn(torch.cat([value, message], dim=-1))

    def export_cross_block_forward(self, value0, value1, mask=None):
        batch0, count0, _ = value0.shape
        batch1, count1, _ = value1.shape
        query_key0 = self.to_qk(value0).reshape(
            batch0, count0, self.heads, -1
        ).permute(0, 2, 1, 3)
        query_key1 = self.to_qk(value1).reshape(
            batch1, count1, self.heads, -1
        ).permute(0, 2, 1, 3)
        content0 = self.to_v(value0).reshape(
            batch0, count0, self.heads, -1
        ).permute(0, 2, 1, 3)
        content1 = self.to_v(value1).reshape(
            batch1, count1, self.heads, -1
        ).permute(0, 2, 1, 3)
        query_key0 = query_key0 * self.scale ** 0.5
        query_key1 = query_key1 * self.scale ** 0.5
        similarity = torch.matmul(
            query_key0, query_key1.permute(0, 1, 3, 2)
        )
        if mask is not None:
            similarity = similarity.masked_fill(~mask, -float("inf"))
        attention01 = F.softmax(similarity, dim=3)
        attention10 = F.softmax(
            similarity.permute(0, 1, 3, 2).contiguous(), dim=3
        )
        message0 = torch.matmul(attention01, content1)
        message1 = torch.matmul(attention10, content0)
        message0 = self.to_out(
            message0.permute(0, 2, 1, 3).reshape(batch0, count0, -1)
        )
        message1 = self.to_out(
            message1.permute(0, 2, 1, 3).reshape(batch1, count1, -1)
        )
        if mask is not None:
            message0, message1 = message0.nan_to_num(), message1.nan_to_num()
        value0 = value0 + self.ffn(torch.cat([value0, message0], dim=-1))
        value1 = value1 + self.ffn(torch.cat([value1, message1], dim=-1))
        return value0, value1

    def export_match_assignment_forward(self, descriptors0, descriptors1):
        projected0 = self.final_proj(descriptors0)
        projected1 = self.final_proj(descriptors1)
        dimension = projected0.shape[-1]
        projected0 = projected0 / dimension ** 0.25
        projected1 = projected1 / dimension ** 0.25
        similarity = torch.matmul(projected0, projected1.permute(0, 2, 1))
        certainty0 = self.matchability(descriptors0)
        certainty1 = self.matchability(descriptors1)
        assignment = lightglue_module.sigmoid_log_double_softmax(
            similarity, certainty0, certainty1
        )
        return assignment, similarity

    def export_matcher_forward(self, data):
        data0, data1 = data["image0"], data["image1"]
        keypoints0, keypoints1 = data0["keypoints"], data1["keypoints"]
        size0, size1 = data0["image_size"], data1["image_size"]
        keypoints0 = lightglue_module.normalize_keypoints(keypoints0, size0).clone()
        keypoints1 = lightglue_module.normalize_keypoints(keypoints1, size1).clone()
        descriptors0 = data0["descriptors"].detach().contiguous()
        descriptors1 = data1["descriptors"].detach().contiguous()
        if torch.is_autocast_enabled():
            descriptors0 = descriptors0.half()
            descriptors1 = descriptors1.half()
        descriptors0 = self.input_proj(descriptors0)
        descriptors1 = self.input_proj(descriptors1)
        encoding0 = self.posenc(keypoints0)
        encoding1 = self.posenc(keypoints1)
        for layer_index in range(self.conf.n_layers):
            descriptors0, descriptors1 = self.transformers[layer_index](
                descriptors0, descriptors1, encoding0, encoding1
            )
        assignment, _ = self.log_assignment[layer_index](descriptors0, descriptors1)
        return {"log_assignment": assignment}

    lighter = LighterGlue(str(matcher_weights))
    matcher = lighter.net.cpu().eval()
    matcher.conf.width_confidence = -1
    matcher.conf.depth_confidence = -1
    matcher.conf.flash = False
    matcher.conf.mp = False
    for layer in matcher.transformers:
        layer.self_attn.inner_attn.enable_flash = False
        layer.self_attn.inner_attn.has_sdp = False

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
    with torch.no_grad():
        reference_assignment = wrapper(*sample)

    lightglue_module.rotate_half = export_rotate_half
    lightglue_module.sigmoid_log_double_softmax = export_sigmoid_log_double_softmax
    lightglue_module.SelfBlock.forward = export_self_block_forward
    lightglue_module.CrossBlock.forward = export_cross_block_forward
    lightglue_module.MatchAssignment.forward = export_match_assignment_forward
    matcher._forward = types.MethodType(export_matcher_forward, matcher)
    with torch.no_grad():
        adapted_assignment = wrapper(*sample)
    torch.testing.assert_close(
        adapted_assignment, reference_assignment, rtol=2e-5, atol=2e-5
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
