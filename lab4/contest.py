"""Архитектуры и предсказания конкурсных ансамблей.

Обучение: train_course.py и train_external.py.
Оценка и пояснения: Task_4_Maksim_Kostromitin_Group_J4132.ipynb и Task_4_Maksim_Kostromitin_Group_J4132_external.ipynb.
"""

import numpy as np
import hw4

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageOps
from scipy.special import expit, logit
from torch import nn


import hashlib
import io
import json
import math
import os

import pandas as pd
import torch
from torch.nn import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_precision():
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def conv_norm(cin, cout, kernel=3, stride=1, groups=1, activation=True):
    layers = [
        nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DeepBasicBlock(nn.Module):

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(
            conv_norm(cin, cout, 3, stride), conv_norm(cout, cout, activation=False)
        )
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2, ceil_mode=True),
                conv_norm(cin, cout, 1, activation=False),
            )
        elif cin != cout:
            self.shortcut = conv_norm(cin, cout, 1, activation=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return torch.relu(self.shortcut(x) + self.body(x))


class DeepResidualNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            conv_norm(3, 24, 3, 2),
            conv_norm(24, 24),
            conv_norm(24, 40),
            nn.MaxPool2d(3, 2, 1),
        )
        layers = []
        cin = 40
        for i, (width, count) in enumerate([(40, 3), (80, 4), (160, 6), (320, 3)]):
            for j in range(count):
                layers.append(DeepBasicBlock(cin, width, 2 if i and j == 0 else 1))
                cin = width
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(320, 1))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)).mean((2, 3))).flatten()


def mobile_spatial_layers(cin, mid, expansion, stride, kernel):
    layers = []
    if expansion != 1:
        layers.extend(
            [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()]
        )
    layers.extend(
        [
            nn.Conv2d(mid, mid, kernel, stride, kernel // 2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        ]
    )
    return nn.Sequential(*layers)


class MobileBottleneck(nn.Module):

    def __init__(self, cin, cout, expansion, stride, kernel):
        super().__init__()
        mid = cin * expansion
        self.spatial = mobile_spatial_layers(cin, mid, expansion, stride, kernel)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, max(8, cin // 4), 1),
            nn.SiLU(),
            nn.Conv2d(max(8, cin // 4), mid, 1),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)
        )
        self.skip = stride == 1 and cin == cout

    def forward(self, x):
        z = self.spatial(x)
        z = self.output(z * self.gate(z))
        return x + z if self.skip else z


MOBILE_STAGES = [
    (24, 1, 1, 3, 1),
    (32, 4, 2, 3, 2),
    (48, 4, 2, 5, 2),
    (80, 4, 2, 3, 3),
    (112, 4, 1, 5, 2),
    (176, 6, 2, 5, 3),
    (256, 6, 1, 3, 1),
]


class MobileContestNet(nn.Module):

    def __init__(self):
        super().__init__()
        blocks = [nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU()]
        cin = 32
        for width, factor, stride, kernel, repeats in MOBILE_STAGES:
            for i in range(repeats):
                blocks.append(
                    MobileBottleneck(
                        cin, width, factor, stride if i == 0 else 1, kernel
                    )
                )
                cin = width
        blocks.extend(
            [nn.Conv2d(cin, 640, 1, bias=False), nn.BatchNorm2d(640), nn.SiLU()]
        )
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(640, 1))

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3))).flatten()


class GeMResNet(DeepResidualNet):
    """Residual CNN with average and generalized mean pooling."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(640, 1))

    def forward(self, x):
        features = self.stages(self.stem(x)).float()
        mean = features.mean((2, 3))
        gem = features.clamp_min(1e-05).pow(3).mean((2, 3)).pow(1 / 3)
        return self.head(torch.cat((mean, gem), 1)).flatten()


def compact_inputs(batch, device, size, rng=None):
    x = (
        batch.permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(127.5)
        .sub_(1)
    )
    if rng is not None:
        edge = x.shape[-1]
        h, w = rng.integers(round(edge * 0.744), edge + 1, 2)
        y, x0 = (rng.integers(edge + 1 - h), rng.integers(edge + 1 - w))
        x = x[:, :, y : y + h, x0 : x0 + w]
        flip = torch.rand((len(x), 1, 1, 1), device=device) < 0.5
        x = torch.where(flip, x.flip(-1), x)
        gain = 0.86 + 0.28 * torch.rand((len(x), 3, 1, 1), device=device)
        shift = (torch.rand((len(x), 1, 1, 1), device=device) - 0.5) * 0.18
        x = (x * gain + shift).clamp(-1, 1)
    return F.interpolate(
        x, (size, size), mode="bilinear", align_corners=False
    ).contiguous(memory_format=torch.channels_last)


LOC_EDGE = 256


def boxed_rgb(rgb, edge=LOC_EDGE):
    image = ImageOps.contain(rgb, (edge, edge), Image.Resampling.BICUBIC)
    w, h = image.size
    x, y = ((edge - w) // 2, (edge - h) // 2)
    array = np.pad(
        np.asarray(image), ((y, edge - h - y), (x, edge - w - x), (0, 0)), mode="edge"
    )
    return (array, (x, y, w, h))


@torch.no_grad()
def smooth_weights(average, model, decay):
    for old, new in zip(average.parameters(), model.parameters()):
        old.lerp_(new, 1 - decay)
    for old, new in zip(average.buffers(), model.buffers()):
        if old.is_floating_point():
            old.lerp_(new, 1 - decay)
        else:
            old.copy_(new)


class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, dim),
        )

    def forward(self, tokens):
        b, n, d = tokens.shape
        q, k, v = (
            self.qkv(self.norm(tokens))
            .view(b, n, 3, 4, d // 4)
            .permute(2, 0, 3, 1, 4)
            .unbind()
        )
        contextual = (
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        )
        tokens = tokens + self.out(contextual)
        return tokens + self.ff(tokens)


def context_features(self, x):
    local = self.base.stages[:13](self.base.stem(x))
    global_map = self.base.stages[13:](local)
    small = self.global_tokens(global_map)
    big = self.local(local)

    def tokens(z):
        return (z + self.pos(z)).flatten(2).transpose(1, 2)

    stack = torch.cat(
        [self.cls.expand(len(x), -1, -1), tokens(big), tokens(small)], dim=1
    )
    return self.blocks(stack)[:, 0]


def context_forward(self, x, embedding=False):
    summary = self.features(x)
    logits = self.classify(summary).flatten()
    if embedding:
        return (logits, F.normalize(self.project(summary).float(), dim=1))
    return logits


class Context(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = GeMResNet()
        self.local = nn.Conv2d(160, 128, 1)
        self.global_tokens = nn.Conv2d(320, 128, 1)
        self.pos = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.cls = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        self.blocks = nn.Sequential(Attention(128), Attention(128), nn.LayerNorm(128))
        self.classify = nn.Linear(128, 1)
        self.project = nn.Linear(128, 64)

    features = context_features
    forward = context_forward


@torch.no_grad()
def average(target, source, rate):
    for a, b in zip(target.parameters(), source.parameters()):
        a.lerp_(b, 1 - rate)
    for a, b in zip(target.buffers(), source.buffers()):
        a.copy_(b)


def augment(x, strong):

    def rand(c=1):
        return torch.rand(len(x), c, 1, 1, device=x.device)

    x = torch.where(rand() < 0.5, x.flip(-1), x)
    gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]
    x = gray + (x - gray) * (rand() * 1.7 if strong else 0.5 + rand())
    x = (x * (0.76 + 0.48 * rand(3)) + (rand() - 0.5) * 0.32).clamp(-1, 1)
    if strong:
        blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        x = x + (x - blur) * (rand() * 1.7 - 0.9)
        x = ((x + 1) * 0.5).clamp(0, 1).pow(0.62 + rand() * 0.91) * 2 - 1
    return x.contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict_component(model, frames, edge):
    model.eval()
    probs = []
    for i in range(0, len(frames), 28):
        x = compact_inputs(torch.from_numpy(frames[i : i + 28].copy()), DEVICE, edge)
        with inference_precision():
            a, b = (model(x).float(), model(x.flip(-1)).float())
        probs.extend(((a + b) * 0.5).sigmoid().cpu().tolist())
    return np.array(probs)


class FocusMobile(MobileContestNet):

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))

    def forward(self, x):
        features = self.features(x).float()
        mean = features.mean((2, 3))
        local = features.clamp_min(1e-05).pow(3).mean((2, 3)).pow(1 / 3)
        return self.head(torch.cat((mean, local), dim=1)).flatten()


def build_detector_stages(model, widths, depths):
    model.stem = nn.Sequential(
        conv_norm(3, 24, 3, 2),
        conv_norm(24, 24),
        conv_norm(24, widths[0]),
        nn.MaxPool2d(3, 2, 1),
    )
    model.stages = nn.ModuleList()
    cin = widths[0]
    for index, (width, depth) in enumerate(zip(widths, depths)):
        layers = []
        for j in range(depth):
            layers.append(DeepBasicBlock(cin, width, 2 if index and j == 0 else 1))
            cin = width
        model.stages.append(nn.Sequential(*layers))


def detector_features(self, x):
    x = self.stem(x)
    for stage in self.stages[:3]:
        x = stage(x)
    return (x, self.stages[3](x))


def detector_global_head(self, deep):
    f = deep.float()
    g = f.clamp_min(1e-05).pow(3).mean((2, 3)).pow(1 / 3)
    return self.head(torch.cat((f.mean((2, 3)), g), dim=1)).flatten()


def detector_both(self, x):
    mid, deep = self.features(x)
    local = self.local(
        self.lateral(mid)
        + F.interpolate(self.top(deep), size=mid.shape[-2:], mode="nearest")
    )
    return (self.global_head(deep), self.heat(local), self.box(local))


class AnimalLocationNet(nn.Module):

    def __init__(self, small=False):
        super().__init__()
        widths = (28, 56, 112, 224) if small else (40, 80, 160, 320)
        depths = (2, 3, 4, 2) if small else (3, 4, 6, 3)
        build_detector_stages(self, widths, depths)
        self.lateral = conv_norm(widths[2], 80, 1)
        self.top = conv_norm(widths[3], 80, 1)
        self.local = conv_norm(80, 80)
        self.heat = nn.Conv2d(80, 2, 1)
        self.box = nn.Conv2d(80, 4, 1)
        self.head = nn.Sequential(nn.Dropout(0.18), nn.Linear(2 * widths[3], 1))
        nn.init.constant_(self.heat.bias, -2.2)
        with torch.no_grad():
            self.box.bias[2:].fill_(-1.0)

    features = detector_features
    global_head = detector_global_head

    def forward(self, x):
        _, deep = self.features(x)
        return self.global_head(deep)

    both = detector_both


@torch.inference_mode()
def context_scores(model, photos, device, size, turns=0):
    rows = []
    for batch in photos.split(64):
        x = compact_inputs(batch, device, size).rot90(turns, (-2, -1))
        rows.append(
            ((model(x).sigmoid() + model(x.flip(-1)).sigmoid()) / 2).cpu().numpy()
        )
    return np.concatenate(rows)


class ConvNeXtBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.spatial = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.mlp = nn.Sequential(
            nn.LayerNorm(c), nn.Linear(c, 3 * c), nn.GELU(), nn.Linear(3 * c, c)
        )
        self.scale = nn.Parameter(torch.full((c,), 0.1))

    def forward(self, x):
        z = self.spatial(x).permute(0, 2, 3, 1)
        return x + (self.mlp(z) * self.scale).permute(0, 3, 1, 2)


class CourseConvNeXt(nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        cin = 3
        for i, (c, depth) in enumerate(((48, 2), (96, 3), (192, 5), (320, 2))):
            stride = 4 if i == 0 else 2
            layers.extend(
                (
                    nn.Conv2d(cin, c, stride, stride),
                    nn.GroupNorm(1, c),
                    *(ConvNeXtBlock(c) for _ in range(depth)),
                )
            )
            cin = c
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(640), nn.Dropout(0.15), nn.Linear(640, 1)
        )

    def forward(self, x):
        z = self.layers(x).float()
        return self.head(
            torch.cat([z.mean((2, 3)), F.adaptive_max_pool2d(z, 1).flatten(1)], 1)
        ).flatten()


NATIVE_GRID = 12


def native_head():
    return nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, 192),
        nn.GELU(),
        nn.Dropout(0.22),
        nn.Linear(192, 1),
    )


def native_maps(self, x):
    r = self.res.stages[:13](self.res.stem(x))
    r2 = self.res.stages[13:](r)
    c = self.conv.layers[:-4](x)
    c2 = self.conv.layers[-4:](c)

    def dense(a, b):
        a = F.adaptive_avg_pool2d(a, (NATIVE_GRID, NATIVE_GRID))
        b = F.adaptive_avg_pool2d(b, (NATIVE_GRID, NATIVE_GRID))
        return torch.cat([a, b], 1).flatten(2).transpose(1, 2).float()

    return (dense(r, r2), dense(c, c2))


def native_encode(self, x):
    return torch.cat(
        [
            (feat - self.offset[k, : feat.shape[-1]])
            @ self.components[k, : feat.shape[-1]]
            for k, feat in enumerate(self.maps(x))
        ],
        dim=-1,
    )


def native_classify(self, features):
    z = self.local(features.float())
    q = self.queries.expand(len(z), -1, -1)
    pooled, _ = self.attention(q, z, z, need_weights=False)
    summary = torch.cat([pooled.flatten(1), z.mean(1)], 1)
    return (
        self.head(summary).flatten()
        + self.site(z).float().logsumexp(1).flatten()
        - math.log(z.shape[1])
    )


class Native(nn.Module):

    def __init__(self):
        super().__init__()
        self.res = GeMResNet()
        self.conv = CourseConvNeXt()
        self.register_buffer("offset", torch.zeros(2, 512))
        self.register_buffer("components", torch.zeros(2, 512, 64))
        self.register_buffer("widths", torch.tensor([480, 512]))
        self.local = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 160), nn.GELU(), nn.Linear(160, 128)
        )
        self.queries = nn.Parameter(torch.randn(5, 128) * 0.035)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.site = nn.Linear(128, 1)
        self.head = native_head()

    maps = native_maps
    encode = native_encode
    classify = native_classify

    def forward(self, x):
        return self.classify(self.encode(x))


NATIVE_EDGE = 448


def native_rgb(item):

    def convert(im):
        return boxed_rgb(ImageOps.exif_transpose(im).convert("RGB"), NATIVE_EDGE)[0]

    if isinstance(item, tuple):
        z, key = item
        with Image.open(io.BytesIO(z.read(key))) as im:
            return convert(im)
    with Image.open(item) as im:
        return convert(im)


@torch.inference_mode()
def native_batches(model, items, projection=True):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(items), 9):
            pixels = np.stack(list(pool.map(native_rgb, items[i : i + 9])))
            x = compact_inputs(torch.from_numpy(pixels), DEVICE, NATIVE_EDGE)
            with inference_precision():
                if projection:
                    yield model.encode(x).float()
                else:
                    yield model.maps(x)


@torch.inference_mode()
def native_infer(model, files):
    model.eval()
    p = []
    for z in native_batches(model, files):
        with inference_precision():
            p.extend(model.classify(z).float().sigmoid().cpu().tolist())
    return np.array(p)


def hybrid_conv(cin, cout, kernel, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(),
    )


class HybridBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.local = nn.Sequential(
            hybrid_conv(dim, dim, 5, groups=dim),
            hybrid_conv(dim, dim * 2, 1),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.local(x)


def hybrid_stem():
    return nn.Sequential(
        hybrid_conv(3, 32, 3, 2),
        hybrid_conv(32, 48, 3, 2),
        HybridBlock(48),
        hybrid_conv(48, 96, 3, 2),
        HybridBlock(96),
        HybridBlock(96),
        hybrid_conv(96, 192, 3, 2),
        HybridBlock(192),
        HybridBlock(192),
        HybridBlock(192),
    )


def hybrid_forward(self, x, teacher=False):
    local = self.stem(x)
    tokens = (local + self.pos(local)).flatten(2).transpose(1, 2)
    z = self.global_layers(torch.cat([self.cls.expand(len(x), -1, -1), tokens], 1))
    shared = self.representation(
        torch.cat([z[:, 0], z[:, 1:].mean(1), local.amax((2, 3))], 1)
    )
    score = self.species(shared).flatten()
    return (score, self.features(shared).float()) if teacher else score


class Hybrid(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = hybrid_stem()
        self.pos = nn.Conv2d(192, 192, 3, padding=1, groups=192)
        self.cls = nn.Parameter(torch.randn(1, 1, 192) * 0.02)
        self.global_layers = nn.Sequential(
            *[Attention(192) for _ in range(4)], nn.LayerNorm(192)
        )
        self.representation = nn.Sequential(
            nn.LayerNorm(576), nn.Linear(576, 384), nn.GELU()
        )
        self.species = nn.Linear(384, 1)
        self.features = nn.Linear(384, 560)

    forward = hybrid_forward


def taxonomy_forward(self, x, aux=False):
    local = self.base.stages[:13](self.base.stem(x))
    glob = self.base.stages[13:](local)

    def seq(z):
        return (z + self.pos(z)).flatten(2).transpose(1, 2)

    tokens = torch.cat(
        [
            self.cls.expand(len(x), -1, -1),
            seq(self.local(local)),
            seq(self.global_tokens(glob)),
        ],
        1,
    )
    h = self.blocks(tokens)[:, 0]
    logits = self.classify(h).flatten()
    if aux:
        return (logits, self.fine(h))
    return logits


class Taxonomy(Context):

    def __init__(self):
        super().__init__()
        self.base.head = nn.Identity()
        self.fine = nn.Linear(128, 37 + 120 + 57)

    forward = taxonomy_forward


def spectrum_forward(self, x, attributes=False):
    lum = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]
    band = []
    for stride in (1, 3):
        smooth = F.avg_pool2d(
            F.pad(lum, (stride, stride, stride, stride), mode="reflect"),
            2 * stride + 1,
            stride=1,
        )
        padded = F.pad(smooth, (stride, stride, stride, stride), mode="reflect")
        gx = (
            padded[:, :, stride:-stride, 2 * stride :]
            - padded[:, :, stride:-stride, : -2 * stride]
        )
        gy = (
            padded[:, :, 2 * stride :, stride:-stride]
            - padded[:, :, : -2 * stride, stride:-stride]
        )
        band.extend([gx, gy, lum - smooth])
    x = x + self.adapter(torch.cat(band, 1))
    z = self.base.stages(self.base.stem(x)).flatten(2).float()
    feat = torch.cat([z.mean(-1), z.clamp_min(1e-05).pow(3).mean(-1).pow(1 / 3)], 1)
    binary = self.base.head(feat).flatten()
    if attributes:
        return (binary, self.attribute(feat))
    return binary


class Spectrum(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = GeMResNet()
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.zeros_(self.adapter.weight)
        self.attribute = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, 15))

    forward = spectrum_forward


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


ROUND2_NAMES = ("deep_resnet", "highres", "mbconv")


def checkpoint_for(name, folder):
    return folder / f"{name}_best.pt"


def new_contest_model(name):
    return {"deep_resnet": DeepResidualNet, "mbconv": MobileContestNet}[name]()


def round2_load(name, folder, device):
    ck = torch.load(checkpoint_for(name, folder), weights_only=True, map_location="cpu")
    arch = ck["parent"] if name == "highres" else name
    model = new_contest_model(arch)
    model.load_state_dict(ck["state_dict"])
    edge = 256 if name == "highres" else 224
    return (
        model.eval().to(device, memory_format=torch.channels_last),
        ck["size"],
        edge,
    )


CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def resized_photos(root, paths, edge=160):

    def load(path):
        with Image.open(Path(root) / path) as image:
            return np.asarray(
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize((edge, edge), Image.Resampling.BILINEAR)
            )

    images = np.empty((len(paths), edge, edge, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, image in enumerate(pool.map(load, paths)):
            images[i] = image
    return torch.from_numpy(images)


def aspect_crops(root, paths, edge):

    def read(path):
        with Image.open(Path(root) / path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(
                ImageOps.fit(rgb, (edge, edge), method=Image.Resampling.BILINEAR)
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        return torch.from_numpy(np.stack(list(pool.map(read, paths))))


@torch.inference_mode()
def compact_scores(net, images, device, size=160):
    net.eval()
    scores = []
    for batch in images.split(64):
        x = compact_inputs(batch, device, size)
        scores.append(
            ((net(x).sigmoid() + net(x.flip(-1)).sigmoid()) / 2).cpu().numpy()
        )
    return np.concatenate(scores)


def contest_view(model, images, device, size, view):
    if view == "centre":
        edge = images.shape[1]
        cut = round(0.07 * edge)
        images = images[:, cut:-cut, cut:-cut, :]
    if view == "large":
        size = round(size * 1.2)
    return compact_scores(model, images, device, size)


def round2_model_views(name, root, paths, folder, device, views):
    model, size, edge = round2_load(name, folder, device)
    return_views = sorted(set(views))
    if set(return_views) - {"full", "centre", "large", "aspect", *CORNERS}:
        raise ValueError(views)
    predictions = {view: [] for view in return_views}
    for start in range(0, len(paths), 384):
        rows = paths[start : start + 384]
        images = resized_photos(root, rows, edge)
        for view in return_views:
            if view == "aspect":
                square = aspect_crops(root, rows, edge)
                p = compact_scores(model, square, device, size)
            elif view in CORNERS:
                side = round(0.82 * edge)
                y = edge - side if view.startswith("bottom") else 0
                x = edge - side if view.endswith("right") else 0
                p = compact_scores(
                    model, images[:, y : y + side, x : x + side, :], device, size
                )
            else:
                p = contest_view(model, images, device, size, view)
            predictions[view].append(p)
    return {v: np.concatenate(chunks) for v, chunks in predictions.items()}


def average_recipe(spec, parts, pooling="probability"):
    per_model = [
        np.mean([parts[name][v] for v in views], axis=0) for name, views in spec.items()
    ]
    if pooling == "log_odds":
        p = np.clip(np.stack(per_model).astype(float), 1e-06, 1 - 1e-06)
        return 1 / (1 + np.exp(-(np.log(p) - np.log1p(-p)).mean(axis=0)))
    assert pooling == "probability"
    return np.mean(per_model, axis=0)


def predict_round2(root, paths, folder, device, record):
    spec = record["members"]
    assert (
        record["selection_split"] == "val"
        and record["threshold"] == 0.5
        and (set(spec) == set(record["checkpoints"]))
    )
    parts = {}
    for name, views in spec.items():
        assert name in ROUND2_NAMES
        assert (
            sha256(checkpoint_for(name, folder).read_bytes()).hexdigest()
            == record["checkpoints"][name]
        )
        parts[name] = round2_model_views(name, root, paths, folder, device, views)
    return average_recipe(spec, parts, record["pooling"])


def context_model(name):
    if name != "context_resnet":
        raise ValueError(name)
    return DeepResidualNet()


ROUND3_EDGE = 224


def context_photos(root, paths, edge=ROUND3_EDGE, geometry="box"):

    def read(path):
        with Image.open(Path(root) / path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            if geometry == "stretch":
                return np.asarray(rgb.resize((edge, edge), Image.Resampling.BILINEAR))
            if geometry == "crop":
                return np.asarray(
                    ImageOps.fit(rgb, (edge, edge), Image.Resampling.BICUBIC)
                )
            resized = np.asarray(
                ImageOps.contain(rgb, (edge, edge), Image.Resampling.BICUBIC)
            )
            h, w = resized.shape[:2]
            pad_h, pad_w = ((edge - h) // 2, (edge - w) // 2)
            return np.pad(
                resized,
                ((pad_h, edge - h - pad_h), (pad_w, edge - w - pad_w), (0, 0)),
                mode="edge",
            )

    result = np.empty((len(paths), edge, edge, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, image in enumerate(pool.map(read, paths)):
            result[index] = image
    return torch.from_numpy(result)


def context_view_scores(
    name, root, paths, folder, device, views, factory=context_model, edge=ROUND3_EDGE
):
    ck = torch.load(folder / f"{name}_best.pt", map_location="cpu", weights_only=True)
    model = factory(name)
    assert ck["architecture"] == name
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device, memory_format=torch.channels_last)
    geometry_order = sorted({geometry for geometry, _ in views})
    results = {view: [] for view in views}
    for start in range(0, len(paths), 384):
        rows = paths[start : start + 384]
        for geometry in geometry_order:
            photos = context_photos(root, rows, edge=edge, geometry=geometry)
            for view in views:
                if view[0] == geometry:
                    results[view].append(
                        context_scores(model, photos, device, ck["size"], view[1])
                    )
    return {view: np.concatenate(chunks) for view, chunks in results.items()}


def combine_context(predictions, weights, pooling):
    if pooling == "probability":
        return np.average(predictions, weights=weights, axis=0)
    assert pooling == "log_odds"
    p = np.clip(np.asarray(predictions, dtype=np.float64), 1e-06, 1 - 1e-06)
    logits = np.log(p) - np.log1p(-p)
    return 1 / (1 + np.exp(-np.average(logits, weights=weights, axis=0)))


def predict_context(root, paths, folder, previous_folder, record, device):
    assert record["split"] == "val" and record["threshold"] == 0.5
    probabilities = []
    for member in record["members"]:
        name = member["model"]
        if name == "round2":
            file = previous_folder / "selection.json"
            assert (
                hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"][name]
            )
            prior = json.loads(file.read_text())
            probabilities.append(
                predict_round2(root, paths, previous_folder, device, prior)
            )
        else:
            file = folder / f"{name}_best.pt"
            assert (
                hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"][name]
            )
            views = [tuple(view) for view in member["views"]]
            parts = context_view_scores(name, root, paths, folder, device, views)
            probabilities.append(np.mean([parts[view] for view in views], axis=0))
    return combine_context(probabilities, record["weights"], record["pool"])


LABEL_RUNS = ("label_control",)


def label_model(name):
    if name not in LABEL_RUNS:
        raise ValueError(name)
    return GeMResNet()


LABEL_CONFIG = {"label_control": {"edge": 256}}


def label_scores(name, root, paths, folder, device, views):
    pieces = context_view_scores(
        name,
        root,
        paths,
        folder,
        device,
        [tuple(view) for view in views],
        factory=label_model,
        edge=LABEL_CONFIG[name]["edge"],
    )
    return np.mean(list(pieces.values()), axis=0)


def predict_labels(root, paths, folder, prior_folder, round2_folder, record, device):
    assert record["split"] == "val" and record["threshold"] == 0.5
    predictions = []
    for member in record["members"]:
        name = member["model"]
        if name == "round3":
            file = prior_folder / "selection.json"
            assert (
                hashlib.sha256(file.read_bytes()).hexdigest() == record["files"][name]
            )
            previous = json.loads(file.read_text())
            score = predict_context(
                root, paths, prior_folder, round2_folder, previous, device
            )
        else:
            file = folder / f"{name}_best.pt"
            assert (
                hashlib.sha256(file.read_bytes()).hexdigest() == record["files"][name]
            )
            score = label_scores(name, root, paths, folder, device, member["views"])
        predictions.append(score)
    return combine_context(predictions, record["weights"], record["pooling"])


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trained_model(weights, name, lock):
    path = weights / name
    if file_hash(path) != lock["checkpoints"][name]:
        raise ValueError(f"Checkpoint hash differs: {name}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = GeMResNet().eval().to(DEVICE, memory_format=torch.channels_last)
    model.load_state_dict(ck["state_dict"])
    return (model, ck["size"])


MASKED_EDGE = 320


@torch.inference_mode()
def masked_predict(model, a, size=MASKED_EDGE):
    parts = []
    model.eval()
    for start in range(0, len(a), 16):
        x = compact_inputs(torch.from_numpy(a[start : start + 16].copy()), DEVICE, size)
        with inference_precision():
            prob = (
                model(x).float().sigmoid() + model(x.flip(-1)).float().sigmoid()
            ) / 2
        parts.extend(prob.cpu().tolist())
    return np.array(parts)


def blend_previous(left, right, weight, kind="prob"):
    if kind == "prob":
        return (1 - weight) * left + weight * right

    def logit(x):
        x = np.clip(x, 1e-06, 1 - 1e-06)
        return np.log(x / (1 - x))

    return 1 / (1 + np.exp(-((1 - weight) * logit(left) + weight * logit(right))))


def current_course(paths, original, weights, lock):
    name, weight, rule = lock["course"]["recipe"]
    if name == "incumbent":
        return original
    if name not in ("masked", "scale"):
        raise ValueError(f"Unsupported published course recipe: {name}")
    model, size = trained_model(weights, name + "_best.pt", lock)
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(
                boxed_rgb(ImageOps.exif_transpose(im).convert("RGB"), size)[0]
            )
    extra = masked_predict(model, np.stack(images), size)
    return blend_previous(original, extra, weight, rule)


def course_prior(paths, baseline, previous):
    folder = baseline / "prior"
    recipe = load_json(folder / "round4/selection.json")
    base = predict_labels(
        Path("."),
        paths,
        folder / "round4",
        folder / "round3",
        folder / "round2",
        recipe,
        DEVICE,
    )
    lock = load_json(previous / "selection.json")
    return (base, current_course(paths, base, previous, lock))


def external_model(name):
    if name == "coco_full":
        return GeMResNet()
    if name == "coco_focus":
        return FocusMobile()
    raise ValueError(name)


EXT_EDGE = 256


def previous_coco(paths, base, baseline):
    current = baseline / "current"
    lock = load_json(current / "selection.json")
    values = []
    for member in lock["members"]:
        name = member["model"]
        if name == "original":
            values.append(base)
            continue
        if file_hash(current / (name + "_best.pt")) != lock["sha256"][name]:
            raise ValueError(f"COCO checkpoint hash differs: {name}")
        views = [tuple(v) for v in member["views"]]
        parts = context_view_scores(
            name,
            Path("."),
            paths,
            current,
            DEVICE,
            views,
            factory=external_model,
            edge=EXT_EDGE,
        )
        values.append(np.mean([parts[v] for v in views], axis=0))
    return combine_context(values, lock["weights"], lock["pooling"])


LOC_CONFIG = ("joint", "detector")


def localization_model(name):
    assert name in LOC_CONFIG
    return AnimalLocationNet(small=name == "detector")


def load_location(name, folder, device):
    ck = torch.load(folder / f"{name}_best.pt", map_location="cpu", weights_only=True)
    assert ck["architecture"] == name
    model = localization_model(name)
    model.load_state_dict(ck["state_dict"], strict=True)
    return (model.eval().to(device, memory_format=torch.channels_last), ck["size"])


def locate_photos(root, paths):

    def open_one(name):
        with Image.open(Path(root) / name) as image:
            return boxed_rgb(ImageOps.exif_transpose(image).convert("RGB"))

    with ThreadPoolExecutor(max_workers=5) as pool:
        photos, geometry = zip(*pool.map(open_one, paths))
    return (torch.from_numpy(np.stack(photos)), geometry)


def decode_locations(heat, regression):
    heat = heat.float()
    n, _, h, w = heat.shape
    index = torch.arange(n, device=heat.device)
    per_class = heat.flatten(2).amax(2)
    idx = heat.amax(1).flatten(1).argmax(1)
    y, x = (idx // w, idx % w)
    p = regression[index, :, y, x].float()
    cx = (x + p[:, 0].sigmoid()) / w
    cy = (y + p[:, 1].sigmoid()) / h
    bw, bh = p[:, 2:].clamp(-3.5, 0.1).exp().unbind(dim=1)
    boxes = torch.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1
    ).clamp(0, 1)
    return (
        boxes,
        per_class.amax(1).sigmoid(),
        (per_class[:, 1] - per_class[:, 0]).sigmoid(),
    )


def bbox_iou(pred, target):
    lo = torch.maximum(pred[:, :2], target[:, :2])
    hi = torch.minimum(pred[:, 2:], target[:, 2:])
    overlap = (hi - lo).clamp_min(0.0).prod(dim=1)
    union = (
        (pred[:, 2:] - pred[:, :2]).prod(dim=1)
        + (target[:, 2:] - target[:, :2]).prod(dim=1)
        - overlap
    )
    return overlap / union.clamp_min(1e-06)


@torch.inference_mode()
def locate_pair(model, photos, device, size):
    returned = []
    for images in photos.split(64):
        x = compact_inputs(images, device, size)
        a, heat1, reg1 = model.both(x)
        b, heat2, reg2 = model.both(x.flip(-1))
        r, s, p = decode_locations(heat1, reg1)
        t, u, q = decode_locations(heat2, reg2)
        t = t[:, [2, 1, 0, 3]].clone()
        t[:, [0, 2]] = 1 - t[:, [0, 2]]
        agreement = bbox_iou(r, t)
        secure = agreement > 0.3
        best = torch.where((s >= u)[:, None], r, t)
        region = torch.where(secure[:, None], (r + t) * 0.5, best)
        quality = torch.maximum(s, u) * (0.5 + 0.5 * agreement)
        pairs = (region, quality, (p + q) * 0.5, (a.sigmoid() + b.sigmoid()) * 0.5)
        returned.append(tuple((z.cpu().numpy() for z in pairs)))
    return [np.concatenate(parts) for parts in zip(*returned)]


def convert_region(box, geometry, width, height):
    x, y, w, h = geometry
    b = (np.asarray(box) * LOC_EDGE - np.array([x, y, x, y])) / np.array([w, h, w, h])
    b = b.clip(0, 1)
    center = (b[:2] + b[2:]) / 2
    half = np.maximum((b[2:] - b[:2]) * 0.82, 0.18)
    rect = np.concatenate(((center - half).clip(0, 1), (center + half).clip(0, 1))) * [
        width,
        height,
        width,
        height,
    ]
    left, top, right, bottom = [int(round(coord)) for coord in rect]
    return (left, top, max(left + 1, right), max(top + 1, bottom))


def region_photos(root, paths, regions, geometry):

    def open_one(row):
        name, box, meta = row
        with Image.open(Path(root) / name) as photo:
            rgb = ImageOps.exif_transpose(photo).convert("RGB")
            region = convert_region(box, meta, *rgb.size)
            patch, _ = boxed_rgb(rgb.crop(region))
        return (patch, region)

    with ThreadPoolExecutor(max_workers=5) as pool:
        patches, corners = zip(*pool.map(open_one, zip(paths, regions, geometry)))
    return (torch.from_numpy(np.stack(patches)), np.asarray(corners))


def detector_proposals(group, root, joint, size_j, detector, size_d, focus, ck, device):
    images, geometry = locate_photos(root, group)
    r, _, _, _ = locate_pair(joint, images, device, size_j)
    t, _, _, _ = locate_pair(detector, images, device, size_d)
    j, _ = region_photos(root, group, r, geometry)
    d, xy = region_photos(root, group, t, geometry)
    p = context_scores(joint, j, device, size_j)
    q = context_scores(focus, d, device, ck["size"])
    return pd.DataFrame(
        dict(
            path=group,
            roi_pair=(p + q) / 2,
            x0=xy[:, 0],
            y0=xy[:, 1],
            x1=xy[:, 2],
            y1=xy[:, 3],
        )
    )


def route_evidence(root, paths, prior, local, basic, device):
    joint, size_j = load_location("joint", local, device)
    detector, size_d = load_location("detector", local, device)
    ck = torch.load(prior / "coco_focus_best.pt", weights_only=True, map_location="cpu")
    focus = (
        external_model("coco_focus")
        .eval()
        .to(device, memory_format=torch.channels_last)
    )
    focus.load_state_dict(ck["state_dict"])
    parts = []
    for start in range(0, len(paths), 320):
        group = paths[start : start + 320]
        parts.append(
            detector_proposals(
                group, root, joint, size_j, detector, size_d, focus, ck, device
            )
        )
    table = pd.concat(parts, ignore_index=True)
    table["previous"] = basic
    return table


def flag_scores(table):
    first = table.previous.to_numpy()
    margin = 1 - 2 * np.abs(first - 0.5)
    crop = table.roi_pair.to_numpy()
    disagreement = np.abs(crop - first)
    return {
        "Уверенность прежнего ансамбля": margin,
        "Уверенность и расхождение с вырезом": margin + 0.6 * disagreement,
    }


def specialist_model(name):
    if name == "pose_resnet":
        return GeMResNet()
    if name == "pose_mobile":
        return FocusMobile()
    raise ValueError(name)


POSE_EDGE = 256


def detected_crop_scores(name, root, rows, folder, device):
    ck = torch.load(folder / f"{name}_best.pt", weights_only=True, map_location="cpu")
    model = specialist_model(name).eval().to(device, memory_format=torch.channels_last)
    model.load_state_dict(ck["state_dict"])

    def crop(row):
        with Image.open(Path(root) / row.path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            region = rgb.crop((row.x0, row.y0, row.x1, row.y1))
            return boxed_rgb(region, POSE_EDGE)[0]

    with ThreadPoolExecutor(max_workers=6) as pool:
        pixels = torch.from_numpy(
            np.stack(list(pool.map(crop, rows.itertuples(index=False))))
        )
    return context_scores(model, pixels, device, ck["size"])


def specialist_scores(name, root, rows, folder, device, policy):
    paths = rows.path.tolist()
    views = sorted({tuple(v) for v in policy if v != "detected_crop"})
    parts = (
        context_view_scores(
            name,
            root,
            paths,
            folder,
            device,
            views,
            factory=specialist_model,
            edge=POSE_EDGE,
        )
        if views
        else {}
    )
    if "detected_crop" in policy:
        parts["detected_crop"] = detected_crop_scores(name, root, rows, folder, device)
    return np.mean(
        [parts[v if v == "detected_crop" else tuple(v)] for v in policy], axis=0
    )


def predict_specialist(root, location, route, folder, lock, device):
    base = location.previous.to_numpy(copy=True)
    subset = location.loc[route].reset_index(drop=True)
    spec = lock["recipe_config"]
    results = []
    cursor = 0
    for name in spec["names"]:
        if name == "previous":
            results.append(base[route])
        else:
            results.append(
                specialist_scores(
                    name, root, subset, folder, device, spec["views"][cursor]
                )
            )
            cursor += 1
    assert spec["minimum"] is None
    base[route] = combine_context(results, spec["weights"], spec["pooling"])
    return base


def crop_region(rgb, box, view):
    w, h = rgb.size
    if view == "full":
        return rgb
    left, top, right, bottom = box
    sx, sy = (right - left, bottom - top)
    cx, cy = ((right + left) / 2, (bottom + top) / 2)
    if view == "wide" or view == "contrast":
        scale = 1.55
    elif view in ("detail_left", "detail_right"):
        scale = 0.7
        cx += (0.14 if view == "detail_right" else -0.14) * sx
        cy -= 0.09 * sy
    else:
        scale = 0.72 if view == "tight" else 1.0
    x0, y0 = (max(0, round(cx - scale * sx / 2)), max(0, round(cy - scale * sy / 2)))
    enhanced = rgb.crop(
        (
            x0,
            y0,
            max(x0 + 1, min(w, round(cx + scale * sx / 2))),
            max(y0 + 1, min(h, round(cy + scale * sy / 2))),
        )
    )
    return ImageOps.autocontrast(enhanced, cutoff=1) if view == "contrast" else enhanced


MODELS = {
    "pose_resnet": ("specialist", specialist_model),
    "pose_mobile": ("specialist", specialist_model),
    "coco_focus": ("current", external_model),
}


def view_pixels(flagged, root, view):
    batch = []
    for row in flagged.itertuples():
        with Image.open(root / row.path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        region = crop_region(rgb, (row.x0, row.y0, row.x1, row.y1), view)
        batch.append(boxed_rgb(region, 288)[0])
    return np.stack(batch)


def view_predictions(model, a, device, size):
    pred = []
    for start in range(0, len(a), 8):
        x = compact_inputs(torch.from_numpy(a[start : start + 8]), device, size)
        pred.extend(
            ((model(x).float().sigmoid() + model(x.flip(-1)).float().sigmoid()) / 2)
            .cpu()
            .tolist()
        )
    return pred


@torch.inference_mode()
def predict_views(frame, root, device, models, views, weights):
    flagged = frame.loc[frame.flagged].reset_index(drop=True)
    out = flagged[["path"]].copy()
    pixels = {}
    for view in views:
        pixels[view] = view_pixels(flagged, root, view)
    for name in models:
        folder, factory = MODELS[name]
        state = torch.load(
            weights / folder / (name + "_best.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model = factory(name).to(device, memory_format=torch.channels_last).eval()
        model.load_state_dict(state["state_dict"])
        for view in views:
            out[f"{name}_{view}"] = view_predictions(
                model, pixels[view], device, state["size"]
            )
    return out


CROP_SIZE = 288


def val_pixels(rows, root):
    result = {}
    for view in ("full", "wide", "tight"):
        batch = []
        for row in rows.itertuples():
            with Image.open(root / row.path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
            batch.append(
                boxed_rgb(
                    crop_region(rgb, (row.x0, row.y0, row.x1, row.y1), view), CROP_SIZE
                )[0]
            )
        result[view] = np.stack(batch)
    return result


@torch.inference_mode()
def crop_predict(model, pixels):
    model.eval()
    parts = []
    for a in pixels.values():
        probs = []
        for start in range(0, len(a), 12):
            x = compact_inputs(
                torch.from_numpy(a[start : start + 12]), DEVICE, CROP_SIZE
            )
            with inference_precision():
                p = (
                    model(x).float().sigmoid() + model(x.flip(-1)).float().sigmoid()
                ) / 2
            probs.extend(p.cpu().tolist())
        parts.append(probs)
    return np.mean(parts, axis=0)


def specialist_vote(rows, flags, base, course, baseline, previous, lock):
    _, names, views, weight = lock["external"]["recipe"]
    scores = predict_views(rows, Path("."), torch.device("cpu"), names, views, baseline)
    votes = scores.drop(columns="path").mean(axis=1).to_numpy()
    model, _ = trained_model(previous, "new_specialist_best.pt", lock)
    crops = val_pixels(rows[flags], Path("."))
    other = np.mean([votes, crop_predict(model, crops), course[flags]], axis=0)
    base[flags] = blend_previous(base[flags], other, weight)
    return base


def external_prior(paths, course_base, course, baseline, previous):
    coco = previous_coco(paths, course_base, baseline)
    recipe = load_json(baseline / "specialist/selection.json")
    rows = route_evidence(
        Path("."), paths, baseline / "current", baseline / "localize", coco, DEVICE
    )
    method = load_json(baseline / "localize/selection.json")["alert"]["method"]
    flags = flag_scores(rows)[method] >= recipe["cutoff"]
    rows["flagged"] = flags
    if not flags.any():
        return coco
    base = predict_specialist(
        Path("."), rows, flags, baseline / "specialist", recipe, DEVICE
    )
    return specialist_vote(
        rows,
        flags,
        base,
        course,
        baseline,
        previous,
        load_json(previous / "selection.json"),
    )


def frozen_tree_delta(X, trees):
    out = np.zeros(len(X))
    for t in trees:
        index = np.zeros(len(X), dtype=int)
        for _ in range(2):
            a = np.array(t["left"])[index]
            b = np.array(t["right"])[index]
            split = np.array(t["var"])[index].clip(0)
            side = X[np.arange(len(X)), split] <= np.asarray(t["at"])[index]
            index = np.where(a < 0, index, np.where(side, a, b))
        out += np.asarray(t["value"])[index]
    return out


def route_features(p, image):
    z = np.clip(logit(p.clip(1e-05, 1 - 1e-05)), -8, 8)
    delta = z[:, 1:] - z[:, :1]
    return np.column_stack(
        [
            z,
            delta,
            abs(delta),
            z.std(1),
            np.mean(z[:, 1:] * z[:, :1] < 0, axis=1),
            image,
        ]
    )


def apply_frozen_trees(p, image, trees):
    return expit(
        logit(p[:, 0].clip(1e-05, 1 - 1e-05))
        + frozen_tree_delta(route_features(p, image), trees).clip(-3, 3)
    )


def colour_and_texture(pix, ratio, scale):
    gray = pix @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = pix.max(2) - pix.min(2)
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    lap = np.diff(
        np.pad(gray, ((0, 0), (1, 1)), mode="reflect"), n=2, axis=1
    ) + np.diff(np.pad(gray, ((1, 1), (0, 0)), mode="reflect"), n=2, axis=0)
    feat = [ratio, scale, *pix.mean((0, 1)), *pix.std((0, 1))]
    for m in (gray, sat, edge, np.abs(lap)):
        center = m[16:48, 16:48]
        feat.extend(
            [
                float(m.mean()),
                float(m.std()),
                *np.quantile(m, [0.1, 0.5, 0.9]),
                float(center.mean() - m.mean()),
            ]
        )
    feat.extend(edge.reshape(4, 16, 4, 16).mean((1, 3)).flatten())
    return feat


def photo_stats(file):
    with Image.open(file) as src:
        rgb = ImageOps.exif_transpose(src).convert("RGB")
        ratio = np.log(rgb.width / rgb.height)
        scale = np.log1p(rgb.width * rgb.height)
        pix = (
            np.asarray(
                rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32
            )
            / 255
        )
    return colour_and_texture(pix, ratio, scale)


def appearance(files):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return np.array(list(pool.map(photo_stats, files)))


def blend(base, component, weight, mode):
    if mode == "prob":
        return (1 - weight) * base + weight * component

    def odds(p):
        return logit(np.clip(p, 1e-06, 1 - 1e-06))

    return expit((1 - weight) * odds(base) + weight * odds(component))


def arbitrate(base, columns, confidence, anchor_limit):
    group = np.column_stack(columns)
    alternative = group.mean(axis=1)
    sure = (group.min(axis=1) >= confidence) | (group.max(axis=1) <= 1 - confidence)
    opposed = (alternative >= 0.5) != (base >= 0.5)
    eligible = np.maximum(base, 1 - base) <= anchor_limit
    return np.where(sure & opposed & eligible, 0.2 * base + 0.8 * alternative, base)


def read_frames(paths):
    frames = []
    for path in paths:
        with Image.open(path) as photo:
            rgb = ImageOps.exif_transpose(photo).convert("RGB")
            frames.append(boxed_rgb(rgb, 320)[0])
    return np.stack(frames)


def context_network():
    model = Context()
    model.base.head = nn.Identity()
    return model


COMPONENTS = {
    "course_context": context_network,
    "external_context": context_network,
    "external_stable": context_network,
    "external_hybrid": Hybrid,
    "external_native": Native,
    "external_joint": Taxonomy,
    "external_spectrum": Spectrum,
}


def component_probability(name, lock, folder, paths, frames):
    path = folder / (name + ".pt")
    if hw4.checksum(path) != lock["hashes"][name]:
        raise ValueError(f"Не совпали веса {name}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = COMPONENTS[name]().to(DEVICE, memory_format=torch.channels_last)
    model.load_state_dict(state["weights"])
    if name == "external_native":
        return native_infer(model, paths)
    return predict_component(model, frames, state["edge"])


def route_base(paths, folder, prior, course, components, rule):
    inputs = np.column_stack([prior, *(components[k] for k in rule["group"]), course])
    corrected = apply_frozen_trees(inputs, appearance(paths), rule["trees"])
    return blend(prior, corrected, rule["weight"], rule["mode"])


def score_external(paths, folder):
    previous = folder / "previous"
    baseline = previous / "baseline"
    initial, course = course_prior(paths, baseline, previous)
    prior = external_prior(paths, initial, course, baseline, previous)
    rule = load_json(folder / "external_route_lock.json")
    final = load_json(folder / "external_arbitration_lock.json")
    frames = read_frames(paths)
    scores = {
        name: component_probability(name, rule, folder, paths, frames)
        for name in rule["group"]
    }
    anchor = route_base(paths, folder, prior, course, scores, rule)
    scores["external_hybrid"] = component_probability(
        "external_hybrid", final, folder, paths, frames
    )
    return arbitrate(
        anchor,
        [scores[k] for k in final["group"]],
        final["confidence"],
        final["anchor_limit"],
    )


def predict_photos(paths, branch):
    scorer = {"course": score_course, "external": score_external}[branch]
    if hw4.RESULTS is None:
        hw4.prepare()
    folder = hw4.open_models("winners")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    files = [str(Path(path).resolve()) for path in paths]
    chunks = []
    for start in range(0, len(files), 64):
        chunks.extend(scorer(files[start : start + 64], folder))
        done = min(start + 64, len(files))
        if len(files) > 64 and (done % 512 == 0 or done == len(files)):
            print(f"{done} / {len(files)}", flush=True)
    probabilities = np.array(chunks)
    return pd.DataFrame(
        {
            "path": files,
            "prob_dog": probabilities,
            "animal": np.where(probabilities >= 0.5, "dog", "cat"),
        }
    )


def test_from_images(branch):
    if hw4.MANIFESTS is None:
        hw4.prepare()
    rows = (
        pd.read_csv(hw4.MANIFESTS / "course_all.csv")
        .query("split == 'test'")
        .sort_values("relative_path")
    )
    folder = hw4.extract_photos(split="test")
    paths = [folder / key for key in rows.relative_path]
    result = predict_photos(paths, branch)
    if list(map(Path, result.path)) != paths:
        raise ValueError("Порядок предсказаний отличается от списка фотографий.")
    return result.assign(
        path=rows.relative_path.to_numpy(), label=rows.label.to_numpy()
    )


def score_course(paths, folder):
    previous = folder / "previous"
    _, base = course_prior(paths, previous / "baseline", previous)
    cfg = load_json(folder / "course_family_lock.json")
    context = component_probability(
        "course_context", cfg, folder, paths, read_frames(paths)
    )
    return (1 - cfg["weight"]) * base + cfg["weight"] * context
