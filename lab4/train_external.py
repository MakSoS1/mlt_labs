"""Повторное обучение моделей с внешними данными и их родительских сетей.

Примеры:
    python train_external.py --stages hybrid native
    python train_external.py --stages --parents course_hybrid course_convnext
"""

import argparse
import copy
import hashlib
import math
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

import hw4
from contest import (
    AnimalLocationNet,
    Context,
    CourseConvNeXt,
    FocusMobile,
    GeMResNet,
    Hybrid,
    Native,
    Spectrum,
    Taxonomy,
    augment,
    average,
    boxed_rgb,
    compact_inputs,
    native_batches,
    predict_component,
    smooth_weights,
)


RETRAIN_EXTERNAL = False


EXTERNAL_STAGES = {
    "coco_full",
    "coco_focus",
    "detector",
    "joint_location",
    "pose_resnet",
    "pose_mobile",
    "new_specialist",
    "context",
    "joint",
    "stable",
    "spectrum",
    "hybrid",
    "native",
}


def training_stage(name):
    return RETRAIN_EXTERNAL and name in EXTERNAL_STAGES


def jpeg_frames(root, names, edge, tag, transpose=True):
    # Порядок строк манифеста определяет индексы в массиве пикселей.
    signature = hashlib.sha256("\n".join(map(str, names)).encode()).hexdigest()[:16]
    file = train_out / f"{tag}-{edge}-{signature}.npy"
    if not file.exists():
        temp = file.with_suffix(".partial.npy")
        pixels = np.lib.format.open_memmap(
            temp, "w+", dtype=np.uint8, shape=(len(names), edge, edge, 3)
        )
        for i, name in enumerate(names):
            with Image.open(root / name) as photo:
                rgb = (ImageOps.exif_transpose(photo) if transpose else photo).convert(
                    "RGB"
                )
                pixels[i] = boxed_rgb(rgb, edge)[0]
        pixels.flush()
        del pixels
        temp.replace(file)
    return np.load(file, mmap_mode="r")


SOURCES_BY_STAGE = {
    "context": ("oxford", "stanford"),
    "hybrid": ("oxford", "stanford"),
    "native": ("petfinder",),
    "stable": ("petfinder",),
    "spectrum": ("petfinder",),
    "joint": ("petfinder", "oxford", "stanford", "coco", "open-images"),
    "new_specialist": ("coco", "open-images"),
    "pose_resnet": ("coco", "open-images"),
    "pose_mobile": ("coco", "open-images"),
    "coco_full": ("coco",),
    "coco_focus": ("coco",),
    "joint_location": ("coco",),
    "detector": ("coco",),
}


SOURCE_INPUTS = {
    "oxford": ("oxford_train", "id", 256),
    "stanford": ("stanford_train", "name", 224),
    "petfinder": ("rescue_train", "name", 224),
    "coco": ("coco_train", "relative_path", 224),
    "open-images": ("open_train", "relative_path", 224),
}


def prepare_training_sources():
    tables = {}
    needed = {k for stage in EXTERNAL_STAGES for k in SOURCES_BY_STAGE[stage]}
    for name in sorted(needed):
        manifest, column, edge = SOURCE_INPUTS[name]
        rows = pd.read_csv(manifests / (manifest + ".csv"))
        root = hw4.download_external(name)
        names = "images/" + rows.id + ".jpg" if name == "oxford" else rows[column]
        transpose = name not in {"stanford", "oxford"}
        frames = jpeg_frames(root, names.tolist(), edge, name, transpose)
        if name == "stanford":
            rows["label"] = 1
        tables[name] = root, rows, frames
    return tables


def train_pixels(array, indices, edge, rng=None):
    return compact_inputs(
        torch.from_numpy(np.array(array[indices])), torch.device("cuda"), edge, rng
    )


def gpu_labels(values):
    return torch.as_tensor(values, device="cuda", dtype=torch.float32)


def to_gpu(model):
    return model.to("cuda", memory_format=torch.channels_last)


def copy_ema(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


def amp_train():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def score_probabilities(target, p):
    p = np.asarray(p)
    hits = int(((p >= 0.5) == target).sum())
    log_likelihood = np.log(np.where(target, p, 1 - p).clip(1e-7, 1)).mean()
    return hits, float(log_likelihood)


def save_training_weights(name, model, epoch, edge, key):
    if key > best_scores.get(name, (-np.inf,)):
        best_scores[name] = key
        file = train_out / (name + ".pt")
        temp = file.with_suffix(".partial")
        torch.save(dict(weights=model.state_dict(), epoch=epoch + 1, edge=edge), temp)
        temp.replace(file)


best_scores = {}


def evaluate_component(name, model, epoch, edge, new_weight, probability=None):
    if probability is None:
        probability = predict_component(model, dev_frames, edge)
    mixed = (1 - new_weight) * dev_table.external.to_numpy() + new_weight * probability
    keep = dev_table.path.map(
        lambda p: int.from_bytes(
            hashlib.sha256(("hw4-subclass-v1/" + p).encode()).digest()[:4], "big"
        )
        % 4
        != 0
    ).to_numpy()
    key = score_probabilities(dev_table.label.to_numpy()[keep], mixed[keep])
    save_training_weights(name, model, epoch, edge, key)
    print(
        f"{name}, эпоха {epoch + 1}, верно {key[0]}/{keep.sum()}, log loss {-key[1]:.5f}"
    )


def apply_update(loss, model, optimizer, ema=None, decay=None, clip=3.0):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    if ema is not None:
        average(ema, model, decay)


def load_parent(model, file, field):
    model.load_state_dict(
        torch.load(file, map_location="cpu", weights_only=True)[field]
    )
    return model


COCO_PLANS = {}


COCO_PLANS["coco_full"] = dict(
    seed=62041,
    epochs=100,
    lr=0.0013,
    change=[0, 24, 55, 82],
    fraction=[0.24, 0.22, 0.14, 0],
    local_start=78,
    crop=0.0,
)


COCO_PLANS["coco_focus"] = dict(
    seed=62177,
    epochs=108,
    lr=0.00165,
    change=[0, 30, 61, 87],
    fraction=[0.24, 0.22, 0.16, 0],
    local_start=84,
    crop=0.48,
)


POSE_PLANS = {}


POSE_PLANS["pose_resnet"] = dict(
    seed=73141,
    epochs=94,
    lr=0.0014,
    change=[0, 22, 50, 77],
    fraction=[0.27, 0.25, 0.19, 0.1],
    hard=[0.12, 0.16, 0.19, 0.22],
    crop=0.32,
    rotation=0.18,
)


POSE_PLANS["pose_mobile"] = dict(
    seed=73189,
    epochs=88,
    lr=0.00165,
    change=[0, 24, 51, 74],
    fraction=[0.28, 0.24, 0.2, 0.1],
    hard=[0.12, 0.16, 0.18, 0.2],
    crop=0.54,
    rotation=0.28,
)


def balanced_ids(labels, count, rng):
    return np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == k),
                count // 2 + (k * int(count % 2)),
                replace=False,
            )
            for k in (0, 1)
        ]
    )


def cosine(epoch, epochs, floor):
    return floor + (1 - floor) * (1 + math.cos(math.pi * epoch / (epochs - 1))) / 2


def external_box_view(rgb, row, margin):
    w, h = rgb.size
    box = np.array([row.left, row.top, row.right, row.bottom])
    a, b, c, d = box * [w, h, w, h]
    dx, dy = margin * (c - a), margin * (d - b)
    region = (
        max(0, int(a - dx)),
        max(0, int(b - dy)),
        min(w, math.ceil(c + dx)),
        min(h, math.ceil(d + dy)),
    )
    img = ImageOps.contain(rgb, (256, 256), Image.Resampling.BICUBIC)
    sw, sh = img.size
    ox, oy = (256 - sw) // 2, (256 - sh) // 2
    box = (box * [sw, sh, sw, sh] + [ox, oy, ox, oy]) / 256
    return boxed_rgb(rgb.crop(region), 256)[0], box


def box_views(group, margin):
    root, rows, _ = sources[group]
    views, boxes = [], []
    for row in rows.itertuples():
        with Image.open(root / row.relative_path) as photo:
            rgb = ImageOps.exif_transpose(photo).convert("RGB")
            image, box = external_box_view(rgb, row, margin)
        views.append(image)
        boxes.append(box)
    return np.stack(views), torch.tensor(np.stack(boxes), dtype=torch.float32)


class JoinedFrames:
    # Считываем только запрошенный батч из файлов, лежащих на диске.
    def __init__(self, *arrays):
        self.arrays = arrays
        self.ends = np.cumsum([len(a) for a in arrays])

    def __getitem__(self, ids):
        ids = np.asarray(ids)
        batch = np.empty((len(ids), *self.arrays[0].shape[1:]), dtype=np.uint8)
        start = 0
        for array, end in zip(self.arrays, self.ends):
            keep = (ids >= start) & (ids < end)
            batch[keep] = array[ids[keep] - start]
            start = end
        return batch


def coco_order(plan, epoch, stage, rng):
    n = len(course_y)
    count = (
        min(len(coco_y), round(n * plan["fraction"][stage]))
        if epoch < plan["local_start"]
        else 0
    )
    order = rng.permutation(n)[: n - count]
    if count:
        extra = balanced_ids(coco_y, count, rng) + n
        order = rng.permutation(np.r_[order, extra])
    return order


def mixed_batch(x, y, rng, alpha):
    t = float(rng.beta(alpha, alpha))
    return x.lerp(x.roll(1, 0), t), y.lerp(y.roll(1, 0), t)


@torch.inference_mode()
def prior_probability(model, frames, edge):
    model.eval()
    parts = []
    for start in range(0, len(frames), 64):
        x = train_pixels(frames, slice(start, start + 64), edge)
        parts.append(
            ((model(x).sigmoid() + model(x.flip(-1)).sigmoid()) / 2).cpu().numpy()
        )
    return np.concatenate(parts)


def coco_epoch(model, opt, plan, epoch, stage, size, rng):
    order = coco_order(plan, epoch, stage, rng)
    model.train()
    for ids in np.array_split(order, math.ceil(len(course_y) / 96)):
        raw = base_images[ids]
        if plan["crop"] and np.any(order >= len(course_y)):
            use = (ids >= len(course_y)) & (rng.random(len(ids)) < plan["crop"])
            raw[use] = coco_crop[ids[use] - len(course_y)]
        x = train_pixels(raw, slice(None), size, rng)
        y = gpu_labels(base_y[ids])
        if rng.random() < 0.035:
            x = x.rot90(int(rng.choice([1, 2, 3], p=[0.4, 0.2, 0.4])), (-2, -1))
        if epoch < plan["epochs"] - 12 and rng.random() < 0.24:
            x, y = mixed_batch(x, y, rng, 0.22)
        with amp_train():
            loss = F.binary_cross_entropy_with_logits(
                model(x).float(), 0.985 * y + 0.0075
            )
        apply_update(loss, model, opt, clip=4)


def fit_coco(name):
    plan = COCO_PLANS[name]
    torch.manual_seed(plan["seed"])
    model = to_gpu(GeMResNet() if name == "coco_full" else FocusMobile())
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=plan["lr"], weight_decay=0.025, fused=True
    )
    for e in range(plan["epochs"]):
        seed = plan["seed"] + 193 * e
        torch.manual_seed(seed)
        stage = sum(e >= i for i in plan["change"]) - 1
        size = [112, 160, 224, 256][stage]
        opt.param_groups[0]["lr"] = (
            plan["lr"] * min(1, (e + 1) / 5) * cosine(e, plan["epochs"], 0.03)
        )
        coco_epoch(model, opt, plan, e, stage, size, np.random.default_rng(seed))
        smooth_weights(ema, model, 0 if e in plan["change"] else 0.72)
        key = score_probabilities(
            dev_table.label.to_numpy(), prior_probability(ema, val256, size)
        )
        save_training_weights(name, ema, e, size, key)


def adjust_location_targets(boxes, device, edge, x, y, side, flip):
    origin = torch.tensor([x, y, x, y], device=device)
    adjusted = (boxes.to(device) * edge - origin) / side
    clipped = adjusted.clamp(0, 1)

    def dims(a):
        return (a[:, 2:] - a[:, :2]).clamp_min(0.0)

    keep = (dims(clipped).prod(dim=1) >= 0.5 * dims(adjusted).prod(dim=1)) & (
        dims(clipped).min(dim=1).values >= 0.015
    )
    reflected = clipped[:, [2, 1, 0, 3]].clone()
    reflected[:, [0, 2]] = 1 - reflected[:, [0, 2]]
    clipped = torch.where(flip[:, None], reflected, clipped)
    return clipped, keep


def train_location_input(raw, boxes, device, size, rng):
    edge = raw.shape[1]
    side = int(rng.integers(round(0.82 * edge), edge + 1))
    y, x = rng.integers(0, edge - side + 1, 2)
    image = compact_inputs(raw[:, y : y + side, x : x + side], device, size)
    flip = torch.rand(len(image), device=device) < 0.5
    image = torch.where(flip[:, None, None, None], image.flip(-1), image)
    colors = torch.rand((len(image), 3, 1, 1), device=device)
    bias = (torch.rand((len(image), 1, 1, 1), device=device) - 0.5) * 0.20
    image = (
        (image * (0.85 + 0.30 * colors) + bias)
        .clamp(-1, 1)
        .contiguous(memory_format=torch.channels_last)
    )
    clipped, keep = adjust_location_targets(boxes, device, edge, x, y, side, flip)
    return image, clipped, keep


def heatmap_negatives(heat, h, w, x, y, bw, bh, index, label):
    grid_y = torch.arange(h, device=heat.device)[None, :, None]
    grid_x = torch.arange(w, device=heat.device)[None, None, :]
    sx = (bw * w / 5).clamp(0.6, 2.3)
    sy = (bh * h / 5).clamp(0.6, 2.3)
    halo = torch.exp(
        -0.5 * ((grid_x - x[:, None, None]) / sx[:, None, None]).square()
        - 0.5 * ((grid_y - y[:, None, None]) / sy[:, None, None]).square()
    )
    suppress = torch.ones_like(heat)
    suppress[index, label] = (1 - halo).pow(4)
    negative = F.softplus(heat) * heat.sigmoid().square() * suppress
    return negative


def train_location_loss(heat, boxes, target, label):
    heat, boxes = heat.float(), boxes.float()
    n, _, h, w = heat.shape
    cx, cy = ((target[:, :2] + target[:, 2:]) / 2).unbind(dim=1)
    bw, bh = (target[:, 2:] - target[:, :2]).clamp_min(0.015).unbind(dim=1)
    x = (cx * w).long().clamp(0, w - 1)
    y = (cy * h).long().clamp(0, h - 1)
    index = torch.arange(n, device=heat.device)
    logpos = heat[index, label, y, x]
    positive = F.softplus(-logpos) * (1 - logpos.sigmoid()).square()
    negative = heatmap_negatives(heat, h, w, x, y, bw, bh, index, label)
    predicted = boxes[index, :, y, x]
    expected = torch.stack([cx * w - x, cy * h - y, bw.log(), bh.log()], dim=1)
    reg = torch.cat([predicted[:, :2].sigmoid(), predicted[:, 2:]], dim=1)
    return (
        positive.mean()
        + negative.sum() / n
        + 1.6 * F.smooth_l1_loss(reg, expected, beta=0.3)
    )


def decode_train_boxes(heat, regression):
    heat = heat.float()
    n, _, h, w = heat.shape
    index = torch.arange(n, device=heat.device)
    per_class = heat.flatten(2).amax(2)
    idx = heat.amax(1).flatten(1).argmax(1)
    y, x = idx // w, idx % w
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


def train_box_iou(pred, target):
    lo = torch.maximum(pred[:, :2], target[:, :2])
    hi = torch.minimum(pred[:, 2:], target[:, 2:])
    overlap = (hi - lo).clamp_min(0.0).prod(dim=1)
    union = (
        (pred[:, 2:] - pred[:, :2]).prod(dim=1)
        + (target[:, 2:] - target[:, :2]).prod(dim=1)
        - overlap
    )
    return overlap / union.clamp_min(1e-6)


LOC_PLANS = {}


LOC_PLANS["joint_location"] = dict(
    seed=72041,
    epochs=108,
    lr=0.0013,
    change=[0, 26, 58, 84],
    sizes=[112, 160, 224, 256],
    fraction=[0.22, 0.2, 0.16, 0.1],
)


LOC_PLANS["detector"] = dict(
    seed=72077, epochs=72, lr=0.0017, change=[0, 18, 44], sizes=[128, 192, 224]
)


def location_partition():
    rank = coco_rows.image_id.map(
        lambda i: hashlib.sha256(f"hw4-coco-box-holdout/{i}".encode()).hexdigest()
    )
    hold = np.sort(
        coco_rows.assign(rank=rank)
        .sort_values("rank")
        .groupby("label")
        .head(256)
        .index.to_numpy()
    )
    return np.setdiff1d(np.arange(len(coco_rows)), hold), hold


@torch.inference_mode()
def location_score(model, hold, size):
    ious, prob = [], []
    for ids in np.array_split(hold, math.ceil(len(hold) / 64)):
        _, heat, reg = model.both(train_pixels(coco256, ids, size))
        boxes, _, p = decode_train_boxes(heat, reg)
        ious.extend(train_box_iou(boxes, coco_box[ids].cuda()).tolist())
        prob.extend(p.cpu().tolist())
    return np.mean(ious), score_probabilities(coco_y[hold], prob)[1]


def location_step(
    model, opt, images, targets, labels, ids, offset, epoch, plan, size, rng
):
    x, boxes, valid = train_location_input(
        torch.from_numpy(images[ids]), targets[ids], "cuda", size, rng
    )
    y = gpu_labels(labels[ids])
    valid &= torch.as_tensor(ids >= offset, device="cuda")
    with amp_train():
        if offset and epoch < plan["epochs"] - 14 and rng.random() < 0.16:
            x, y = mixed_batch(x, y, rng, 0.22)
            output = model(x)
            auxiliary = output.new_zeros(())
        else:
            output, heat, reg = model.both(x)
            auxiliary = (
                train_location_loss(
                    heat[valid], reg[valid], boxes[valid], y[valid].long()
                )
                if valid.any()
                else output.new_zeros(())
            )
        binary = F.binary_cross_entropy_with_logits(output.float(), 0.985 * y + 0.0075)
        loss = binary + 0.22 * auxiliary if offset else 0.25 * binary + auxiliary
    apply_update(loss, model, opt, clip=4)


def location_epoch(model, opt, name, e, plan, stage, size, rng, train_ids):
    if name == "joint_location":
        offset = len(course_y)
        count = min(len(coco_rows), round(offset * plan["fraction"][stage])) // 2 * 2
        added = balanced_ids(coco_y, count, rng) + offset
        order = rng.permutation(np.r_[rng.permutation(offset)[: offset - count], added])
        images, labels, boxes = (
            base_images,
            base_y,
            torch.cat([torch.zeros(offset, 4), coco_box]),
        )
    else:
        offset = 0
        order = rng.permutation(train_ids)
        images, labels, boxes = coco256, coco_y, coco_box
    model.train()
    for ids in np.array_split(order, math.ceil(len(order) / 96)):
        location_step(
            model, opt, images, boxes, labels, ids, offset, e, plan, size, rng
        )


def choose_locator_epoch(name, ema, hold, e, size):
    key = (
        location_score(ema, hold, size)
        if name == "detector"
        else score_probabilities(
            dev_table.label.to_numpy(), prior_probability(ema, val256, size)
        )
    )
    save_training_weights(name, ema, e, size, key)


def fit_locator(name):
    plan = LOC_PLANS[name]
    torch.manual_seed(plan["seed"])
    model = to_gpu(AnimalLocationNet(small=name == "detector"))
    opt = torch.optim.AdamW(
        model.parameters(), lr=plan["lr"], weight_decay=0.025, fused=True
    )
    ema = copy_ema(model)
    train_ids, hold = location_partition()
    for e in range(plan["epochs"]):
        seed = plan["seed"] + 271 * e
        torch.manual_seed(seed)
        stage = sum(e >= i for i in plan["change"]) - 1
        size = plan["sizes"][stage]
        opt.param_groups[0]["lr"] = (
            plan["lr"] * min(1, (e + 1) / 4) * cosine(e, plan["epochs"], 0.025)
        )
        location_epoch(
            model,
            opt,
            name,
            e,
            plan,
            stage,
            size,
            np.random.default_rng(seed),
            train_ids,
        )
        smooth_weights(ema, model, 0 if e in plan["change"] else 0.72)
        choose_locator_epoch(name, ema, hold, e, size)


def pose_augment(x, rng, quarter_turn):
    if rng.random() < quarter_turn:
        x = x.rot90(int(rng.choice([1, 2, 3], p=[0.42, 0.16, 0.42])), (-2, -1))
    if rng.random() < 0.32:
        angle = float(rng.uniform(-0.37, 0.37))
        c, s = math.cos(angle), math.sin(angle)
        matrix = x.new_tensor([[c, -s, 0.0], [s, c, 0.0]]).expand(len(x), -1, -1)
        grid = F.affine_grid(matrix, x.shape, align_corners=False)
        x = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
    grey = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    grayscale = torch.rand((len(x), 1, 1, 1), device=x.device) < 0.19
    x = torch.where(grayscale, grey, x)
    if rng.random() < 0.28:
        brightness = 0.50 + 0.37 * torch.rand((len(x), 1, 1, 1), device=x.device)
        x = (x + 1.0) * brightness - 1.0
    return x.contiguous(memory_format=torch.channels_last)


def pose_order(plan, stage, rng):
    n = len(course_y)
    quota = round(plan["fraction"][stage] * n)

    def balanced(labels, ids, count, replace=False):
        groups = [ids[labels == k] for k in (0, 1)]
        half = count // 2 if replace else min(count // 2, len(groups[0]))
        return np.concatenate([rng.choice(g, half, replace=replace) for g in groups])

    oi = balanced(
        oi_rows.label.to_numpy(),
        np.arange(len(oi_rows)) + len(base_y),
        round(0.62 * quota),
    )
    coco = balanced(coco_y, np.arange(len(coco_y)) + n, quota - len(oi))
    hard = balanced(course_y[hard_ids], hard_ids, round(plan["hard"][stage] * n), True)
    ordinary = rng.permutation(n)[: n - len(oi) - len(coco) - len(hard)]
    return rng.permutation(np.r_[ordinary, hard, oi, coco])


def pose_epoch(model, opt, e, plan, stage, size, rng):
    model.train()
    order = pose_order(plan, stage, rng)
    for ids in np.array_split(order, math.ceil(len(course_y) / 96)):
        batch = pose_images[ids]
        zoom = (ids >= len(base_y)) & (rng.random(len(ids)) < plan["crop"])
        batch[zoom] = oi_crop[ids[zoom] - len(base_y)]
        x = pose_augment(
            train_pixels(batch, slice(None), size, rng), rng, plan["rotation"]
        )
        y = gpu_labels(pose_y[ids])
        if e < plan["epochs"] - 12 and rng.random() < 0.18:
            x, y = mixed_batch(x, y, rng, 0.18)
        with amp_train():
            loss = F.binary_cross_entropy_with_logits(
                model(x).float(), 0.985 * y + 0.0075
            )
        apply_update(loss, model, opt, clip=4)


def choose_pose_epoch(name, ema, stage, flag, y, e, size):
    if stage > 0:
        p = prior_probability(ema, val256[flag], size)
        key = (
            *score_probabilities(y, (gate.previous.to_numpy()[flag] + p) / 2),
            score_probabilities(y, p)[1],
        )
        save_training_weights(name, ema, e, size, key)


def fit_pose(name):
    plan = POSE_PLANS[name]
    torch.manual_seed(plan["seed"])
    model = to_gpu(GeMResNet() if name == "pose_resnet" else FocusMobile())
    opt = torch.optim.AdamW(
        model.parameters(), lr=plan["lr"], weight_decay=0.028, fused=True
    )
    ema = copy_ema(model)
    flag = gate.flagged.to_numpy()
    y = gate.label.to_numpy()[flag]
    for e in range(plan["epochs"]):
        seed = plan["seed"] + 307 * e
        torch.manual_seed(seed)
        stage = sum(e >= i for i in plan["change"]) - 1
        size = [112, 160, 224, 256][stage]
        opt.param_groups[0]["lr"] = (
            plan["lr"] * min(1, (e + 1) / 5) * cosine(e, plan["epochs"], 0.03)
        )
        pose_epoch(model, opt, e, plan, stage, size, np.random.default_rng(seed))
        smooth_weights(ema, model, 0 if e in plan["change"] else 0.72)
        choose_pose_epoch(name, ema, stage, flag, y, e, size)


def obscure_train_photo(x, rng):
    batch, size = x.shape[0], x.shape[-1]
    for i in np.where(rng.random(batch) < 0.45)[0]:
        h, w = rng.uniform(0.14, 0.46, 2)
        h, w = round(h * size), round(w * size)
        r, c = rng.integers(size - h + 1), rng.integers(size - w + 1)
        # Заполняем прямоугольник средним цветом или рисуем узкие полосы.
        if rng.random() < 0.45:
            gap = int(rng.integers(19, 47))
            width = int(rng.integers(2, 6))
            offset = int(rng.integers(gap))
            for a in range(offset, size, gap):
                x[i, :, :, a : a + width] = float(rng.uniform(-0.75, 0.75))
        else:
            x[i, :, r : r + h, c : c + w] = x[i].mean(dim=(1, 2), keepdim=True)
    return x


def specialist_example(job):
    row, root, seed = job
    rng = np.random.default_rng(seed)
    raw = (root / row.relative_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row.sha256:
        raise ValueError(row.relative_path)
    with Image.open(BytesIO(raw)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    if rng.random() < 0.80:
        w, h = image.size
        box = np.array([row.left * w, row.top * h, row.right * w, row.bottom * h])
        dim = box[2:] - box[:2]
        center = (box[2:] + box[:2]) / 2 + rng.uniform(-0.15, 0.15, 2) * dim
        half = rng.uniform(0.86, 2.0) * 0.5 * dim
        a = np.clip(np.floor(center - half), 0, [w - 1, h - 1]).astype(int)
        b = np.maximum(a + 1, np.minimum([w, h], np.ceil(center + half).astype(int)))
        image = image.crop((*a, *b))
    if rng.random() < 0.20:
        image = ImageOps.autocontrast(image, cutoff=1)
    return boxed_rgb(image, 288)[0], row.label


def specialist_view(rgb, box, view):
    if view == "full":
        return rgb
    scale = {"wide": 1.55, "tight": 0.72}[view]
    w, h = rgb.size
    left, top, right, bottom = box
    cx, cy = (left + right) / 2, (top + bottom) / 2
    dx, dy = (right - left) * scale / 2, (bottom - top) * scale / 2
    x, y = max(0, round(cx - dx)), max(0, round(cy - dy))
    return rgb.crop(
        (x, y, max(x + 1, min(w, round(cx + dx))), max(y + 1, min(h, round(cy + dy))))
    )


def specialist_validation(model, gate):
    rows = gate.loc[gate.flagged]
    parts = []
    model.eval()
    for view in ("full", "wide", "tight"):
        group = []
        for row in rows.itertuples():
            with Image.open(course_root / row.path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                box = (row.x0, row.y0, row.x1, row.y1)
                group.append(boxed_rgb(specialist_view(rgb, box, view), 288)[0])
        probs = []
        with torch.inference_mode(), amp_train():
            for start in range(0, len(group), 12):
                x = train_pixels(np.stack(group[start : start + 12]), slice(None), 288)
                p = (
                    model(x).float().sigmoid() + model(x.flip(-1)).float().sigmoid()
                ) / 2
                probs.extend(p.cpu().tolist())
        parts.append(probs)
    return np.mean(parts, axis=0)


def specialist_batch(tables, hard, rng):
    ix = np.r_[
        rng.choice(hard, 4, replace=False), rng.choice(len(course_y), 4, replace=False)
    ]
    x0 = train_pixels(train_frames, ix, 288, rng)
    jobs = [
        (table[i], root, int(rng.integers(2**31)))
        for table, root in tables
        for i in rng.choice(len(table), 4, replace=False)
    ]
    photos, labels = zip(*(specialist_example(job) for job in jobs))
    x1 = train_pixels(np.stack(photos), slice(None), 288, rng)
    return obscure_train_photo(torch.cat([x0, x1]), rng), gpu_labels(
        np.r_[course_y[ix], labels]
    )


def crop_specialist_epoch(model, ema, opt, tables, hard, rng):
    model.train()
    opt.zero_grad(set_to_none=True)
    for step in range(256):
        x, y = specialist_batch(tables, hard, rng)
        with amp_train():
            loss = F.binary_cross_entropy_with_logits(
                model(x).float(), 0.985 * y + 0.0075
            )
        (loss / 4).backward()
        if step % 4 == 3:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            smooth_weights(ema, model, 0.965)


def crop_specialist_sources():
    tables = [
        (list(sources[k][1].itertuples(index=False)), sources[k][0])
        for k in ("coco", "open-images")
    ]
    hard = pd.read_csv(manifests / "hard_train.csv")
    hard = (
        pd.Series(np.arange(len(course_y)), index=train_table.relative_path)
        .loc[hard.relative_path]
        .to_numpy()
    )
    gate = pd.read_csv(results / "external_training_gate.csv")
    assert gate.path.str.startswith("val/").all() and gate.flagged.sum() == 124
    return tables, hard, gate


def fit_crop_specialist():
    torch.manual_seed(82177)
    model = to_gpu(
        load_parent(
            GeMResNet(), prev / "baseline/current/coco_full_best.pt", "state_dict"
        )
    )
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.0001, weight_decay=0.02, fused=True
    )
    tables, hard, gate = crop_specialist_sources()
    for e in range(10):
        torch.manual_seed(82177 + e)
        rng = np.random.default_rng(82177 + e)
        opt.param_groups[0]["lr"] = 0.0001 * cosine(e, 10, 0.12)
        crop_specialist_epoch(model, ema, opt, tables, hard, rng)
        p = gate.current.to_numpy(copy=True)
        p[gate.flagged] = (p[gate.flagged] + specialist_validation(ema, gate)) / 2
        save_training_weights(
            "new_specialist", ema, e, 288, score_probabilities(gate.label.to_numpy(), p)
        )


def same_species_loss(z, y, temperature):
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    logits = ((z @ z.T).float() / temperature).masked_fill(diagonal, -1e4)
    positive = ((y >= 0.5)[:, None] == (y >= 0.5)[None, :]) & ~diagonal
    return -(
        (logits.log_softmax(1) * positive).sum(1) / positive.sum(1).clamp_min(1)
    ).mean()


def oxford_stanford(x, y, edge, rng, n_ox, n_st):
    xs, ys = [x], [y]
    for name, count in (("oxford", n_ox), ("stanford", n_st)):
        _, table, pixels = sources[name]
        j = rng.choice(len(table), count, replace=False)
        xs.append(train_pixels(pixels, j, edge))
        ys.append(table.label.to_numpy()[j])
    return torch.cat(xs), gpu_labels(np.concatenate(ys))


def split_optimizer(model, lr, decay):
    groups = [
        {"params": model.base.parameters()},
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if not name.startswith("base.")
            ]
        },
    ]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=decay, fused=True)


def context_epoch(model, ema, opt, e, edge, rng, hard):
    model.train()
    quota = (28 if edge < 320 else 20) - 6 - 4
    order = rng.permutation(len(course_y))[:8600]
    for start in range(0, len(order) - quota + 1, quota):
        ix = np.r_[order[start : start + quota], rng.choice(hard, 4, replace=False)]
        x = train_pixels(train_frames, ix, edge)
        y = 0.96 * course_y[ix] + 0.04 * teacher_p[ix]
        x, y = oxford_stanford(x, y, edge, rng, 4, 2)
        x = augment(x, True)
        with amp_train():
            logits, z = model(x, embedding=True)
            binary = F.binary_cross_entropy_with_logits(
                logits.float(), 0.992 * y + 0.004
            )
            loss = binary + 0.09 * same_species_loss(z, y, 0.14)
        apply_update(loss, model, opt, ema, 0.982)


def fit_full_context():
    torch.manual_seed(86018)
    model = Context()
    load_parent(model.base, prev / "baseline/current/coco_full_best.pt", "state_dict")
    model.base.head = nn.Identity()
    model = to_gpu(model)
    ema = copy_ema(model)
    opt = split_optimizer(model, 0.0001, 0.04)
    hard = np.flatnonzero((hardness >= np.quantile(hardness, 0.8)) & (hardness <= 0.86))
    for e in range(22):
        torch.manual_seed(86018 + e)
        edge = 192 if e < 6 else 256 if e < 16 else 320
        rate = 0.00019 * cosine(e, 22, 0.08)
        opt.param_groups[0]["lr"] = rate * min(1, (e + 1) / 4)
        opt.param_groups[1]["lr"] = 3.8 * rate
        context_epoch(model, ema, opt, e, edge, np.random.default_rng(86018 + e), hard)
        evaluate_component("context", ema, e, edge, 0.35)


def taxonomy_sources():
    rec = sources["petfinder"][1].merge(
        pd.read_csv(manifests / "rescue_breeds.csv"),
        how="left",
        on="name",
        sort=False,
        validate="one_to_one",
    )
    assert rec.breed.notna().all()
    data = [(train_frames, course_y, None, None)]
    for name, cols in (
        ("petfinder", (157, 214)),
        ("oxford", (0, 37)),
        ("stanford", (37, 157)),
        ("coco", None),
        ("open-images", None),
    ):
        _, row, pix = sources[name]
        breed = (
            rec.breed.to_numpy()
            if name == "petfinder"
            else row.breed.to_numpy() if cols else None
        )
        data.append((pix, row.label.to_numpy(), breed, cols))
    return data


def taxonomy_batch(data, phase, edge, rng):
    counts = (
        [(4, 4), (5, 2), (2, 2), (0, 3), (2, 2), (2, 2)]
        if phase < 0.60
        else [(7, 7), (4, 2), (1, 1), (0, 2), (1, 1), (1, 1)]
    )
    frames, labels, aux = [], [], []
    for (pix, label, breed, cols), pair in zip(data, counts):
        ix = np.concatenate(
            [
                rng.choice(np.flatnonzero(label == k), n, False)
                for k, n in enumerate(pair)
                if n
            ]
        )
        if breed is not None:
            start = sum(map(len, labels))
            aux.append(
                (
                    slice(start, start + len(ix)),
                    slice(*cols),
                    torch.as_tensor(breed[ix], device="cuda"),
                )
            )
        frames.append(train_pixels(pix, ix, edge))
        labels.append(label[ix])
    return augment(torch.cat(frames), True), gpu_labels(np.concatenate(labels)), aux


def taxonomy_epoch(model, ema, opt, data, edge, phase, rng):
    model.train()
    for _ in range(300):
        x, y, aux = taxonomy_batch(data, phase, edge, rng)
        with amp_train():
            logits, fine = model(x, True)
            loss = F.binary_cross_entropy_with_logits(logits.float(), 0.99 * y + 0.005)
            for rows, cols, labels in aux:
                known = labels >= 0
                if known.any():
                    loss = loss + (0.16 - 0.06 * phase) * F.cross_entropy(
                        fine[rows, cols][known].float(),
                        labels[known],
                        label_smoothing=0.06,
                    )
        apply_update(loss, model, opt, ema, 0.985)


def initialize_taxonomy():
    model = Taxonomy()
    state = torch.load(
        archive / "external_context.pt", map_location="cpu", weights_only=True
    )["weights"]
    missing, unknown = model.load_state_dict(state, strict=False)
    assert set(missing) == {"fine.weight", "fine.bias"} and not unknown
    model = to_gpu(model)
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not n.startswith("fine.")
                ]
            },
            {"params": model.fine.parameters()},
        ],
        lr=0.00016,
        weight_decay=0.035,
        fused=True,
    )
    return model, ema, opt


def fit_taxonomy():
    torch.manual_seed(79009)
    model, ema, opt = initialize_taxonomy()
    data = taxonomy_sources()
    for e in range(38):
        torch.manual_seed(79009 + e)
        phase = e / 37
        edge = 224 if phase < 0.45 else 256 if phase < 0.82 else 288
        rate = 0.0002 * cosine(e, 38, 0.08)
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = rate, 4 * rate
        taxonomy_epoch(
            model, ema, opt, data, edge, phase, np.random.default_rng(79009 + e)
        )
        evaluate_component("joint", ema, e, edge, 0.28)


def stable_objective(model, x, y):
    with amp_train():
        logit, z = model(x, embedding=True)
        return F.binary_cross_entropy_with_logits(
            logit.float(), 0.99 * y + 0.005
        ) + 0.07 * same_species_loss(z, y, 0.18)


def stable_update(model, ema, opt, x, y):
    opt.zero_grad(set_to_none=True)
    stable_objective(model, x, y).backward()
    params = [p for p in model.parameters() if p.grad is not None]
    with torch.no_grad():
        norm = torch.linalg.vector_norm(
            torch.stack([p.grad.norm() for p in params])
        ).clamp_min(1e-7)
        offsets = [p.grad * (0.035 / norm) for p in params]
        torch._foreach_add_(params, offsets)
    opt.zero_grad(set_to_none=True)
    layers = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)]
    for layer in layers:
        layer.track_running_stats = False
    try:
        stable_objective(model, x, y).backward()
    finally:
        with torch.no_grad():
            torch._foreach_sub_(params, offsets)
        for layer in layers:
            layer.track_running_stats = True
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    opt.step()
    average(ema, model, 0.98)


def stable_epoch(model, ema, opt, edge, rng, hard, pixels, rescue_y):
    model.train()
    for _ in range(185):
        ix = np.r_[rng.choice(len(course_y), 10, False), rng.choice(hard, 4, False)]
        j = balanced_ids(rescue_y, 10, rng)
        x = torch.cat(
            [train_pixels(train_frames, ix, edge), train_pixels(pixels, j, edge)]
        )
        y = gpu_labels(np.r_[0.95 * course_y[ix] + 0.05 * teacher_p[ix], rescue_y[j]])
        stable_update(model, ema, opt, augment(x, True), y)


def fit_stable_context():
    torch.manual_seed(90042)
    model = Context()
    model.base.head = nn.Identity()
    model = to_gpu(load_parent(model, archive / "external_context.pt", "weights"))
    ema = copy_ema(model)
    opt = split_optimizer(model, 0.00013, 0.035)
    hard = np.flatnonzero((hardness > np.quantile(hardness, 0.76)) & (hardness < 0.83))
    _, rec, pixels = sources["petfinder"]
    rescue_y = rec.label.to_numpy()
    for e in range(20):
        torch.manual_seed(90042 + e)
        rng = np.random.default_rng(90042 + e)
        edge = 224 if e < 5 else 256 if e < 15 else 288
        rate = 0.00009 * cosine(e, 20, 0.14)
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = rate, 1.2 * rate
        stable_epoch(model, ema, opt, edge, rng, hard, pixels, rescue_y)
        evaluate_component("stable", ema, e, edge, 0.26)


def style_pair(x):
    # Фазу исходного изображения сохраняем.
    fft = torch.fft.rfft2(x.float(), norm="ortho")
    amp = fft.abs().clamp_min(0.00001)
    donor = amp.roll(1, 0)
    alpha = torch.rand(len(x), 1, 1, 1, device=x.device) * 0.62 + 0.10
    ratio = ((donor.log() - amp.log()) * alpha).clamp(-1.6, 1.6).exp()
    return (
        torch.fft.irfft2(fft * ratio, s=x.shape[-2:], norm="ortho")
        .clamp(-1, 1)
        .contiguous(memory_format=torch.channels_last)
    )


def spectrum_objective(model, ema, x, y, original, attributes, n, j):
    with amp_train():
        logit, aux = model(x, True)
        loss = F.binary_cross_entropy_with_logits(logit.float(), 0.99 * y + 0.005)
        with torch.no_grad():
            target = ema(original).float().sigmoid()
        loss = loss + 0.16 * F.mse_loss(logit[:6].float().sigmoid(), target)
        for columns, values in attributes:
            labels = torch.as_tensor(values[j], device="cuda")
            loss = loss + 0.12 * F.cross_entropy(
                aux[n:, columns].float(), labels, label_smoothing=0.06
            )
    return loss


def spectrum_epoch(model, ema, opt, edge, phase, rng, hard, attributes):
    _, rec, pixels = sources["petfinder"]
    rescue_y = rec.label.to_numpy()
    extra = 12 if phase < 0.65 else 8
    n = 24 - extra
    model.train()
    for _ in range(205):
        ix = np.r_[rng.choice(len(course_y), n - 3, False), rng.choice(hard, 3, False)]
        j = balanced_ids(rescue_y, extra, rng)
        rng.shuffle(j)
        x = torch.cat(
            [train_pixels(train_frames, ix, edge), train_pixels(pixels, j, edge)]
        )
        x = augment(x, False)
        original = x[:6]
        if rng.random() < 0.68:
            x = style_pair(x)
        y = gpu_labels(np.r_[0.94 * course_y[ix] + 0.06 * teacher_p[ix], rescue_y[j]])
        loss = spectrum_objective(model, ema, x, y, original, attributes, n, j)
        apply_update(loss, model, opt, ema, 0.984)


def rescue_attributes():
    rec = sources["petfinder"][1]
    ages = rec.age.str.extract(r"Quantile (\d+)-", expand=False).astype(int)
    attributes = [
        (slice(0, 8), np.searchsorted(sorted(ages.unique()), ages)),
        (slice(8, 12), rec["size"].to_numpy() - 1),
        (slice(12, 15), rec.fur.to_numpy() - 1),
    ]
    return attributes


def fit_spectrum():
    torch.manual_seed(78114)
    model = Spectrum()
    load_parent(model.base, prev / "baseline/current/coco_full_best.pt", "state_dict")
    model = to_gpu(model)
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        [
            {"params": m.parameters()}
            for m in (model.base, model.adapter, model.attribute)
        ],
        lr=0.0002,
        weight_decay=0.025,
        fused=True,
    )
    attributes = rescue_attributes()
    hard = np.flatnonzero((hardness >= np.quantile(hardness, 0.7)) & (hardness < 0.8))
    for e in range(32):
        torch.manual_seed(78114 + e)
        edge = 224 if e / 31 < 0.6 else 288
        for group, scale in zip(opt.param_groups, (1, 8, 5)):
            group["lr"] = 0.00030 * cosine(e, 32, 0.10) * scale
        spectrum_epoch(
            model,
            ema,
            opt,
            edge,
            e / 31,
            np.random.default_rng(78114 + e),
            hard,
            attributes,
        )
        evaluate_component("spectrum", ema, e, edge, 0.35)


RETRAIN_PARENTS = False


PARENT_STAGES = {"course_hybrid", "course_convnext"}


def parent_photos(names, edge, geometry):
    stamp = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    path = parent_work / f"{geometry}_{edge}_{stamp}.npy"
    if not path.exists():
        tmp = path.with_suffix(".partial.npy")
        photos = np.lib.format.open_memmap(
            tmp, "w+", dtype=np.uint8, shape=(len(names), edge, edge, 3)
        )
        for i, name in enumerate(names):
            with Image.open(parent_root / name) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                if geometry == "center":
                    photos[i] = np.asarray(
                        ImageOps.fit(rgb, (edge, edge), Image.Resampling.BICUBIC)
                    )
                else:
                    photos[i] = boxed_rgb(rgb, edge)[0]
        photos.flush()
        del photos
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def checkpoint_parent(path, key, best, state):
    if key > best:
        temp = path.with_suffix(".partial.pt")
        torch.save(state, temp)
        temp.replace(path)
        return key
    return best


def parent_validation(name):
    rows = pd.read_csv(results / f"{name}.csv")
    assert (
        len(rows) == 2487
        and rows.path.is_unique
        and rows.path.str.startswith("val/").all()
    )
    return rows


def augment_convnext_parent(x, y, epoch, rng):
    side = x.shape[-1]
    if rng.random() < 0.20:
        x = ((x + 1) / 2).clamp(0, 1).pow(float(rng.uniform(0.65, 1.6))) * 2 - 1
    if rng.random() < 0.09:
        x = x.rot90(int(rng.choice([1, 3])), (-2, -1)).contiguous(
            memory_format=torch.channels_last
        )
    if epoch < 54 and rng.random() < 0.25:
        width = int(side * rng.uniform(0.22, 0.63))
        top, left = rng.integers(side - width + 1, size=2)
        patch = x[:, :, top : top + width, left : left + width]
        patch.copy_(patch.roll(1, 0))
        y = y.lerp(y.roll(1, 0), (width / side) ** 2)
    return x, y


def convnext_parent_epoch(model, opt, photos, targets, edge, epoch, rng):
    model.train()
    batch = 80 if edge == 112 else 56 if edge == 160 else 40
    order = rng.permutation(len(photos))
    for ids in np.array_split(order, math.ceil(len(order) / batch)):
        x = train_pixels(photos, ids, edge, rng)
        x, y = augment_convnext_parent(x, gpu_labels(targets[ids]), epoch, rng)
        with amp_train():
            loss = F.binary_cross_entropy_with_logits(
                model(x).float(), 0.99 * y + 0.005
            )
        apply_update(loss, model, opt)


@torch.inference_mode()
def independent_validation(model, photos, edge):
    model.eval()
    p = []
    for start in range(0, len(photos), 40):
        x = train_pixels(photos, slice(start, start + 40), edge)
        with amp_train():
            prob = (
                model(x).float().sigmoid() + model(x.flip(-1)).float().sigmoid()
            ) / 2
        p.extend(prob.cpu().tolist())
    return np.array(p)


def parent_key(y, p, floor):
    return int(((p >= 0.5) == y).sum()), float(
        np.log(np.where(y, p, 1 - p).clip(floor, 1)).mean()
    )


def select_convnext_parent(ema, valid, images, edge, e, best):
    p = independent_validation(ema, images, edge)
    mix = 0.75 * valid.current.to_numpy() + 0.25 * p
    key = parent_key(valid.label.to_numpy(), mix, 1e-8)
    state = dict(
        state_dict=ema.state_dict(), size=edge, epoch=e + 1, init="random", seed=82231
    )
    return checkpoint_parent(parent_work / "independent_best.pt", key, best, state)


def parent_convnext_frames():
    valid = parent_validation("course_round4_validation")
    photos = parent_photos(parent_train.relative_path.tolist(), 224, "center")
    validation = parent_photos(valid.path.tolist(), 224, "center")
    targets = 0.88 * parent_y.astype(np.float32) + 0.12 * parent_soft.astype(np.float32)
    return valid, photos, validation, targets


def fit_parent_convnext():
    torch.manual_seed(82231)
    valid, photos, validation, targets = parent_convnext_frames()
    model = to_gpu(CourseConvNeXt())
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.0012, weight_decay=0.035, fused=True
    )
    best = (-1, -10.0)
    for e in range(60):
        torch.manual_seed(82231 + e)
        edge = 112 if e < 15 else 160 if e < 33 else 192 if e < 47 else 224
        opt.param_groups[0]["lr"] = 0.0012 * min(1, (e + 1) / 4) * cosine(e, 60, 0.018)
        convnext_parent_epoch(
            model, opt, photos, targets, edge, e, np.random.default_rng(82231 + e)
        )
        smooth_weights(ema, model, 0 if e in (0, 15, 33, 47) else 0.65)
        if e >= 32:
            best = select_convnext_parent(ema, valid, validation, edge, e, best)
    return best


def parent_descriptor(net, x):
    z = net.stem(x)
    parts = []
    for i, block in enumerate(net.stages):
        z = block(z)
        if i in (6, 12, 15):
            parts.append(z.float().clamp_min(1e-5).pow(3).mean((2, 3)).pow(1 / 3))
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def masked_feature_targets(photos):
    weights = hw4.open_models("winners") / "previous/masked_best.pt"
    teacher = to_gpu(load_parent(GeMResNet(), weights, "state_dict")).eval()
    vectors = []
    for start in range(0, len(photos), 16):
        x = train_pixels(photos, slice(start, start + 16), 256)
        with amp_train():
            z = (
                parent_descriptor(teacher, x) + parent_descriptor(teacher, x.flip(-1))
            ) * 0.5
        vectors.append(z.cpu().numpy().astype(np.float16))
    bank = np.concatenate(vectors).astype(np.float32)
    return gpu_labels((bank - bank.mean(0)) / bank.std(0).clip(0.01))


def hybrid_parent_epoch(model, ema, opt, photos, reference, edge, e, rng):
    model.train()
    batch = {128: 80, 176: 64, 224: 40, 288: 28}[edge]
    order = rng.permutation(len(photos))
    for start in range(0, len(order) - batch + 1, batch):
        ids = order[start : start + batch]
        x = augment(train_pixels(photos, ids, edge), True)
        y = gpu_labels(0.88 * parent_y[ids] + 0.12 * parent_soft[ids])
        target = reference[ids]
        if rng.random() < 0.35:
            t = float(rng.beta(0.3, 0.3))
            x, y, target = (a.lerp(a.roll(1, 0), t) for a in (x, y, target))
        with amp_train():
            logit, z = model(x, teacher=True)
            binary = F.binary_cross_entropy_with_logits(
                logit.float(), 0.994 * y + 0.003
            )
            distance = 1 - F.cosine_similarity(z, target).mean()
            loss = binary + (0.70 - 0.40 * e / 63) * distance
        apply_update(loss, model, opt, ema, 0.988)


def hybrid_parent_data():
    val = parent_validation("external_training_leaders").sort_values("path")
    photos = parent_photos(parent_train.relative_path.tolist(), 320, "box")
    valid = parent_photos(val.path.tolist(), 320, "box")
    keep = val.path.map(
        lambda p: int.from_bytes(
            hashlib.sha256(("hw4-subclass-v1/" + p).encode()).digest()[:4], "big"
        )
        % 4
        != 0
    ).to_numpy()
    return photos, valid, val, keep


def select_hybrid_parent(ema, valid, val, keep, edge, e, best):
    p = predict_component(ema, valid, edge)
    mix = 0.65 * val.course.to_numpy() + 0.35 * p
    key = parent_key(val.label.to_numpy()[keep], mix[keep], 1e-7)
    state = dict(weights=ema.state_dict(), edge=edge, epoch=e + 1)
    return checkpoint_parent(parent_work / "course_hybrid.pt", key, best, state)


def fit_parent_hybrid():
    photos, valid, val, keep = hybrid_parent_data()
    reference = masked_feature_targets(photos)
    torch.manual_seed(57229)
    model = to_gpu(Hybrid())
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.00085, weight_decay=0.045, fused=True
    )
    best = (-1, -1000.0)
    for e in range(64):
        torch.manual_seed(57229 + e)
        phase = e / 63
        edge = (
            128
            if phase < 0.25
            else 176 if phase < 0.55 else 224 if phase < 0.82 else 288
        )
        opt.param_groups[0]["lr"] = 0.0009 * min(1, (e + 1) / 3) * cosine(e, 64, 0.025)
        hybrid_parent_epoch(
            model,
            ema,
            opt,
            photos,
            reference,
            edge,
            e,
            np.random.default_rng(57229 + e),
        )
        if e >= 12:
            best = select_hybrid_parent(ema, valid, val, keep, edge, e, best)
    return best


@torch.inference_mode()
def hybrid_targets():
    teacher = to_gpu(
        load_parent(
            GeMResNet(), prev / "baseline/current/coco_full_best.pt", "state_dict"
        )
    ).eval()
    rows = []
    for start in range(0, len(course_y), 16):
        x = train_pixels(train_frames, slice(start, start + 16), 256)
        descriptors = []
        for view in (x, x.flip(-1)):
            with amp_train():
                z = teacher.stem(view)
                parts = []
                for i, block in enumerate(teacher.stages):
                    z = block(z)
                    if i in (6, 12, 15):
                        parts.append(
                            z.float().clamp_min(1e-5).pow(3).mean((2, 3)).pow(1 / 3)
                        )
                descriptors.append(torch.cat(parts, dim=1))
        rows.append(
            ((descriptors[0] + descriptors[1]) * 0.5).cpu().numpy().astype(np.float16)
        )
    vectors = np.concatenate(rows).astype(np.float32)
    assert vectors.shape == (len(course_y), 560)
    return gpu_labels((vectors - vectors.mean(0)) / vectors.std(0).clip(0.01))


def hybrid_epoch(model, ema, opt, reference, epoch, edge, rng):
    phase = epoch / 31
    batch = {176: 64, 224: 40, 288: 28}[edge]
    count = batch - 12
    order = rng.permutation(len(course_y))
    model.train()
    for start in range(0, len(order) - count + 1, count):
        ix = order[start : start + count]
        x = train_pixels(train_frames, ix, edge)
        target = 0.88 * course_y[ix] + 0.12 * teacher_p[ix]
        x, y = oxford_stanford(x, target, edge, rng, 8, 4)
        x = augment(x, True)
        with amp_train():
            logit, embedding = model(x, teacher=True)
            species = F.binary_cross_entropy_with_logits(
                logit.float(), 0.994 * y + 0.003
            )
            distance = 1 - F.cosine_similarity(embedding[:count], reference[ix]).mean()
            loss = species + (0.70 - 0.40 * phase) * distance
        apply_update(loss, model, opt, ema, 0.988)


def fit_hybrid():
    reference = hybrid_targets()
    torch.manual_seed(57230)
    model = to_gpu(load_parent(Hybrid(), parents / "course_hybrid.pt", "weights"))
    ema = copy_ema(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.00085, weight_decay=0.045, fused=True
    )
    for e in range(32):
        torch.manual_seed(57230 + e)
        phase = e / 31
        edge = 176 if phase < 0.55 else 224 if phase < 0.82 else 288
        opt.param_groups[0]["lr"] = 0.00024 * min(1, (e + 1) / 3) * cosine(e, 32, 0.025)
        hybrid_epoch(
            model, ema, opt, reference, e, edge, np.random.default_rng(57230 + e)
        )
        if e >= 4:
            evaluate_component("hybrid", ema, e, edge, 0.35)


@torch.inference_mode()
def write_native_tokens(model, files, name):
    path = train_out / (name + "_tokens.npy")
    tmp = path.with_suffix(".partial.npy")
    tokens = np.lib.format.open_memmap(
        tmp, "w+", dtype=np.float16, shape=(len(files), 144, 128)
    )
    start = 0
    for batch in native_batches(model, files):
        n = len(batch)
        tokens[start : start + n] = batch.cpu().numpy()
        start += n
    assert start == len(files)
    tokens.flush()
    del tokens
    tmp.replace(path)
    return np.load(path, mmap_mode="r")


def native_data(model):
    paths = [course_root / path for path in train_table.relative_path]
    # Сохранённые коэффициенты исключают вариацию собственных векторов между GPU.
    state = torch.load(
        parents / "external_native_projection.pt", map_location="cpu", weights_only=True
    )
    assert state["parents"] == [
        hw4.checksum(prev / "baseline/current/coco_full_best.pt"),
        hw4.checksum(parents / "independent_best.pt"),
    ]
    model.offset.copy_(state["offset"])
    model.components.copy_(state["components"])
    root, rec, _ = sources["petfinder"]
    paths.extend(root / path for path in rec.name)
    tokens = write_native_tokens(model, paths, "native_train")
    validation = write_native_tokens(
        model, [course_root / p for p in dev_table.path], "native_val"
    )
    labels = np.r_[course_y, rec.label.to_numpy()]
    weight = np.r_[np.ones(len(course_y)), np.full(len(rec), 0.65)]
    return tokens, validation, labels, weight


@torch.inference_mode()
def native_probabilities(model, tokens):
    p = []
    model.eval()
    for start in range(0, len(tokens), 64):
        x = torch.from_numpy(tokens[start : start + 64].copy()).cuda()
        with amp_train():
            p.extend(model.classify(x).float().sigmoid().cpu().tolist())
    return np.array(p)


def native_epoch(model, opt, tokens, labels, weights, rng):
    model.train()
    order = rng.permutation(len(tokens))
    for ix in np.array_split(order, math.ceil(len(order) / 96)):
        x = gpu_labels(tokens[ix])
        y = gpu_labels(labels[ix])
        sw = gpu_labels(weights[ix])
        if rng.random() < 0.55:
            t = float(rng.beta(0.3, 0.3))
            x = x.lerp(x.roll(1, 0), t)
            y = y.lerp(y.roll(1, 0), t)
            sw = sw.lerp(sw.roll(1, 0), t)
        with amp_train():
            logits = model.classify(x).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, 0.986 * y + 0.007, weight=sw
            )
        apply_update(loss, model, opt)


def fit_native():
    torch.manual_seed(7237)
    model = Native()
    load_parent(model.res, prev / "baseline/current/coco_full_best.pt", "state_dict")
    load_parent(model.conv, parents / "independent_best.pt", "state_dict")
    model.res.requires_grad_(False)
    model.conv.requires_grad_(False)
    model = to_gpu(model).eval()
    tokens, validation, labels, weights = native_data(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=0.0006, weight_decay=0.05, fused=True)
    for e in range(24):
        opt.param_groups[0]["lr"] = 0.0007 * cosine(e, 24, 0.12)
        native_epoch(
            model, opt, tokens, labels, weights, np.random.default_rng(8771 + e)
        )
        evaluate_component(
            "native", model, e, 448, 0.35, native_probabilities(model, validation)
        )


def run_training():
    global archive, base_images, base_y, coco256, coco_box, coco_crop, coco_rows
    global coco_y, course_root, course_y, dev_frames, dev_table, gate, hard_ids
    global hardness, oi_crop, oi_rows, parent_root, parent_soft, parent_train
    global parent_work, parent_y, parents, pose_images, pose_y, prev, sources
    global teacher_p, train_frames, train_out, train_table, val256

    if not (RETRAIN_EXTERNAL or RETRAIN_PARENTS):
        return
    best_scores.clear()

    if RETRAIN_EXTERNAL:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Для полного обучения нужен CUDA GPU с поддержкой bfloat16.")
        torch.set_num_threads(2)
        train_out = hw4.CACHE / "external-retraining"
        train_out.mkdir(parents=True, exist_ok=True)
        archive = hw4.open_models("winners")
        prev = archive / "previous"
        if EXTERNAL_STAGES.intersection({"hybrid", "native"}):
            parents = hw4.open_models("parents")
        train_table = pd.read_csv(manifests / "train_audit_scores.csv")
        dev_table = pd.read_csv(results / "external_training_leaders.csv").sort_values(
            "path"
        )
        assert train_table.relative_path.str.startswith("train/").all()
        assert dev_table.path.str.startswith("val/").all()
        course_root = hw4.extract_photos("course", split="train")
        hw4.extract_photos("course", split="val")
        course_y = train_table.label.to_numpy()
        teacher_p = (
            train_table[["deep_resnet", "mbconv", "context_resnet"]].mean(axis=1).to_numpy()
        )
        hardness = np.where(course_y, 1 - teacher_p, teacher_p)

    if RETRAIN_EXTERNAL:
        sources = prepare_training_sources()
        train_frames = jpeg_frames(
            course_root, train_table.relative_path.tolist(), 320, "course-train"
        )
        dev_frames = jpeg_frames(course_root, dev_table.path.tolist(), 320, "course-val")

    if RETRAIN_EXTERNAL and EXTERNAL_STAGES.intersection(
        {*COCO_PLANS, *POSE_PLANS, "detector", "joint_location"}
    ):
        course256 = jpeg_frames(
            course_root, train_table.relative_path.tolist(), 256, "course-train"
        )
        val256 = jpeg_frames(course_root, dev_table.path.tolist(), 256, "course-val")
        coco_root, coco_rows, _ = sources["coco"]
        coco256 = jpeg_frames(coco_root, coco_rows.relative_path.tolist(), 256, "coco")
        coco_crop, coco_box = box_views("coco", 0.30)
        coco_y = coco_rows.label.to_numpy()
        base_images = JoinedFrames(course256, coco256)
        base_y = np.r_[course_y, coco_y]

    for stage_name in COCO_PLANS:
        if training_stage(stage_name):
            fit_coco(stage_name)

    for stage_name in LOC_PLANS:
        if training_stage(stage_name):
            fit_locator(stage_name)

    if RETRAIN_EXTERNAL and EXTERNAL_STAGES.intersection(POSE_PLANS):
        oi_root, oi_rows, _ = sources["open-images"]
        oi256 = jpeg_frames(oi_root, oi_rows.relative_path.tolist(), 256, "open-images")
        oi_crop, _ = box_views("open-images", 0.35)
        pose_images = JoinedFrames(course256, coco256, oi256)
        pose_y = np.r_[base_y, oi_rows.label.to_numpy()]
        hard_table = pd.read_csv(manifests / "hard_train.csv")
        hard_ids = (
            pd.Series(np.arange(len(course_y)), index=train_table.relative_path)
            .loc[hard_table.relative_path]
            .to_numpy()
        )
        gate = dev_table[["path", "label"]].merge(
            pd.read_csv(results / "pose_training_gate.csv"),
            on="path",
            validate="one_to_one",
        )
        assert gate.flagged.sum() == 124

    for stage_name in POSE_PLANS:
        if training_stage(stage_name):
            fit_pose(stage_name)

    if training_stage("new_specialist"):
        fit_crop_specialist()

    if training_stage("context"):
        fit_full_context()

    if training_stage("joint"):
        fit_taxonomy()

    if training_stage("stable"):
        fit_stable_context()

    if training_stage("spectrum"):
        fit_spectrum()

    if RETRAIN_PARENTS:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Для обучения родительских моделей нужен CUDA GPU с bfloat16."
            )
        torch.set_num_threads(2)
        parent_work = hw4.CACHE / "parent-retraining"
        parent_work.mkdir(parents=True, exist_ok=True)
        parent_root = hw4.extract_photos("course", split="train")
        hw4.extract_photos("course", split="val")
        parent_train = pd.read_csv(manifests / "train_audit_scores.csv")
        assert (
            len(parent_train) == 19991
            and parent_train.relative_path.str.startswith("train/").all()
        )
        parent_y = parent_train.label.to_numpy()
        parent_soft = (
            parent_train[["deep_resnet", "mbconv", "context_resnet"]]
            .mean(axis=1)
            .to_numpy()
        )

    if RETRAIN_PARENTS and "course_convnext" in PARENT_STAGES:
        previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        try:
            print("ConvNeXt, максимум на validation", fit_parent_convnext())
        finally:
            torch.set_float32_matmul_precision(previous_precision)

    if RETRAIN_PARENTS and "course_hybrid" in PARENT_STAGES:
        print("Hybrid, максимум на validation-dev", fit_parent_hybrid())

    if training_stage("hybrid"):
        fit_hybrid()

    if training_stage("native"):
        fit_native()


def main(argv=None):
    global RETRAIN_EXTERNAL, EXTERNAL_STAGES, RETRAIN_PARENTS, PARENT_STAGES
    global results, manifests
    parser = argparse.ArgumentParser(
        description="Обучение конкурсных моделей с внешними данными; нужен CUDA GPU с bfloat16."
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        choices=sorted(SOURCES_BY_STAGE),
        default=["hybrid", "native"],
        help="Внешние модели; по умолчанию hybrid native. Пустой список пропускает их обучение.",
    )
    parser.add_argument(
        "--parents",
        nargs="*",
        choices=["course_hybrid", "course_convnext"],
        default=[],
        help="Родительские модели; по умолчанию их повторное обучение отключено.",
    )
    args = parser.parse_args(argv)
    EXTERNAL_STAGES, PARENT_STAGES = set(args.stages), set(args.parents)
    RETRAIN_EXTERNAL, RETRAIN_PARENTS = bool(EXTERNAL_STAGES), bool(PARENT_STAGES)
    if not (RETRAIN_EXTERNAL or RETRAIN_PARENTS):
        print("Этапы обучения не выбраны.")
        return 0
    results, manifests = hw4.prepare()
    run_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
