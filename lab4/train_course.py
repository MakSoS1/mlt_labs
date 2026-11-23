"""Повторное обучение конкурсных моделей на данных курса.

Пример: python train_course.py --models course_context
"""

import argparse
import copy
import hashlib
import math
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

import hw4
from contest import (
    Context,
    DeepResidualNet,
    GeMResNet,
    MobileContestNet,
    augment,
    average,
    boxed_rgb,
    compact_inputs,
)


# seed, число эпох, начальный LR, переходы (с нуля), размер входа
COURSE_RECIPES = {
    "deep_resnet": (20101, 84, 0.00125, (0, 24, 52), (112, 160, 224)),
    "mbconv": (20077, 64, 0.0018, (0, 20, 43), (112, 160, 224)),
    "highres": (20131, 18, 0.00024, (0, 6), (224, 256)),
    "context_resnet": (42139, 86, 0.00135, (0, 28, 56), (112, 160, 224)),
    "label_control": (62041, 92, 0.0013, (0, 24, 53, 78), (112, 160, 224, 256)),
}


COURSE_RUNS = ("course_context",)


def pack_course_images(root, paths, edge, boxed, folder):
    paths = list(paths)
    signature = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:16]
    mode = "box" if boxed else "stretch"
    dest = folder / f"{signature}_{mode}_{edge}.npy"
    if not dest.is_file():
        partial = dest.with_suffix(".partial.npy")
        frames = np.lib.format.open_memmap(
            partial, mode="w+", dtype="uint8", shape=(len(paths), edge, edge, 3)
        )
        for i, path in enumerate(paths):
            with Image.open(root / path) as original:
                rgb = ImageOps.exif_transpose(original).convert("RGB")
                if boxed:
                    frames[i] = boxed_rgb(rgb, edge)[0]
                else:
                    frames[i] = np.asarray(
                        rgb.resize((edge, edge), Image.Resampling.BILINEAR)
                    )
        frames.flush()
        del frames
        partial.replace(dest)
    return np.load(dest, mmap_mode="r")


def course_amp(device):
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()


@torch.inference_mode()
def course_scores(net, frames, edge, device, batch, amp=False, average_logits=False):
    net.eval()
    parts = []
    for start in range(0, len(frames), batch):
        raw = torch.from_numpy(frames[start : start + batch].copy())
        x = compact_inputs(raw, device, edge)
        with course_amp(device) if amp else nullcontext():
            a, b = net(x).float(), net(x.flip(-1)).float()
        p = (
            ((a + b) / 2).sigmoid()
            if average_logits
            else (a.sigmoid() + b.sigmoid()) / 2
        )
        parts.append(p.cpu().numpy())
    return np.concatenate(parts)


def course_validation_key(y, prob, eps=1e-8):
    y, prob = np.asarray(y), np.asarray(prob, dtype=np.float64)
    hits = int(np.count_nonzero((prob >= 0.5) == y))
    correct_prob = np.where(y, prob, 1 - prob)
    nll = -np.log(np.clip(correct_prob, eps, 1)).mean()
    return hits, -float(nll)


@torch.no_grad()
def course_smooth(ema, net, decay):
    for old, new in zip(ema.parameters(), net.parameters()):
        old.lerp_(new, 1 - decay)
    for old, new in zip(ema.buffers(), net.buffers()):
        if old.is_floating_point():
            old.lerp_(new, 1 - decay)
        else:
            old.copy_(new)


def augment_round2(name, x, y, epoch, epochs, rng):
    size = x.shape[-1]
    if name == "deep_resnet" and epoch < epochs - 10 and rng.random() < 0.22:
        side = int(size * np.sqrt(rng.uniform(0.05, 0.5)))
        top, left = rng.integers(0, size - side + 1, size=2)
        patch = x[:, :, top : top + side, left : left + side]
        patch.copy_(patch.roll(1, 0))
        fraction = (side / size) ** 2
        y = (1 - fraction) * y + fraction * y.roll(1, 0)
    elif epoch < epochs - 6 and rng.random() < 0.30:
        mix = float(rng.beta(0.20, 0.20))
        x = mix * x + (1 - mix) * x.roll(1, 0)
        y = mix * y + (1 - mix) * y.roll(1, 0)
    return x, y


def augment_round3(name, x, y, epoch, epochs, rng):
    rotation_prob = 0.035 if name == "label_control" else 0.10
    if rng.random() < rotation_prob:
        turns = int(rng.choice([1, 2, 3], p=[0.4, 0.2, 0.4]))
        x = x.rot90(turns, (-2, -1))
    if epoch < epochs - 12 and rng.random() < 0.24:
        mix = float(rng.beta(0.22, 0.22))
        x = x.lerp(x.roll(1, 0), mix)
        y = y.lerp(y.roll(1, 0), mix)
    return x, y


def course_optimizer_step(net, optimizer, loss, clip):
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), clip)
    optimizer.step()


def init_course_cnn(name, package):
    if name == "mbconv":
        return MobileContestNet()
    if name == "label_control":
        return GeMResNet()
    net = DeepResidualNet()
    if name == "highres":
        path = package / "previous/baseline/prior/round2/deep_resnet_best.pt"
        net.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        )
    return net


def early_course_epoch(name, net, optimizer, frames, labels, epoch, edge, device):
    seed, epochs, _, _, _ = COURSE_RECIPES[name]
    boxed = name in ("context_resnet", "label_control")
    epoch_seed = seed + epoch * (193 if boxed else 1009)
    torch.manual_seed(epoch_seed)
    rng = np.random.default_rng(epoch_seed)
    batches = np.array_split(rng.permutation(len(labels)), math.ceil(len(labels) / 96))
    loss_sum = 0.0
    net.train()
    for ids in batches:
        x = compact_inputs(torch.from_numpy(frames[ids]), device, edge, rng)
        y = torch.as_tensor(labels[ids], device=device, dtype=torch.float32)
        augment_batch = augment_round3 if boxed else augment_round2
        x, y = augment_batch(name, x, y, epoch, epochs, rng)
        target = 0.985 * y + 0.0075 if boxed else 0.98 * y + 0.01
        optimizer.zero_grad(set_to_none=True)
        with course_amp(device):
            loss = F.binary_cross_entropy_with_logits(net(x).float(), target)
        course_optimizer_step(net, optimizer, loss, 4.0 if boxed else 5.0)
        loss_sum += loss.detach().item() * len(y)
    return loss_sum / len(labels)


def save_course_epoch(folder, name, state, val, probabilities):
    temp = folder / f"{name}.partial.pt"
    torch.save(state, temp)
    temp.replace(folder / f"{name}.pt")
    pd.DataFrame({"path": val.path, "prob": probabilities}).to_csv(
        folder / f"{name}_validation.csv", index=False
    )


def early_course_schedule(name, e):
    _, epochs, base_lr, steps, edges = COURSE_RECIPES[name]
    boxed = name in ("context_resnet", "label_control")
    edge = edges[sum(e >= step for step in steps) - 1]
    floor, warmup = (0.03, 5) if boxed else (0.025, 3)
    cosine = (1 + math.cos(math.pi * e / (epochs - 1))) / 2
    lr = base_lr * min(1, (e + 1) / warmup) * (floor + (1 - floor) * cosine)
    ema_decay = 0 if e in steps else (0.72 if boxed else 0.66)
    return edge, lr, ema_decay


def prepare_early_course(name, package, device):
    seed, _, base_lr, _, _ = COURSE_RECIPES[name]
    torch.manual_seed(seed)
    net = init_course_cnn(name, package).to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    boxed = name in ("context_resnet", "label_control")
    decay = {"context_resnet": 0.035, "label_control": 0.025}.get(name, 0.012)
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=base_lr,
        weight_decay=decay,
        fused=boxed and device.type == "cuda",
    )
    return net, ema, opt


def train_course_cnn(name, frames, valid, tr, val, package, folder, device):
    net, ema, opt = prepare_early_course(name, package, device)
    epochs = COURSE_RECIPES[name][1]
    labels = tr.label.to_numpy()
    best, rows = (-1, -float("inf")), []
    for e in range(epochs):
        edge, lr, ema_decay = early_course_schedule(name, e)
        opt.param_groups[0]["lr"] = lr
        loss = early_course_epoch(name, net, opt, frames, labels, e, edge, device)
        course_smooth(ema, net, ema_decay)
        p = course_scores(ema, valid, edge, device, batch=64)
        key = course_validation_key(val.label, p)
        if key > best:
            best = key
            state = dict(
                state_dict=ema.state_dict(), size=edge, epoch=e + 1, architecture=name
            )
            save_course_epoch(folder, name, state, val, p)
        rows.append(
            dict(
                epoch=e + 1,
                edge=edge,
                lr=lr,
                train_loss=loss,
                hits=key[0],
                val_loss=-key[1],
            )
        )
        pd.DataFrame(rows).to_csv(folder / f"{name}_history.csv", index=False)
    return pd.DataFrame(rows)


def obscure_course_batch(x, rng):
    size = x.shape[-1]
    for i in np.flatnonzero(rng.random(len(x)) < 0.45):
        h, w = (rng.uniform(0.14, 0.46, 2) * size).round().astype(int)
        top, left = rng.integers(size - h + 1), rng.integers(size - w + 1)
        if rng.random() < 0.45:
            gap = int(rng.integers(19, 47))
            width = int(rng.integers(2, 6))
            for col in range(int(rng.integers(gap)), size, gap):
                x[i, :, :, col : col + width] = float(rng.uniform(-0.75, 0.75))
        else:
            x[i, :, top : top + h, left : left + w] = x[i].mean((1, 2), keepdim=True)
    return x


def course_parent(net, path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    state = torch.load(path, weights_only=True, map_location="cpu")
    net.load_state_dict(state["state_dict"], strict=True)
    return digest


def masked_course_epoch(net, ema, opt, frames, labels, epoch, device):
    seed = 81307
    rng, masks = np.random.default_rng(seed + epoch), np.random.default_rng(
        seed + epoch + 55000
    )
    torch.manual_seed(seed + epoch)
    order = np.random.default_rng(seed + epoch // 2).permutation(len(frames))
    ids = np.array_split(order, 2)[epoch % 2]
    batches = np.array_split(ids, math.ceil(len(ids) / 16))
    net.train()
    opt.zero_grad(set_to_none=True)
    for i, rows in enumerate(batches):
        x = compact_inputs(torch.from_numpy(frames[rows]), device, 320, rng)
        x = obscure_course_batch(x, masks)
        y = torch.as_tensor(labels[rows], dtype=torch.float32, device=device)
        with course_amp(device):
            loss = F.binary_cross_entropy_with_logits(
                net(x).float(), 0.985 * y + 0.0075
            )
        group_size = min(4, len(batches) - 4 * (i // 4))
        (loss / group_size).backward()
        if i % 4 == 3 or i == len(batches) - 1:
            nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            course_smooth(ema, net, 0.975)


def init_masked_course(package, device):
    torch.manual_seed(81307)
    net = GeMResNet()
    parent = course_parent(
        net, package / "previous/baseline/prior/round4/label_control_best.pt"
    )
    net.to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    opt = torch.optim.AdamW(
        net.parameters(), lr=0.0001, weight_decay=0.02, fused=device.type == "cuda"
    )
    return net, ema, opt, parent


def train_masked_course(frames, valid, tr, val, package, folder, device):
    net, ema, opt, parent = init_masked_course(package, device)
    prior = pd.read_csv(results / "course_round4_validation.csv").set_index("path")
    reference = prior.loc[val.path, "current"].to_numpy()
    best, rows = (-1, -float("inf")), []
    for e in range(12):
        lr = 0.00012 * (0.12 + 0.88 * (1 + math.cos(math.pi * e / 11)) / 2)
        opt.param_groups[0]["lr"] = lr
        masked_course_epoch(net, ema, opt, frames, tr.label.to_numpy(), e, device)
        p = course_scores(ema, valid, 320, device, batch=16, amp=True)
        key = course_validation_key(val.label, 0.65 * reference + 0.35 * p)
        if key > best:
            best = key
            state = dict(
                state_dict=ema.state_dict(),
                parent=parent,
                epoch=e + 1,
                size=320,
                arm="masked",
            )
            save_course_epoch(folder, "masked", state, val, p)
        rows.append(dict(epoch=e + 1, lr=lr, hits=key[0], val_loss=-key[1]))
        pd.DataFrame(rows).to_csv(folder / "masked_history.csv", index=False)
    return pd.DataFrame(rows)


def course_teacher(tr):
    teacher = tr[["deep_resnet", "mbconv", "context_resnet"]].mean(axis=1).to_numpy()
    y = tr.label.to_numpy()
    hardness = np.where(y, 1 - teacher, teacher)
    hard = np.flatnonzero(
        (hardness >= np.quantile(hardness, 0.80)) & (hardness <= 0.86)
    )
    return 0.96 * y + 0.04 * teacher, hard


def course_context_batches(n, hard, batch, rng):
    order = rng.permutation(n)
    count, regular = min(n, 8600), batch - 4
    for start in range(0, count - regular + 1, regular):
        yield np.r_[order[start : start + regular], rng.choice(hard, 4, replace=False)]


def course_context_loss(logits, z, y):
    ce = F.binary_cross_entropy_with_logits(logits.float(), 0.992 * y + 0.004)
    similarity = (z @ z.T).float() / 0.14
    same_image = torch.eye(len(z), device=z.device, dtype=torch.bool)
    log_p = similarity.masked_fill(same_image, -1e4).log_softmax(dim=1)
    positive = ((y > 0.5)[:, None] == (y > 0.5)[None, :]) & ~same_image
    metric = -(log_p * positive).sum(1) / positive.sum(1).clamp_min(1)
    return ce + 0.09 * metric.mean(), ce


def course_context_epoch(net, ema, opt, frames, target, hard, e, device):
    seed = 86017 + e
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    edge = 192 if e < 6 else 256 if e < 16 else 320
    batch = 28 if edge < 320 else 20
    lr = 0.00019 * (0.08 + 0.92 * (1 + math.cos(math.pi * e / 21)) / 2)
    opt.param_groups[0]["lr"] = lr * min(1, (e + 1) / 4)
    opt.param_groups[1]["lr"] = 3.8 * lr
    net.train()
    losses = []
    for ids in course_context_batches(len(frames), hard, batch, rng):
        x = compact_inputs(torch.from_numpy(frames[ids]), device, edge)
        x = augment(x, True)
        y = torch.as_tensor(target[ids], device=device, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        with course_amp(device):
            logits, embeddings = net(x, embedding=True)
            loss, ce = course_context_loss(logits, embeddings, y)
        course_optimizer_step(net, opt, loss, 3.0)
        average(ema, net, 0.982)
        losses.append(ce.detach())
    return edge, float(torch.stack(losses).mean())


def init_course_context(package, device):
    torch.manual_seed(86017)
    net = Context()
    parent = course_parent(net.base, package / "previous/masked_best.pt")
    net.base.head = nn.Identity()
    net.to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    heads = [p for name, p in net.named_parameters() if not name.startswith("base.")]
    opt = torch.optim.AdamW(
        [{"params": net.base.parameters()}, {"params": heads}],
        lr=0.0001,
        weight_decay=0.04,
        fused=device.type == "cuda",
    )
    return net, ema, opt, parent


def course_dev_mask(paths):
    def fold(path):
        digest = hashlib.sha256(("hw4-subclass-v1/" + path).encode()).digest()
        return int.from_bytes(digest[:4], "big") % 4

    return np.array([fold(path) != 0 for path in paths])


def restore_course_context(resume, net, ema, opt, parent):
    if not resume.is_file():
        return [], (-1, -float("inf"))
    ck = torch.load(resume, weights_only=True, map_location="cpu")
    if ck["parent"] != parent:
        raise ValueError(
            "Начальные веса не совпадают с сохранённым состоянием обучения"
        )
    net.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
    opt.load_state_dict(ck["opt"])
    return ck["hist"], tuple(ck["best"])


def save_course_progress(resume, net, ema, opt, parent, rows, best):
    state = dict(
        parent=parent,
        model=net.state_dict(),
        ema=ema.state_dict(),
        opt=opt.state_dict(),
        hist=rows,
        best=best,
    )
    partial = resume.with_suffix(".partial.pt")
    torch.save(state, partial)
    partial.replace(resume)


def train_course_context(frames, valid, tr, val, package, folder, device):
    net, ema, opt, parent = init_course_context(package, device)
    target, hard = course_teacher(tr)
    dev = course_dev_mask(val.path)
    resume = folder / "course_context_resume.pt"
    rows, best = restore_course_context(resume, net, ema, opt, parent)
    for e in range(len(rows), 22):
        edge, loss = course_context_epoch(
            net, ema, opt, frames, target, hard, e, device
        )
        p = course_scores(
            ema, valid, edge, device, batch=28, amp=True, average_logits=True
        )
        combined = 0.65 * val.course.to_numpy() + 0.35 * p
        key = course_validation_key(val.label.to_numpy()[dev], combined[dev], eps=1e-7)
        if key > best:
            best = key
            state = dict(
                weights=ema.state_dict(), edge=edge, epoch=e + 1, parent=parent
            )
            save_course_epoch(folder, "course_context", state, val, p)
        rows.append(
            dict(epoch=e + 1, edge=edge, hits=key[0], dev_loss=-key[1], train_loss=loss)
        )
        save_course_progress(resume, net, ema, opt, parent, rows, best)
        pd.DataFrame(rows).to_csv(folder / "course_context_history.csv", index=False)
        print(
            f"course_context, эпоха {e + 1:2d}: {key[0]}/{dev.sum()}, log loss {-key[1]:.5f}"
        )
    return pd.DataFrame(rows)


def course_training_tables():
    tr = pd.read_csv(manifests / "train_audit_scores.csv")
    val = pd.read_csv(results / "validation.csv").sort_values("path")
    assert len(tr) == 19991 and tr.relative_path.str.startswith("train/").all()
    assert len(val) == 2487 and val.path.str.startswith("val/").all()
    return tr, val


def repeat_course_contest(names):
    if not names:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    package = hw4.open_models("winners")
    folder = hw4.CACHE / "course-contest-training"
    folder.mkdir(parents=True, exist_ok=True)
    tr, val = course_training_tables()
    roots = {split: hw4.extract_photos("course", split) for split in ("train", "val")}
    for name in names:
        if name not in {*COURSE_RECIPES, "masked", "course_context"}:
            raise ValueError(f"Неизвестная конкурсная сеть {name}")
        boxed = name not in ("deep_resnet", "mbconv", "highres")
        edge = max(COURSE_RECIPES[name][4]) if name in COURSE_RECIPES else 320
        train_pixels = pack_course_images(
            roots["train"], tr.relative_path, edge, boxed, folder
        )
        val_pixels = pack_course_images(roots["val"], val.path, edge, boxed, folder)
        args = (train_pixels, val_pixels, tr, val, package, folder, device)
        if name == "course_context":
            history = train_course_context(*args)
        elif name == "masked":
            history = train_masked_course(*args)
        else:
            history = train_course_cnn(name, *args)
        print(history.tail(3).to_string(index=False))


def main(argv=None):
    global results, manifests
    parser = argparse.ArgumentParser(
        description="Обучение конкурсных моделей на данных курса."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=[*COURSE_RECIPES, "masked", "course_context"],
        default=list(COURSE_RUNS),
        help="Модели для обучения; по умолчанию course_context. Пустой список — без запуска.",
    )
    args = parser.parse_args(argv)
    if not args.models:
        print("Модели не выбраны.")
        return 0
    results, manifests = hw4.prepare()
    repeat_course_contest(args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
