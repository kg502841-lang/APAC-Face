"""Train one experiment from config.yaml.

    python -m src.train --experiment baseline
    python -m src.train --experiment proposed --resume
    python -m src.train --experiment baseline --smoke      # 2 epochs on a small subset

Writes checkpoints/<experiment>.pth (best validation NME, inference weights only),
checkpoints/<experiment>_last.pth (full training state for --resume) and
results/history_<experiment>.json (per-epoch losses, validation NME, learning rate).
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augmentations import AugmentationParams, LandmarkAugmenter, apply_affine, crop_matrix
from .dataset import (ATTRIBUTES, DatasetNotFoundError, WFLWCropDataset, balanced_sample_weights,
                      build_cache, canonical_cache_box)
from .metrics import per_sample_nme
from .model import build_model, count_parameters, save_inference_checkpoint
from .utils import (LOG_FORMAT, auto_batch_size, describe_device, get_device, get_logger,
                    load_config, resolve_path, save_json, seed_worker, set_seed)

log = get_logger("train")


class WingLoss(nn.Module):
    """Wing loss (Feng et al., CVPR 2018) with width/curvature in normalised coordinates."""

    def __init__(self, width: float = 10 / 224, curvature: float = 2 / 224):
        super().__init__()
        self.w, self.e = width, curvature
        self.c = width - width * math.log(1 + width / curvature)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (pred - target).abs()
        return torch.where(x < self.w, self.w * torch.log1p(x / self.e), x - self.c).mean()


def make_loss(train_cfg: dict[str, Any]) -> nn.Module:
    name = train_cfg.get("loss", "smooth_l1")
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=float(train_cfg.get("smooth_l1_beta", 0.01)))
    if name == "l1":
        return nn.L1Loss()
    if name == "wing":
        return WingLoss()
    raise ValueError(f"Unknown loss '{name}' (choose smooth_l1, l1 or wing)")


def mean_shape(cfg: dict[str, Any], dataset: WFLWCropDataset) -> np.ndarray:
    size = int(cfg["data"]["input_size"])
    m = crop_matrix(canonical_cache_box(cfg), size)
    lm = apply_affine(dataset.landmarks.reshape(-1, 2), m).reshape(-1, 98, 2) / size
    return lm.mean(axis=0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, grad_clip) -> float:
    model.train()
    total, count = 0.0, 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(images)
        loss = criterion(pred.float(), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss - lower the learning rate")
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(images)
        count += len(images)
    return total / max(count, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, amp) -> tuple[float, np.ndarray]:
    model.eval()
    total, count, preds, gts = 0.0, 0, [], []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(images)
        pred = pred.float()
        total += criterion(pred, targets).item() * len(images)
        count += len(images)
        preds.append(pred.cpu().numpy())
        gts.append(targets.cpu().numpy())
    nme = per_sample_nme(np.concatenate(preds), np.concatenate(gts))
    return total / max(count, 1), nme


def train(cfg: dict[str, Any], experiment: str, resume: bool = False, smoke: bool = False,
          epochs: int | None = None, batch_size: int | None = None, device_pref: str = "auto"
          ) -> dict[str, Any]:
    if experiment not in cfg["experiments"]:
        raise KeyError(f"Unknown experiment '{experiment}'. Options: {list(cfg['experiments'])}")
    run_name = f"{experiment}_smoke" if smoke else experiment
    log_path = resolve_path(cfg["paths"]["results"]) / "logs" / f"train_{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a" if resume else "w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    try:
        return _train(cfg, experiment, run_name, resume, smoke, epochs, batch_size, device_pref)
    finally:
        log.removeHandler(handler)
        handler.close()


def _train(cfg: dict[str, Any], experiment: str, run_name: str, resume: bool, smoke: bool,
           epochs: int | None, batch_size: int | None, device_pref: str) -> dict[str, Any]:
    exp_cfg = cfg["experiments"][experiment]
    tcfg = cfg["train"]
    seed = int(cfg["seed"])
    set_seed(seed, bool(cfg.get("deterministic", False)))
    device = get_device(device_pref)
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    epochs = int(epochs or (2 if smoke else tcfg["epochs"]))

    build_cache(cfg)
    size = int(cfg["data"]["input_size"])
    params = AugmentationParams.from_config(cfg["augmentation"])
    train_ds = WFLWCropDataset(cfg, "train", LandmarkAugmenter(params, size, exp_cfg["augment"]),
                               limit=512 if smoke else None)
    val_ds = WFLWCropDataset(cfg, "val", LandmarkAugmenter(params, size, enabled=False),
                             limit=128 if smoke else None)

    bs = auto_batch_size(int(batch_size or tcfg["batch_size"]), device)
    gen = torch.Generator().manual_seed(seed)
    sampler = None
    if exp_cfg["balanced_sampling"]:
        cols = [ATTRIBUTES.index(a) for a in cfg["sampling"]["attributes"]]
        weights = balanced_sample_weights(train_ds.attributes, cols, float(cfg["sampling"]["power"]))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True, generator=gen)
        log.info("Balanced sampling: weight range %.2f-%.2f", weights.min(), weights.max())
    # On Windows every worker process loads the CUDA libraries (~1.3 GB of committed memory), so
    # only the training loader gets workers; validation crops are few and load in-process.
    workers = int(cfg["data"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                              drop_last=True, generator=gen, num_workers=workers,
                              pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")

    model = build_model(cfg["model"]).to(device)
    shape0 = mean_shape(cfg, train_ds)
    model.init_mean_shape(shape0)
    criterion = make_loss(tcfg)
    optimizer = torch.optim.AdamW(model.parameter_groups(float(tcfg["lr"]), float(tcfg["head_lr_mult"])),
                                  weight_decay=float(tcfg["weight_decay"]))
    if tcfg.get("scheduler", "plateau") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(tcfg["plateau_factor"]),
            patience=int(tcfg["plateau_patience"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    best_path, last_path = ckpt_dir / f"{run_name}.pth", ckpt_dir / f"{run_name}_last.pth"
    history: dict[str, list] = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_nme", "lr", "seconds")}
    start_epoch, best_nme, best_epoch, bad_epochs = 0, math.inf, 0, 0

    if resume:
        if last_path.is_file():
            state = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            scaler.load_state_dict(state["scaler_state"])
            history = state["history"]
            start_epoch, best_nme = state["epoch"], state["best_nme"]
            best_epoch, bad_epochs = state["best_epoch"], state["bad_epochs"]
            torch.set_rng_state(state["torch_rng"].cpu())
            log.info("Resumed %s from epoch %d (best val NME %.4f)", run_name, start_epoch, best_nme)
        else:
            log.warning("--resume given but %s does not exist; starting from scratch", last_path)

    patience = int(tcfg["early_stopping_patience"])
    log.info("Experiment %s | %s | device %s | batch %d | epochs %d | %d train / %d val faces | %.2fM params",
             run_name, exp_cfg.get("description", ""), describe_device(device), bs, epochs,
             len(train_ds), len(val_ds), count_parameters(model) / 1e6)
    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp,
                                     float(tcfg.get("grad_clip", 0)))
        val_loss, val_nme_all = validate(model, val_loader, criterion, device, amp)
        val_nme = float(np.nanmean(val_nme_all))
        lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_nme)
        else:
            scheduler.step()
        for k, v in zip(history, (epoch + 1, train_loss, val_loss, val_nme, lr, time.time() - t0)):
            history[k].append(v)

        improved = val_nme < best_nme - 1e-6
        if improved:
            best_nme, best_epoch, bad_epochs = val_nme, epoch + 1, 0
            save_inference_checkpoint(best_path, model, cfg, experiment, extra={
                "epoch": best_epoch, "val_nme": best_nme, "mean_shape": shape0.tolist(),
                "experiment_cfg": exp_cfg, "batch_size": bs,
            })
        else:
            bad_epochs += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "history": history, "epoch": epoch + 1, "best_nme": best_nme, "best_epoch": best_epoch,
            "bad_epochs": bad_epochs, "torch_rng": torch.get_rng_state(),
        }, last_path)
        log.info("epoch %3d/%d  train %.5f  val %.5f  val NME %.4f%s  lr %.1e  %.0fs",
                 epoch + 1, epochs, train_loss, val_loss, val_nme, " *" if improved else "", lr,
                 time.time() - t0)
        if bad_epochs >= patience:
            log.info("Early stopping: no validation improvement for %d epochs", patience)
            break

    summary = {
        "experiment": experiment, "run_name": run_name, "config": exp_cfg,
        "best_epoch": best_epoch, "best_val_nme": best_nme, "epochs_run": len(history["epoch"]),
        "batch_size": bs, "device": describe_device(device), "amp": amp,
        "train_faces": len(train_ds), "val_faces": len(val_ds),
        "parameters": count_parameters(model), "train_seconds": time.time() - t_start,
        "history": history,
    }
    save_json(summary, resolve_path(cfg["paths"]["results"]) / f"history_{run_name}.json")
    log.info("Done. Best val NME %.4f at epoch %d -> %s", best_nme, best_epoch, best_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an APAC-Face landmark model")
    parser.add_argument("--experiment", required=True, help="key under `experiments` in config.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true", help="continue from <experiment>_last.pth")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on a small subset")
    args = parser.parse_args()
    try:
        train(load_config(args.config), args.experiment, resume=args.resume, smoke=args.smoke,
              epochs=args.epochs, batch_size=args.batch_size, device_pref=args.device)
    except DatasetNotFoundError as e:
        print(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
