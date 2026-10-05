"""Task 2 — compact Autoformer forecaster for the 168-step leaderboard challenge.

Architecture follows Autoformer (Wu et al., 2021, §3.1–3.2): progressive series decomposition inside
every encoder/decoder layer, and an Auto-Correlation mechanism (FFT period-based dependency discovery
plus top-k time-delay aggregation) in place of point-wise self-attention. Adapted from the paper's
description and the Part 3 aggregation written for Task 1 of this assignment; the delay aggregation
is per-example (as in Task 1) rather than batch-shared during training.

Optional external variables enter as covariate embeddings: past values on encoder positions, and
label-window plus the 168 supplied future values on decoder positions. Future target values are
never supplied to the model.

Usage:
    python autoformer_task2.py --seed 0 --exog all  --out runs/exog_s0
    python autoformer_task2.py --seed 0 --exog none --out runs/noexog_s0
    python autoformer_task2.py --seed 0 --exog all  --final --epochs 6 --out runs/final_s0
    python autoformer_task2.py --seed 0 --trend-init exog --features engineered \
        --target-transform log1p --anchor --out runs/exogtrend_s0

Experiments are normally driven from task2.ipynb through task2_experiments.py.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "Data"
PRED_LEN = 168
VAL_LEN = 4368            # last 26 weeks of the given history form the chronological validation region
VAL_STRIDE = 24           # one validation forecast origin per day -> 176 overlapping 168-step blocks
CONTINUOUS = ["feature_A", "feature_B", "feature_C", "feature_D", "feature_E", "feature_F"]
LOG_FIRST = ["feature_D", "feature_E", "feature_F"]   # cumulative, heavy-tailed, mostly zero
BINARY = ["feature_G", "feature_H", "feature_I", "feature_J"]


# ----------------------------------------------------------------------------- data

def trailing_mean(values, window):
    """Mean of the last `window` values up to and including each position (shorter at the start)."""
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    i = np.arange(len(values))
    lo = np.maximum(0, i - window + 1)
    return (cumulative[i + 1] - cumulative[lo]) / (i + 1 - lo)


def difference(values, lag):
    out = np.zeros_like(values)
    out[lag:] = values[lag:] - values[:-lag]
    return out


def engineered_covariates(ext):
    """Extra continuous covariates built only from the external file (never from the target).

    The target's level depends on how conditions have evolved over the preceding hours, not only on
    the current row, so trailing summaries are added. Every input is known on the full 43,824-step
    grid, so these are available across the forecast horizon as well.
    """
    a, b, c = (ext[f"feature_{k}"].to_numpy(np.float64) for k in "ABC")
    d = ext["feature_D"].to_numpy(np.float64)
    e, f = ext["feature_E"].to_numpy(np.float64), ext["feature_F"].to_numpy(np.float64)
    states = ext[BINARY].to_numpy(np.float64)
    # D accumulates while the categorical state persists and restarts when it changes; its step
    # increment recovers a per-step magnitude (the value itself where it restarted).
    increment = difference(d, 1)
    restarted = increment < 0
    increment[restarted] = d[restarted]
    columns = {
        "B_minus_A": b - a,
        "B_minus_A_mean24": trailing_mean(b - a, 24),
        "A_mean24": trailing_mean(a, 24),
        "C_diff3": difference(c, 3),
        "C_diff24": difference(c, 24),
        "logD_mean6": trailing_mean(np.log1p(d), 6),
        "logD_mean24": trailing_mean(np.log1p(d), 24),
        "D_step_mean12": trailing_mean(increment, 12),
        "E_active24": trailing_mean((e > 0).astype(float), 24),
        "F_active24": trailing_mean((f > 0).astype(float), 24),
    }
    for j, name in enumerate(BINARY):
        columns[f"{name}_share24"] = trailing_mean(states[:, j], 24)
        columns[f"{name}_x_D_step6"] = states[:, j] * trailing_mean(increment, 6)
    return columns


def load(exog: str, train_end: int, features: str = "basic", target_transform: str = "none"):
    """Return target (scaled), covariates (scaled), the target scaler, and the raw target.

    All statistics are estimated on positions [0, train_end) only, so validation never leaks into
    scaling. Scaling is global (one mean/std for the whole series), not per window: the long-run level
    is the strongest naive predictor of this series, and per-window normalisation would discard it.
    With target_transform="log1p" the scaler applies to log(1 + y); predict() inverts it.
    """
    y = pd.read_csv(DATA / "student_train.csv")["value"].to_numpy(np.float64)
    # The distributed CSV is padded for readability (", feature_A", etc.).
    # skipinitialspace keeps both padded and conventional copies compatible.
    ext = pd.read_csv(DATA / "optional_external_data.csv", skipinitialspace=True)
    base = np.log1p(y) if target_transform == "log1p" else y
    mean, std = base[:train_end].mean(), base[:train_end].std()
    target = ((base - mean) / std).astype(np.float32)
    if exog == "none":
        covariates = np.zeros((len(ext), 0), dtype=np.float32)
    else:
        continuous = []
        for name in CONTINUOUS:
            column = ext[name].to_numpy(np.float64)
            continuous.append(np.log1p(column) if name in LOG_FIRST else column)
        if features == "engineered":
            continuous += list(engineered_covariates(ext).values())
        parts = [(column - column[:train_end].mean()) / (column[:train_end].std() + 1e-8)
                 for column in continuous]
        parts += [ext[name].to_numpy(np.float64) for name in BINARY]
        covariates = np.stack(parts, axis=1).astype(np.float32)
    return target, covariates, (mean, std), y


def load_for_config(config, train_end):
    """load() with the data options a run was trained with (older runs default to basic/none)."""
    return load(config["exog"], train_end, config.get("features", "basic"),
                config.get("target_transform", "none"))


def window_index(origins, seq_len, label_len):
    enc = origins[:, None] + np.arange(-seq_len, 0)[None, :]
    dec = origins[:, None] + np.arange(-label_len, PRED_LEN)[None, :]
    tgt = origins[:, None] + np.arange(PRED_LEN)[None, :]
    return enc, dec, tgt


def batches(target, covariates, origins, seq_len, label_len):
    enc, dec, tgt = window_index(origins, seq_len, label_len)
    x_enc = torch.from_numpy(target[enc][..., None])
    c_enc = torch.from_numpy(covariates[enc])
    c_dec = torch.from_numpy(covariates[dec])
    y = torch.from_numpy(target[tgt]) if tgt.max() < len(target) else None
    return x_enc, c_enc, c_dec, y


def resolve_device(requested: str):
    """Resolve auto/cpu/cuda and fail clearly when CUDA was requested but is unavailable."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def to_device(tensors, device):
    """Move a batch to the selected device; preserve None for unavailable future targets."""
    non_blocking = device.type == "cuda"
    return tuple(t.to(device, non_blocking=non_blocking) if t is not None else None for t in tensors)


# ----------------------------------------------------------------------------- model

class MovingAvg(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):                                   # [B,L,C]
        q = (self.kernel - 1) // 2
        padded = F.pad(x.transpose(1, 2), (q, self.kernel - 1 - q), mode="replicate")
        return F.avg_pool1d(padded, self.kernel, stride=1).transpose(1, 2)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.average = MovingAvg(kernel)

    def forward(self, x):
        trend = self.average(x)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """Period-based dependencies: FFT autocorrelation scores, top-k delays, delay aggregation."""

    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def forward(self, q, k, v):                             # q [B,L,H,E]; k, v [B,S,H,E]
        B, L, H, E = q.shape
        S = k.shape[1]
        output_dtype = q.dtype
        if L > S:
            pad = torch.zeros_like(q[:, : L - S])
            k, v = torch.cat([k, pad], 1), torch.cat([v, pad], 1)
        else:
            k, v = k[:, :L], v[:, :L]
        # CUDA half-precision FFT only supports power-of-two lengths. Autoformer lengths such as
        # 168 and 216 are not powers of two, so compute correlation safely in float32 under AMP.
        q_f = torch.fft.rfft(q.float().permute(0, 2, 3, 1), dim=-1)
        k_f = torch.fft.rfft(k.float().permute(0, 2, 3, 1), dim=-1)
        corr = torch.fft.irfft(q_f * k_f.conj(), n=L, dim=-1)   # [B,H,E,L], one score per delay
        score = corr.mean(dim=(1, 2))                           # [B,L]
        top_k = max(1, int(self.factor * math.log(L)))
        weights, delays = torch.topk(score, top_k, dim=-1)      # [B,K]
        weights = torch.softmax(weights, dim=-1)
        values = v.float().permute(0, 2, 3, 1)                  # [B,H,E,L]
        positions = torch.arange(L, device=values.device)
        out = torch.zeros_like(values)
        for j in range(top_k):
            # corr[tau] = sum_t q[t] k[t - tau], so a high score means position t should read t - tau
            index = (positions[None, :] - delays[:, j:j + 1]) % L
            index = index[:, None, None, :].expand(B, H, E, L)
            out = out + weights[:, j].view(B, 1, 1, 1) * values.gather(-1, index)
        return out.permute(0, 3, 1, 2).to(output_dtype)         # [B,L,H,E]


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, heads, factor):
        super().__init__()
        self.heads = heads
        self.inner = AutoCorrelation(factor)
        self.q, self.k, self.v, self.o = (nn.Linear(d_model, d_model) for _ in range(4))

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        h = self.heads
        out = self.inner(self.q(queries).view(B, L, h, -1), self.k(keys).view(B, S, h, -1),
                         self.v(values).view(B, S, h, -1))
        return self.o(out.reshape(B, L, -1))


class SeasonalNorm(nn.Module):
    """Autoformer's layer norm for the seasonal part: normalise, then remove the time mean."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, kernel, factor, dropout):
        super().__init__()
        self.attention = AutoCorrelationLayer(d_model, heads, factor)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.decomp1, self.decomp2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.drop(self.attention(x, x, x)))
        x, _ = self.decomp2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, kernel, factor, dropout):
        super().__init__()
        self.self_attention = AutoCorrelationLayer(d_model, heads, factor)
        self.cross_attention = AutoCorrelationLayer(d_model, heads, factor)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomp(kernel) for _ in range(3))
        self.trend_projection = nn.Conv1d(d_model, 1, 3, padding=1, padding_mode="circular",
                                          bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        x, trend1 = self.decomp1(x + self.drop(self.self_attention(x, x, x)))
        x, trend2 = self.decomp2(x + self.drop(self.cross_attention(x, memory, memory)))
        x, trend3 = self.decomp3(x + self.drop(self.ff(x)))
        trend = self.trend_projection((trend1 + trend2 + trend3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class Embedding(nn.Module):
    """Value embedding (local conv, as in Autoformer) plus a linear covariate embedding."""

    def __init__(self, n_covariates, d_model, dropout):
        super().__init__()
        self.value = nn.Conv1d(1, d_model, 3, padding=1, padding_mode="circular", bias=False)
        self.covariate = nn.Linear(n_covariates, d_model) if n_covariates else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, covariates):
        out = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.covariate is not None:
            out = out + self.covariate(covariates)
        return self.drop(out)


class ExogTrendInit(nn.Module):
    """Known covariates -> one trend level per decoder step, from a short temporal convolution.

    Standard Autoformer starts the trend-cyclical branch from the flat context mean; that branch is
    the only one whose time-mean is not removed by SeasonalNorm, so it is where covariates can move
    the forecast's level. The output layer is zero-initialised, so training starts from the
    standard Autoformer initialisation.
    """

    def __init__(self, n_covariates, hidden, kernel, dropout):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(n_covariates, hidden, kernel)
        self.out = nn.Conv1d(hidden, 1, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, covariates):                          # [B,T,C] -> [B,T,1]
        q = (self.kernel - 1) // 2
        x = F.pad(covariates.transpose(1, 2), (q, self.kernel - 1 - q), mode="replicate")
        return self.out(self.drop(F.gelu(self.conv(x)))).transpose(1, 2)


class Autoformer(nn.Module):
    def __init__(self, n_covariates, seq_len=168, label_len=48, d_model=32, heads=4, d_ff=64,
                 e_layers=2, d_layers=1, kernel=25, factor=1.0, dropout=0.1,
                 direct_exog=False, exog_hidden=16, level_residual=False, level_window=24,
                 trend_init="mean", trend_hidden=16, trend_kernel=13,
                 anchor=False, anchor_tau=6.0, target_transform="none"):
        super().__init__()
        self.seq_len, self.label_len = seq_len, label_len
        self.level_window = level_window
        self.target_transform = target_transform
        if target_transform == "log1p":
            # Per-horizon smearing factor E[exp(log residual)]: converts a log-space forecast into
            # a mean forecast. A buffer, not a parameter: estimated from training residuals.
            self.register_buffer("smear", torch.ones(PRED_LEN))
        self.decomp = SeriesDecomp(kernel)
        self.enc_embedding = Embedding(n_covariates, d_model, dropout)
        self.dec_embedding = Embedding(n_covariates, d_model, dropout)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, heads, d_ff, kernel, factor, dropout)
                                     for _ in range(e_layers))
        self.enc_norm = SeasonalNorm(d_model)
        self.decoder = nn.ModuleList(DecoderLayer(d_model, heads, d_ff, kernel, factor, dropout)
                                     for _ in range(d_layers))
        self.dec_norm = SeasonalNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

        # A zero-initialised shortcut lets known future conditions make a direct
        # per-horizon correction without changing the baseline at initialisation.
        self.direct_exog = bool(direct_exog and n_covariates)
        self.exog_head = None
        if self.direct_exog:
            self.exog_head = nn.Sequential(
                nn.Linear(n_covariates, exog_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(exog_hidden, 1),
            )
            nn.init.zeros_(self.exog_head[-1].weight)
            nn.init.zeros_(self.exog_head[-1].bias)

        # This bounded gate can shift the forecast when the recent target level
        # differs from the full context. Zero initialisation recovers baseline.
        self.level_residual = bool(level_residual)
        self.level_gate = nn.Parameter(torch.zeros(())) if self.level_residual else None

        self.trend_init = (ExogTrendInit(n_covariates, trend_hidden, trend_kernel, dropout)
                           if trend_init == "exog" and n_covariates else None)

        # Short-horizon anchor: blend toward the last observed value with weight
        # sigmoid(a) * exp(-h / tau), so the first steps can follow persistence and later steps not.
        self.anchor = bool(anchor)
        if self.anchor:
            self.anchor_logit = nn.Parameter(torch.zeros(()))
            self.anchor_log_tau = nn.Parameter(torch.tensor(math.log(anchor_tau)))

    def forward(self, x_enc, c_enc, c_dec):                    # x_enc [B,L,1] -> [B,PRED_LEN]
        mean = x_enc.mean(dim=1, keepdim=True).expand(-1, PRED_LEN, -1)
        zeros = torch.zeros_like(mean)
        seasonal, trend = self.decomp(x_enc)
        if self.trend_init is not None:
            # Convolve over label + horizon so each future step sees the covariates around it.
            mean = mean + self.trend_init(c_dec)[:, -PRED_LEN:]
        trend = torch.cat([trend[:, -self.label_len:], mean], dim=1)
        seasonal = torch.cat([seasonal[:, -self.label_len:], zeros], dim=1)

        memory = self.enc_embedding(x_enc, c_enc)
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.enc_norm(memory)

        x = self.dec_embedding(seasonal, c_dec)
        for layer in self.decoder:
            x, residual_trend = layer(x, memory)
            trend = trend + residual_trend
        out = trend + self.projection(self.dec_norm(x))
        forecast = out[:, -PRED_LEN:, 0]
        if self.exog_head is not None:
            forecast = forecast + self.exog_head(c_dec[:, -PRED_LEN:]).squeeze(-1)
        if self.level_gate is not None:
            window = min(self.level_window, x_enc.shape[1])
            recent_level = x_enc[:, -window:, 0].mean(dim=1)
            context_level = x_enc[:, :, 0].mean(dim=1)
            correction = torch.tanh(self.level_gate) * (recent_level - context_level)
            forecast = forecast + correction[:, None]
        if self.anchor:
            horizon = torch.arange(PRED_LEN, device=forecast.device, dtype=forecast.dtype)
            weight = torch.sigmoid(self.anchor_logit) * torch.exp(-horizon / self.anchor_log_tau.exp())
            forecast = forecast + weight[None, :] * (x_enc[:, -1:, 0] - forecast)
        return forecast


# ----------------------------------------------------------------------------- training

def metrics(prediction, truth):
    error = prediction - truth
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAE": float(np.mean(np.abs(error))),
            "sMAPE": float(100 * np.mean(2 * np.abs(error) / (np.abs(truth) + np.abs(prediction) + 1e-9)))}


@torch.no_grad()
def predict(model, target, covariates, origins, seq_len, label_len, scaler, chunk=256,
            device=None, amp=False, clip=True):
    model.eval()
    device = device or next(model.parameters()).device
    use_amp = bool(amp and device.type == "cuda")
    out = []
    for i in range(0, len(origins), chunk):
        batch = batches(target, covariates, origins[i:i + chunk], seq_len, label_len)
        x_enc, c_enc, c_dec, _ = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(x_enc, c_enc, c_dec)
        out.append(prediction.float().cpu().numpy())
    mean, std = scaler
    decoded = np.concatenate(out) * std + mean
    if getattr(model, "target_transform", "none") == "log1p":
        decoded = np.exp(decoded) * model.smear.cpu().numpy()[None, :] - 1.0
    return np.clip(decoded, 0, None) if clip else decoded


@torch.no_grad()
def estimate_smearing(model, target, covariates, origins, seq_len, label_len, std, device,
                      amp=False, chunk=256):
    """Set model.smear to the per-horizon mean of exp(log-space residual) on training windows.

    Inverting a log-space forecast with exp() gives roughly the median, which underestimates the
    mean on a right-skewed series and so costs RMSE. Duan's smearing estimate corrects that bias.
    Training residuals are a little optimistic, so this slightly under-corrects.
    """
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    total = torch.zeros(PRED_LEN, dtype=torch.float64)
    for i in range(0, len(origins), chunk):
        batch = to_device(batches(target, covariates, origins[i:i + chunk], seq_len, label_len),
                          device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(*batch[:3])
        residual = (batch[3] - prediction.float()).double().cpu() * std
        total += torch.exp(residual).sum(dim=0)
    model.smear.copy_((total / len(origins)).float().to(model.smear.device))


ARCHITECTURE_KEYS = ("seq_len", "label_len", "d_model", "heads", "d_ff", "e_layers", "d_layers",
                     "kernel", "factor", "dropout", "direct_exog", "exog_hidden", "level_residual",
                     "level_window", "trend_init", "trend_hidden", "trend_kernel", "anchor",
                     "anchor_tau", "target_transform")


def model_from_config(n_covariates, config):
    """Recreate both legacy and improved checkpoints from saved configuration."""
    return Autoformer(n_covariates, **{k: config[k] for k in ARCHITECTURE_KEYS if k in config})


def design_name(direct_exog, level_residual, trend_init="mean", anchor=False):
    parts = []
    if trend_init == "exog":
        parts.append("exogtrend")
    if direct_exog:
        parts.append("direct")
    if level_residual:
        parts.append("level")
    if anchor:
        parts.append("anchor")
    legacy = {("direct", "level"): "direct_level", ("direct",): "direct_exog",
              ("level",): "level_residual", (): "baseline"}
    return legacy.get(tuple(parts), "_".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exog", choices=["all", "none"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--final", action="store_true",
                        help="fit on the whole history for --epochs and forecast the hidden 168 steps")
    parser.add_argument("--history-end", type=int,
                        help="final-mode backtest cutoff: train on only this many observed rows")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--direct-exog", action="store_true",
                        help="add a small direct future-covariate residual head")
    parser.add_argument("--exog-hidden", type=int, default=16)
    parser.add_argument("--level-residual", action="store_true",
                        help="add a bounded recent-level correction to the full horizon")
    parser.add_argument("--level-window", type=int, default=24)
    parser.add_argument("--features", choices=["basic", "engineered"], default="basic",
                        help="engineered adds trailing summaries of the external variables")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none",
                        help="log1p trains on log(1+y) and inverts with a smearing correction")
    parser.add_argument("--trend-init", choices=["mean", "exog"], default="mean",
                        help="exog adds a covariate-conditioned level to the decoder trend start")
    parser.add_argument("--trend-hidden", type=int, default=16)
    parser.add_argument("--trend-kernel", type=int, default=13)
    parser.add_argument("--anchor", action="store_true",
                        help="learned, decaying blend toward the last observed value")
    parser.add_argument("--anchor-tau", type=float, default=6.0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=25, help="moving-average decomposition kernel")
    parser.add_argument("--factor", type=float, default=1.0, help="top-k = factor * ln(L) delays")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recent-blocks", type=int, default=24,
                        help="also report metrics on this many most recent validation blocks")
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["cosine", "legacy-half", "none"],
                        default="cosine")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="training device; auto selects CUDA when available")
    parser.add_argument("--amp", action="store_true",
                        help="use CUDA mixed precision for faster training and lower memory use")
    parser.add_argument("--selection-epochs", type=int, default=0,
                        help="epochs already spent selecting this final model; reporting only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.exog_hidden < 1:
        parser.error("--exog-hidden must be positive")
    if not 1 <= args.level_window <= args.seq_len:
        parser.error("--level-window must be between 1 and --seq-len")
    if args.d_model % args.heads:
        parser.error("--d-model must be divisible by --heads")
    if args.trend_init == "exog" and args.exog == "none":
        parser.error("--trend-init exog needs --exog all")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    n_history = len(pd.read_csv(DATA / "student_train.csv"))
    if args.history_end is not None:
        if not args.final or not args.seq_len + PRED_LEN <= args.history_end <= n_history - PRED_LEN:
            parser.error('--history-end requires --final and a cutoff with 168 known future targets')
        n_history = args.history_end
    val_start = n_history - VAL_LEN
    train_end = n_history if args.final else val_start
    target, covariates, scaler, raw = load(args.exog, train_end, args.features,
                                           args.target_transform)

    # training origins: every window's 168-step target ends before train_end (no overlap with validation)
    train_origins = np.arange(args.seq_len, train_end - PRED_LEN + 1, args.train_stride)
    val_origins = np.arange(val_start, n_history - PRED_LEN + 1, VAL_STRIDE)
    recent = slice(-min(args.recent_blocks, len(val_origins)), None)
    val_truth = np.stack([raw[o:o + PRED_LEN] for o in val_origins]) if not args.final else None
    # Fixed training subsample for the log-space smearing estimate (training data only).
    smear_origins = np.random.default_rng(args.seed).choice(
        train_origins, size=min(2048, len(train_origins)), replace=False)

    model = model_from_config(covariates.shape[1], vars(args)).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    ) if args.lr_schedule == "cosine" else None)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = np.random.default_rng(args.seed)

    design = design_name(args.direct_exog, args.level_residual, args.trend_init, args.anchor)
    print(json.dumps({"device": str(device), "amp": use_amp, "design": design,
                      "parameters": parameters, "features": args.features,
                      "target_transform": args.target_transform,
                      "train_windows": len(train_origins), "validation_windows": len(val_origins),
                      "encoder_covariates": covariates.shape[1],
                      "future_covariates_in_decoder": args.exog != "none"}), flush=True)

    history, best, best_state, best_epoch, stale = [], float("inf"), None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        order = generator.permutation(train_origins)
        losses = []
        for i in range(0, len(order), args.batch):
            batch = batches(target, covariates, order[i:i + args.batch],
                            args.seq_len, args.label_len)
            x_enc, c_enc, c_dec, y = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.mse_loss(model(x_enc, c_enc, c_dec), y)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            losses.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        elif args.lr_schedule == "legacy-half" and epoch >= 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        row = {"epoch": epoch, "train_mse_scaled": float(np.mean(losses)),
               "learning_rate": epoch_lr,
               "seconds": round(time.perf_counter() - started, 1)}
        if not args.final:
            if args.target_transform == "log1p":
                estimate_smearing(model, target, covariates, smear_origins, args.seq_len,
                                  args.label_len, scaler[1], device, amp=use_amp)
            raw_prediction = predict(model, target, covariates, val_origins, args.seq_len,
                                     args.label_len, scaler, device=device, amp=use_amp, clip=False)
            prediction = np.clip(raw_prediction, 0, None)
            row.update(metrics(prediction, val_truth))
            row.update({f"recent_{k}": v
                        for k, v in metrics(prediction[recent], val_truth[recent]).items()})
            row.update({"raw_negative_count": int((raw_prediction < 0).sum()),
                        "raw_min": float(raw_prediction.min()),
                        "clipped_zero_count": int((prediction == 0).sum())})
            if row["RMSE"] < best:
                best, best_epoch, stale = row["RMSE"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if not args.final and stale >= args.patience:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "exog": args.exog, "design": design,
               "final": args.final, "parameters": parameters,
               "device_used": str(device), "amp_used": use_amp,
               "epochs_run": len(history),
               "declared_epochs": args.selection_epochs + len(history),
               "history": history,
               "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    if args.final:
        if args.target_transform == "log1p":
            estimate_smearing(model, target, covariates, smear_origins, args.seq_len,
                              args.label_len, scaler[1], device, amp=use_amp)
        raw_forecast = predict(model, target, covariates, np.array([n_history]), args.seq_len,
                               args.label_len, scaler, device=device, amp=use_amp, clip=False)[0]
        forecast = np.clip(raw_forecast, 0, None)
        pd.DataFrame({"time_idx": np.arange(n_history + 1, n_history + PRED_LEN + 1),
                      "value": forecast}).to_csv(args.out / "forecast.csv", index=False)
        pd.DataFrame({"time_idx": np.arange(n_history + 1, n_history + PRED_LEN + 1),
                      "value": raw_forecast}).to_csv(args.out / "forecast_unclipped.csv", index=False)
        (args.out / "predictions.txt").write_text(", ".join(f"{v:.4f}" for v in forecast))
        summary["forecast_diagnostics"] = {
            "raw_negative_count": int((raw_forecast < 0).sum()),
            "raw_min": float(raw_forecast.min()),
            "clipped_zero_count": int((forecast == 0).sum()),
            "clipped_mean": float(forecast.mean()),
            "clipped_max": float(forecast.max()),
        }
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   args.out / "model.pt")
    else:
        model.load_state_dict(best_state)
        summary.update({"best_epoch": best_epoch, "best_val": history[best_epoch - 1]})
        torch.save(best_state, args.out / "model.pt")
        # Saved so seed ensembles can be scored on validation without reloading every model.
        np.save(args.out / "val_predictions.npy",
                predict(model, target, covariates, val_origins, args.seq_len, args.label_len,
                        scaler, device=device, amp=use_amp))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"parameters={parameters} epochs_run={len(history)}"
          + ("" if args.final else f" best_epoch={best_epoch} best_val={summary['best_val']}"))


if __name__ == "__main__":
    main()
