"""Online adaptation: replay buffer, CUSUM regime detection, incremental fitting.

Financial relationships drift, so a net trained once on 2024 and applied to 2026 goes
stale. But naively training only on recent data causes catastrophic forgetting: the
model loses the rare-regime behaviour it will need next time volatility spikes.

The compromise here is a replay buffer that mixes a recent window with a reservoir
sample of all history seen so far, plus a CUSUM detector on prediction error that
temporarily raises the learning rate when the error distribution shifts, so the model
adapts quickly to a genuine regime change without thrashing on noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplayBuffer:
    """Recent window plus a uniform reservoir sample of everything older.

    Reservoir sampling keeps an unbiased sample of the full history in bounded memory,
    which is what stops the net from forgetting regimes it has not seen for months.
    """

    capacity: int
    recent_fraction: float = 0.5
    seed: int = 0
    _recent: list = field(default_factory=list, repr=False)
    _reservoir: list = field(default_factory=list, repr=False)
    _seen: int = 0
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def recent_capacity(self) -> int:
        return max(1, int(self.capacity * self.recent_fraction))

    @property
    def reservoir_capacity(self) -> int:
        return max(1, self.capacity - self.recent_capacity)

    def add(self, x: np.ndarray, g: np.ndarray, y: np.ndarray) -> None:
        for i in range(len(y)):
            item = (x[i], g[i], y[i])
            self._recent.append(item)
            # Evicted recent items get a chance to enter the long-term reservoir.
            if len(self._recent) > self.recent_capacity:
                evicted = self._recent.pop(0)
                self._offer(evicted)
            self._seen += 1

    def _offer(self, item: tuple) -> None:
        assert self._rng is not None
        if len(self._reservoir) < self.reservoir_capacity:
            self._reservoir.append(item)
            return
        j = self._rng.integers(0, self._seen)
        if j < self.reservoir_capacity:
            self._reservoir[int(j)] = item

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._rng is not None
        pool = self._recent + self._reservoir
        if not pool:
            raise ValueError("replay buffer is empty")
        n = min(n, len(pool))
        idx = self._rng.choice(len(pool), size=n, replace=False)
        xs, gs, ys = zip(*(pool[int(i)] for i in idx), strict=True)
        return np.stack(xs), np.stack(gs), np.asarray(ys, dtype="float32")

    def __len__(self) -> int:
        return len(self._recent) + len(self._reservoir)


@dataclass
class CusumDetector:
    """Two-sided CUSUM on standardised prediction error.

    Fires when cumulative drift in error magnitude exceeds a threshold, which is the
    signal that the learned relationship has moved and the net should adapt faster.
    """

    threshold: float = 5.0
    drift: float = 0.5
    pos: float = 0.0
    neg: float = 0.0
    mean: float = 0.0
    var: float = 1.0
    count: int = 0

    def update(self, error: float) -> bool:
        self.count += 1
        # Welford update keeps a running standardisation without storing history.
        delta = error - self.mean
        self.mean += delta / self.count
        self.var += delta * (error - self.mean)
        sd = np.sqrt(self.var / max(self.count - 1, 1)) if self.count > 1 else 1.0
        z = (error - self.mean) / max(sd, 1e-9)
        self.pos = max(0.0, self.pos + z - self.drift)
        self.neg = max(0.0, self.neg - z - self.drift)
        if max(self.pos, self.neg) > self.threshold:
            self.pos = self.neg = 0.0
            return True
        return False


@dataclass
class OnlineTrainer:
    """Wraps a torch module with replay-based incremental training."""

    model: object
    cfg: object
    n_features: int
    n_gate: int
    buffer: ReplayBuffer = field(init=False)
    detector: CusumDetector = field(init=False)
    _opt: object = field(default=None, repr=False)
    lr_multiplier: float = 1.0
    regime_shifts: int = 0

    def __post_init__(self) -> None:
        import torch  # noqa: PLC0415

        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            recent_fraction=self.cfg.recent_fraction,
            seed=self.cfg.seed,
        )
        self.detector = CusumDetector(threshold=self.cfg.cusum_threshold)
        self._opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def _run_epochs(self, n_epochs: int, batch_size: int) -> float:
        import torch  # noqa: PLC0415

        if len(self.buffer) < batch_size:
            return float("nan")
        self.model.train()
        last = float("nan")
        for _ in range(n_epochs):
            x, g, y = self.buffer.sample(batch_size)
            xt = torch.from_numpy(x)
            gt = torch.from_numpy(g)
            yt = torch.from_numpy(y)
            pred = self.model(xt, gt)
            # Huber rather than MSE: hourly crypto residuals are heavy-tailed and a
            # squared loss would let a handful of jumps dominate every gradient step.
            loss = torch.nn.functional.huber_loss(pred, yt, delta=1.0)
            self._opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._opt.step()
            last = float(loss.item())
        return last

    def warmup(self, x: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
        self.buffer.add(x, g, y)
        return self._run_epochs(self.cfg.warmup_epochs, self.cfg.batch_size)

    def observe(
        self, x: np.ndarray, g: np.ndarray, y: np.ndarray, errors: np.ndarray, *, n_chunks: int = 20
    ) -> float:
        """Ingest a new block of realised outcomes and adapt.

        Errors are fed to the detector in chunks rather than as a single fold-level
        mean: a CUSUM needs a sequence to accumulate against, and one observation per
        fold gives it 19 points across the whole backtest, far too few to ever fire.
        """
        self.buffer.add(x, g, y)
        shifted = False
        if len(errors):
            for chunk in np.array_split(np.abs(errors), min(n_chunks, max(1, len(errors)))):
                if chunk.size and self.detector.update(float(np.nanmean(chunk))):
                    shifted = True
        if shifted:
            self.regime_shifts += 1
            self.lr_multiplier = self.cfg.lr_boost
        else:
            # Decay back toward the base rate so a single shock does not leave the
            # model permanently jumpy.
            self.lr_multiplier = max(1.0, self.lr_multiplier * 0.7)
        for group in self._opt.param_groups:  # type: ignore[attr-defined]
            group["lr"] = self.cfg.lr * self.lr_multiplier
        return self._run_epochs(self.cfg.online_epochs, self.cfg.batch_size)

    def predict(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        if len(x) == 0:
            return np.empty(0, dtype="float32")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(x), torch.from_numpy(g))
        return out.numpy()
