"""The staff reader: a picture of one staff in, its tokens out.

A ConvNeXt-style network turns the staff, HEIGHT pixels tall, into a grid
five rows high and an eighth of its width across. Two transformer layers
let every place on the staff see every other, so a key signature or a
clef informs a note far to its right. A transformer decoder then writes
the staff's tokens one at a time, looking back over the grid."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import staffcrop, tokens
from . import settings

# Where models read: the GPU, or the CPU where the device setting says so
# (to try a command while the GPU trains).
DEVICE = settings.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')

HEIGHT = 160  # pixels from the top of the window to its bottom
WINDOW = 14.0  # staff spaces the window spans, the staff in its middle
MAX_WIDTH = 2400
MAX_TOKENS = 768
VOCAB = tokens.vocabulary()
INDEX = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = INDEX['<pad>'], INDEX['<s>'], INDEX['</s>']


def window(stored: np.ndarray, *, spaces: float = WINDOW, shift: float = 0.0, stretch: float = 1.0) -> np.ndarray:
    """The network's view of a staff cut by staffcrop: [spaces] staff
    spaces tall around the staff, moved [shift] spaces down, scaled to
    HEIGHT pixels, and [stretch] wider or narrower."""
    px = staffcrop.PX_PER_SPACE
    center = stored.shape[0] / 2 + shift * px
    top = center - spaces * px / 2
    zoom = HEIGHT / (spaces * px)
    width = min(MAX_WIDTH, max(32, int(round(stored.shape[1] * zoom * stretch))))
    m = np.array([[zoom * stretch, 0, 0], [0, zoom, -top * zoom]], np.float32)
    return cv2.warpAffine(stored, m, (width, HEIGHT), flags=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR,
                          borderValue=255)


def encode(seq: list[str]) -> list[int]:
    return [BOS] + [INDEX[t] for t in seq] + [EOS]


def decode(ids: list[int]) -> list[str]:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i not in (BOS, PAD):
            out.append(VOCAB[i])
    return out


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    """A ConvNeXt block: a wide depthwise convolution, then a pointwise
    MLP, added back to its input."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.gamma * self.mlp(self.norm(y))
        return x + y.permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(self, dims=(64, 128, 256, 384), depths=(2, 2, 6, 3), strides=((4, 4), (2, 2), (2, 1), (2, 1))):
        super().__init__()
        # By default height falls by 32 and width by 8: (160, W) -> (5, W/8).
        layers = []
        for k, (dim, depth, stride) in enumerate(zip(dims, depths, strides)):
            if k == 0:
                layers += [nn.Conv2d(1, dim, stride, stride=stride), LayerNorm2d(dim)]
            else:
                layers += [LayerNorm2d(dims[k - 1]), nn.Conv2d(dims[k - 1], dim, stride, stride=stride)]
            layers += [Block(dim) for _ in range(depth)]
        self.net = nn.Sequential(*layers)
        self.dim = dims[-1]

    def forward(self, x):
        return self.net(x)


def sinusoid(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    out = torch.zeros(n, dim, device=device)
    out[:, 0::2] = torch.sin(pos * freq)
    out[:, 1::2] = torch.cos(pos * freq)
    return out


class Reader(nn.Module):
    def __init__(self, dim: int = 384, heads: int = 8, context_layers: int = 2, decoder_layers: int = 6):
        super().__init__()
        self.encoder = Encoder()
        self.rows = nn.Parameter(torch.zeros(HEIGHT // 32, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, context_layers)
        self.embed = nn.Embedding(len(VOCAB), dim, padding_idx=PAD)
        self.positions = nn.Embedding(MAX_TOKENS + 2, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, len(VOCAB))

    def memory(self, images: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The staff as a sequence of places, and which are padding."""
        f = self.encoder(images)  # B, C, 5, W/8
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1) + self.rows[None, :, None, :] + sinusoid(w, c, f.device)[None, None]
        f = f.reshape(b, h * w, c)
        cols = torch.arange(w, device=f.device)[None, :] >= (widths[:, None] + 7) // 8
        pad = cols[:, None, :].expand(b, h, w).reshape(b, h * w)
        return self.context(f, src_key_padding_mask=pad), pad

    def logits(self, memory, pad, seq):
        t = seq.shape[1]
        x = self.embed(seq) + self.positions(torch.arange(t, device=seq.device))[None]
        causal = torch.triu(torch.ones(t, t, device=seq.device, dtype=torch.bool), 1)
        y = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=seq == PAD,
                         memory_key_padding_mask=pad)
        return self.out(self.norm(y))

    def forward(self, images, widths, seq):
        memory, pad = self.memory(images, widths)
        return self.logits(memory, pad, seq)

    @torch.no_grad()
    def read(self, images: torch.Tensor, widths: torch.Tensor, max_len: int = MAX_TOKENS,
             bar_limits: list | None = None, prefixes: list | None = None,
             bar_counts: list | None = None) -> list[list[str]]:
        """Greedy reading of a batch of staves that can never loop: once a
        bar's notes fill twice its time signature (the staff's own time
        token, or [bar_limits] given by the caller), or BAR_CAP notes, only
        a chord note, a mark on the last note or a barline may follow.
        A staff's reading may be made to begin with given tokens
        ([prefixes]), such as a key the rest of its system agrees on. Where
        the layout counted a staff's bars ([bar_counts]), its reading ends
        after that many barlines, no sooner and no later: in a run of like
        bars, such as whole-bar rests, the model alone can lose count."""
        memory, pad = self.memory(images, widths)
        n = images.shape[0]
        seq = torch.full((n, 1), BOS, device=images.device, dtype=torch.long)
        done = torch.zeros(n, dtype=torch.bool, device=images.device)
        bars = [Budget(bar_limits[i] if bar_limits else None) for i in range(n)]
        closed = [0] * n
        for step in range(max_len + 1):
            logits = self.logits(memory, pad, seq)[:, -1].float()
            for i, b in enumerate(bars):
                if not done[i] and b.full():
                    logits[i, STARTS] = float('-inf')
                want = bar_counts[i] if bar_counts else None
                if not done[i] and want:
                    if closed[i] < want:
                        logits[i, EOS] = float('-inf')
                    else:
                        logits[i] = float('-inf')
                        logits[i, EOS] = 0.0
            nxt = logits.argmax(-1)
            if prefixes:
                for i, prefix in enumerate(prefixes):
                    if prefix and step < len(prefix):
                        nxt[i] = INDEX[prefix[step]]
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    bars[i].add(VOCAB[t])
                    closed[i] += VOCAB[t] in tokens.BARLINES
            done |= nxt == EOS
            if done.all():
                break
        return [decode(s) for s in seq.tolist()]


    @torch.no_grad()
    def chances(self, images: torch.Tensor, widths: torch.Tensor, starts: list[list[str]]) -> torch.Tensor:
        """For each staff, how likely each token is to come next after the
        tokens [starts] (all of one length): a distribution over the
        vocabulary per staff."""
        memory, pad = self.memory(images, widths)
        seq = torch.tensor([[BOS] + [INDEX[t] for t in st] for st in starts], device=images.device)
        return self.logits(memory, pad, seq)[:, -1].float().softmax(-1)


BAR_CAP = 64  # notes in one bar; no music we read has more
STARTS = [INDEX[t] for t in VOCAB if t.startswith(('N:', 'acc:')) or t in ('R', 'R:measure', 'gr', 'S')]


class Budget:
    """How full the bar being read is, against its time signature."""

    def __init__(self, limit=None):
        from fractions import Fraction
        self.limit = limit
        self.used = Fraction(0)
        self.notes = 0
        self.last = None  # the note whose look is still being written
        self.after_chord = False

    def full(self) -> bool:
        if self.after_chord:
            return False
        if self.notes >= BAR_CAP:
            return True
        # Only a bar twice as long as its time signature is cut: the time read
        # may be wrong or stale, and a barline is the model's to place.
        return self.limit is not None and self.used + self._pending() >= 2 * self.limit

    def _pending(self):
        return tokens.length(self.last) if self.last and not self.last.get('chord') and not self.last.get('grace') \
            and not self.last.get('measure') else 0

    def add(self, t: str):
        from fractions import Fraction
        if t in tokens.BARLINES:
            self.used, self.notes, self.last, self.after_chord = Fraction(0), 0, None, False
            return
        if t == '&':  # the next voice starts again at the bar's beginning
            self.used, self.last, self.after_chord = Fraction(0), None, False
            return
        if t.startswith('time:'):
            b, bt, _ = tokens._time_value(t[5:])
            self.limit = Fraction(4 * b, bt)
            return
        if t == '+':
            self.after_chord = True
            return
        if t.startswith(('N:', 'R')) or t in ('gr', 'S'):
            self.used += self._pending()
            if t == 'R:measure' and self.limit is not None:
                self.used = self.limit
            self.last = {'type': 'quarter', 'dots': 0, 'tuplet': None, 'chord': self.after_chord,
                         'grace': t == 'gr', 'measure': t == 'R:measure'}
            self.after_chord = False
            self.notes += 1
            if t == 'R:measure':
                self.last = None
        elif self.last is not None:
            if t.startswith('d:'):
                self.last['type'] = t[2:]
            elif t == '.':
                self.last['dots'] += 1
            elif t.startswith('tup:'):
                self.last['tuplet'] = t[4:]


def batch_images(images: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Gray staves as one tensor, ink bright, padded on the right with
    paper to a width the network divides evenly."""
    widths = [im.shape[1] for im in images]
    w = max(32, ((max(widths) + 31) // 32) * 32)
    out = np.zeros((len(images), 1, HEIGHT, w), np.float32)
    for k, im in enumerate(images):
        out[k, 0, :, :im.shape[1]] = (255 - im.astype(np.float32)) / 255
    return torch.from_numpy(out).to(device), torch.tensor(widths, device=device)


def grown(weights: dict, model: nn.Module) -> dict:
    """A checkpoint's weights fitted to a model whose vocabulary has grown
    since: new tokens only ever join the end, so the old rows are copied
    and the new ones keep the model's fresh values."""
    mine = model.state_dict()
    out = dict(weights)
    for name in ('embed.weight', 'out.weight', 'out.bias'):
        old, new = weights.get(name), mine[name]
        if old is not None and old.shape != new.shape and old.shape[1:] == new.shape[1:]:
            fitted = new.clone().to(old.device)
            fitted[:old.shape[0]] = old
            if name == 'out.bias':
                # A token the checkpoint never learned starts unlikely, so an
                # older model reads as it did until training teaches it.
                fitted[old.shape[0]:] = -8.0
            out[name] = fitted
    return out


def load(path, device=DEVICE) -> Reader:
    state = torch.load(path, map_location=device, weights_only=True)
    model = Reader(**state.get('config', {}))
    model.load_state_dict(grown(state['model'], model))
    return model.to(device).eval()
