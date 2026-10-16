"""The staff reader: a picture of one staff in, its tokens out.

A ConvNeXt-style network turns the staff, HEIGHT pixels tall, into a grid
five rows high and an eighth of its width across. Two transformer layers
let every place on the staff see every other, so a key signature or a
clef informs a note far to its right. A transformer decoder then writes
the staff's tokens one at a time, looking back over the grid."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
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
        out = self.out(self.norm(y))
        learned = getattr(self, 'learned', None)
        if learned is not None and not self.training:
            # A checkpoint trained before the vocabulary grew neither writes
            # nor weighs a token it never learned: this model's scores run
            # low, so even a low bias left such tokens a share of every
            # probability (2026-10-15). Training still reaches them.
            out[..., learned:] = float('-inf')
        return out

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
        run = [0] * n  # marks written since the last note, rest or barline
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
            # A mark never follows itself ('ped ped ...'), and no more than
            # MARK_RUN marks stand between two notes: a reading could
            # otherwise loop on one mark, or on two in turn ('ped hp> ped
            # hp< ...'), as E12's did on a few staves.
            for i in range(n):
                if done[i]:
                    continue
                last = int(seq[i, -1])
                if last in NO_REPEAT:
                    logits[i, last] = float('-inf')
                if run[i] >= MARK_RUN:
                    logits[i, MARKS_INDEX.to(logits.device)] = float('-inf')
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
                    run[i] = 0 if t in STARTS_SET or VOCAB[t] in tokens.BARLINES else run[i] + (t in NO_REPEAT)
            done |= nxt == EOS
            if done.all():
                break
        return [decode(s) for s in seq.tolist()]


    @torch.no_grad()
    def fits(self, images: torch.Tensor, widths: torch.Tensor, seqs: list[list[str]], limits: list | None = None,
             lead: list | None = None, most: float = 0.5, tries: int = 48,
             skip: list | None = None) -> list[list['Fit']]:
        """For each staff's reading [seqs], each bar a voice of which runs
        past its time signature, read again as triplets. A triplet printed
        without its "3" is read as plain notes, which overfills the bar by
        a third of the group; the notes and their pitches are right. So
        runs of three like notes in that voice are tried as triplets until
        the voice fills the bar (triplet_options), and the model scores
        each option given the picture. Each Fit says how much less likely
        the model found the best option than the bar as read (Fit.cost, in
        nats), so a caller accepts it only where the model was unsure
        (apply_fits). A staff most of whose bars overfill is left alone:
        its time signature is wrong or unknown, not its bars. [limits] are
        the bar lengths in force where each staff begins, where known;
        [lead] the tokens each reading was made to begin with, such as its
        task, which [seqs] leave out; [skip], for each staff, the numbers of
        bars to leave alone (meter_changes)."""
        out: list[list[Fit]] = [[] for _ in seqs]
        jobs = []
        for i, seq in enumerate(seqs):
            spans = bar_spans(seq, limits[i] if limits else None)
            over = [sp for b, sp in enumerate(spans) if sp.limit is not None and max(sp.voices, default=0) > sp.limit
                    and not (skip and b in skip[i])]
            if over and len(over) <= most * len(spans):
                jobs += [(i, sp) for sp in over]
        if not jobs:
            return out
        memory, pad = self.memory(images, widths)
        for i, sp in jobs:
            read = seqs[i][sp.start:sp.end + 1]
            options = triplet_options(read, sp.limit, tries)
            if not options:
                continue
            head = [BOS] + [INDEX[t] for t in (lead[i] if lead else [])] + [INDEX[t] for t in seqs[i][:sp.start]]
            scores = self._scores(memory[i:i + 1], pad[i:i + 1], head, [read] + options)
            k = max(range(1, len(scores)), key=lambda j: scores[j])
            out[i].append(Fit(sp.start, sp.end, read, options[k - 1], scores[0] - scores[k]))
        return out

    def _scores(self, mem, pad, head: list[int], options: list[list[str]], batch: int = 16) -> list[float]:
        """The log-probability the model gives each of [options] after [head]."""
        out = []
        for k in range(0, len(options), batch):
            chunk = options[k:k + batch]
            ids = [head + [INDEX[t] for t in o] for o in chunk]
            longest = max(map(len, ids))
            seq = torch.full((len(ids), longest), PAD, device=mem.device, dtype=torch.long)
            for j, x in enumerate(ids):
                seq[j, :len(x)] = torch.tensor(x, device=mem.device)
            logp = self.logits(mem.expand(len(ids), -1, -1), pad.expand(len(ids), -1), seq[:, :-1]).float().log_softmax(-1)
            for j, o in enumerate(chunk):
                at = torch.arange(len(head) - 1, len(head) - 1 + len(o), device=mem.device)
                out.append(float(logp[j, at, seq[j, at + 1]].sum()))
        return out

    @torch.no_grad()
    def chances(self, images: torch.Tensor, widths: torch.Tensor, starts: list[list[str]]) -> torch.Tensor:
        """For each staff, how likely each token is to come next after the
        tokens [starts] (all of one length): a distribution over the
        vocabulary per staff."""
        memory, pad = self.memory(images, widths)
        seq = torch.tensor([[BOS] + [INDEX[t] for t in st] for st in starts], device=images.device)
        return self.logits(memory, pad, seq)[:, -1].float().softmax(-1)


BAR_CAP = 64  # notes in one bar; no music we read has more


def _no_repeat() -> set[int]:
    from . import marks
    toks = marks.STEM + marks.NOTE + marks.DIRECTIONS + marks.STEM_V3 + marks.NOTE_V3 + list(marks.OPENERS_V3)
    # A dot may follow a dot (a double-dotted note), and a word's letters or
    # a count's digits repeat.
    return {INDEX[t] for t in toks if t in INDEX and t not in ('.', ')') and not t.startswith(('c:', 'mr:'))}


NO_REPEAT = _no_repeat()
MARKS_INDEX = torch.tensor(sorted(NO_REPEAT), dtype=torch.long)
MARK_RUN = 6  # marks between two notes; a printed score rarely stacks more
STARTS = [INDEX[t] for t in VOCAB if t.startswith(('N:', 'acc:')) or t in ('R', 'R:measure', 'R:multi', 'gr', 'S')]
STARTS_SET = set(STARTS)


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
            whole = t in ('R:measure', 'R:multi')  # fills its bar, whatever its length
            if whole and self.limit is not None:
                self.used = self.limit
            self.last = {'type': 'quarter', 'dots': 0, 'tuplet': None, 'chord': self.after_chord,
                         'grace': t == 'gr', 'measure': whole}
            self.after_chord = False
            self.notes += 1
            if whole:
                self.last = None
        elif self.last is not None:
            if t.startswith('d:'):
                self.last['type'] = t[2:]
            elif t == '.':
                self.last['dots'] += 1
            elif t.startswith('tup:'):
                self.last['tuplet'] = t[4:]


class BarSum(Budget):
    """A Budget that also keeps each finished voice's length, so a whole
    bar can be checked against its time signature."""

    def __init__(self, limit=None):
        super().__init__(limit)
        self.voices = []
        self.closed = None  # every voice's length at the last barline

    def lengths(self) -> list:
        return self.voices + [self.used + self._pending()]

    def add(self, t: str):
        if t in tokens.BARLINES:
            self.closed, self.voices = self.lengths(), []
        elif t == '&':
            self.voices.append(self.used + self._pending())
        super().add(t)

    def fits(self, limit) -> bool:
        """The bar just closed fills [limit] in every voice that holds music."""
        voices = [v for v in self.closed or [] if v > 0]
        return bool(voices) and all(v == limit for v in voices)

    def over(self, limit) -> bool:
        """The voice being read already runs past [limit] with its finished notes."""
        return self.used > limit


@dataclass
class Span:
    start: int  # the bar's first token in the reading
    end: int  # its barline
    limit: object  # the bar's length by its time signature, or None
    voices: list  # each voice's length as read
    state: BarSum  # the reading's state where the bar starts


@dataclass
class Fit:
    start: int
    end: int
    read: list  # the bar as read, barline included
    fit: list  # the likeliest reading of it that fills the bar
    cost: float  # how much less likely the model found the fit, in nats


def bar_spans(seq: list[str], limit=None) -> list[Span]:
    """Every bar of a reading, with the length each of its voices adds up to."""
    import copy
    st = BarSum(limit)
    spans, start, at_start = [], 0, copy.deepcopy(st)
    for k, t in enumerate(seq):
        st.add(t)
        if t in tokens.BARLINES:
            spans.append(Span(start, k, at_start.limit if at_start.limit is not None else st.limit,
                              [v for v in st.closed if v > 0], at_start))
            start, at_start = k + 1, copy.deepcopy(st)
    return spans


def apply_fits(seq: list[str], fits: list[Fit], margin: float) -> list[str]:
    """[seq] with each bar the re-reading fitted replaced, where the model
    found the fit at most [margin] nats less likely than its first reading."""
    out = list(seq)
    for f in sorted(fits, key=lambda f: -f.start):  # from the end, so earlier places hold
        if f.cost <= margin:
            out[f.start:f.end + 1] = f.fit
    return out


def meter_changes(seqs: list[list[str]], limits: list | None = None) -> list[set[int]]:
    """For each staff of one system, the bars every staff of the system
    fills to the same length, longer than its time signature: a change of
    meter that was not printed or not read, not triplets missing their
    "3", which overfill one staff while the others fill the bar. Bars are
    matched by number, so only staves read with as many bars count."""
    spans = [bar_spans(seq, limits[i] if limits else None) for i, seq in enumerate(seqs)]
    counts = collections.Counter(len(sp) for sp in spans)
    if not counts:
        return [set() for _ in seqs]
    n, _ = counts.most_common(1)[0]
    same = [i for i, sp in enumerate(spans) if len(sp) == n]
    changed = set()
    for b in range(n):
        lengths = {max(spans[i][b].voices) for i in same if spans[i][b].voices}
        limit = next((spans[i][b].limit for i in same if spans[i][b].limit is not None), None)
        if len(same) >= 2 and len(lengths) == 1 and limit is not None and lengths.pop() > limit:
            changed.add(b)
    return [changed if i in same else set() for i in range(len(seqs))]


def triplet_options(bar: list[str], limit, most: int = 48) -> list[list[str]]:
    """Readings of one bar, [bar] as read with its barline, in which some
    runs of three like notes or rests of an overfull voice are made
    triplets (tup:3:2 after each one's length, chord notes included), so
    that every voice fills [limit] exactly. Runs of three are taken within
    runs of like events that carry no tuplet, from the run's start first;
    at most [most] options, fewest triplets first."""
    from fractions import Fraction
    voices, cur = [], []
    for k, t in enumerate(bar[:-1]):
        if t == '&':
            voices.append(cur)
            cur = []
        else:
            cur.append(k)
    voices.append(cur)
    per_voice = []  # for each voice: the ways to mend it, as token positions to mark
    for idx in voices:
        events, chord, grace = [], False, False  # each event: [type, dots, tuplet, places to mark]
        for k in idx:
            t = bar[k]
            if t == '+':
                chord = True
            elif t == 'gr':
                grace = True
            elif t.startswith('N:') or t in ('R', 'S'):
                if chord and events:
                    events[-1][3].append(None)
                else:  # a spacer counts in the voice's length but is never made a triplet
                    events.append([None, 0, None, [None], 'grace' if grace else 'space' if t == 'S' else None])
                chord = grace = False
            elif t.startswith('d:') and events:
                if events[-1][0] is None:
                    events[-1][0] = t[2:]
                events[-1][3][-1] = k
            elif t == '.' and events and events[-1][3][-1] == k - 1:  # a dot right after a length
                if len(events[-1][3]) == 1:
                    events[-1][1] += 1
                events[-1][3][-1] = k
            elif t.startswith('tup:') and events:
                events[-1][2] = t[4:]
        events = [e for e in events if e[0] and e[4] != 'grace' and None not in e[3]]
        total = sum(tokens.length({'type': e[0], 'dots': e[1], 'tuplet': e[2]}) for e in events)
        if limit is None or total <= limit:
            per_voice.append([[]])
            continue
        triples = []
        for s in range(len(events) - 2):
            run = events[s:s + 3]
            if all(e[2] is None and e[4] is None and (e[0], e[1]) == (run[0][0], run[0][1]) for e in run):
                one = tokens.length({'type': run[0][0], 'dots': run[0][1], 'tuplet': None})
                triples.append((s, one, [p for e in run for p in e[3]]))  # three of [one] become two
        need, ways, steps = total - limit, [], [0]

        def walk(start: int, left, used: list, last: int):
            steps[0] += 1
            if len(ways) >= most or steps[0] > 20 * most:  # a long run of like notes has very many choices
                return
            if left == 0:
                ways.append([p for t in used for p in t[2]])
                return
            for j in range(start, len(triples)):
                t = triples[j]
                if t[0] <= last or t[1] > left:
                    continue
                walk(j + 1, left - t[1], used + [t], t[0] + 2)
        walk(0, need, [], -1)
        if not ways:
            return []
        per_voice.append(sorted(ways, key=len))
    options = [[]]
    for ways in per_voice:
        options = [a + w for a in options for w in ways][:most]
    out = []
    for marks in options:
        if not marks:
            continue
        marked = set(marks)
        new = []
        for k, t in enumerate(bar):
            new.append(t)
            if k in marked:
                new.append('tup:3:2')
        out.append(new)
    return out


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
            # A token the checkpoint never learned starts unlikely, so an
            # older model reads as it did until training teaches it: its
            # output row is zero and its bias low, so it never outscores a
            # learned token (a random row did, 2026-10-15).
            if name == 'out.weight':
                fitted[old.shape[0]:] = 0.0
            if name == 'out.bias':
                fitted[old.shape[0]:] = -8.0
            out[name] = fitted
    return out


def load(path, device=DEVICE) -> Reader:
    state = torch.load(path, map_location=device, weights_only=True)
    model = Reader(**state.get('config', {}))
    model.load_state_dict(grown(state['model'], model))
    known = state['model']['out.bias'].shape[0]
    if known < len(VOCAB):
        model.learned = known  # the tokens it was trained on (Reader.logits)
    return model.to(device).eval()
