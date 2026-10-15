"""The test ladder: each case names a score we hold, the bars it covers,
and the pictures of it a reader is given.

A track is either synthetic, the excerpt engraved by MuseScore (clean, or
made to look scanned), or a scan, one page of a printed edition, whose
answer is the case's bars."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

import yaml

from . import canonical, render
from . import settings

HERE = pathlib.Path(__file__).resolve().parents[2]
CASES = HERE / 'cases'
SCANS = settings.path('scans')


@dataclass
class Track:
    kind: str  # 'synthetic' or 'scan'
    profile: str = 'clean'  # synthetic: 'clean' or 'scan'
    pdf: str | None = None
    page: int | None = None
    layout: list | None = None  # a scan's systems as counted: [[staves, bars], ...]
    # Parts this edition prints on another part's staff, such as a bass on
    # the cello's, which the answer key writes apart: no reader can tell
    # them from the page, so they are left out of this track's answer.
    shared: list | None = None

    @property
    def name(self) -> str:
        return f'synthetic-{self.profile}' if self.kind == 'synthetic' else f'scan-p{self.page}'


@dataclass
class Case:
    id: str
    level: int
    title: str
    proves: str
    key: str  # its answer key (answers.py)
    first: int
    last: int
    tracks: list[Track]
    masks: frozenset[str] = field(default_factory=frozenset)

    @property
    def truth_file(self) -> pathlib.Path:
        """The whole score the case's bars are cut from: its answer key."""
        from . import answers
        return answers.path(self.key)

    def truth(self, track: Track | None = None) -> canonical.Score:
        score = canonical.read(self.truth_file).excerpt(self.first, self.last)
        if track is not None and track.shared:
            score = canonical.Score([p for p in score.parts if p.name not in track.shared])
        return score

    def work(self) -> pathlib.Path:
        return render.CACHE / 'cases' / self.id

    def pages(self, track: Track) -> list[tuple[pathlib.Path, int | None]]:
        """The pictures for [track], made on first use, as (file, page):
        PNG pages of a synthetic track, with no page number, or the scan's
        PDF with the page to read."""
        out = self.work() / track.name
        if track.kind == 'scan':
            return [(SCANS / track.pdf, track.page)]
        done = sorted(out.glob('page-*.png'))
        if done:
            return [(p, None) for p in done]
        x = render.excerpt(self.truth_file, self.first, self.last, self.work() / 'excerpt.musicxml')
        pdf = render.engrave(x, self.work() / 'excerpt.pdf')
        render.engrave(x, self.work() / 'excerpt.svg')
        clean = render.rasterize(pdf, self.work() / 'synthetic-clean')
        if track.profile == 'clean':
            return [(p, None) for p in clean]
        return [(render.scanlike(p, out / p.name, seed=i), None) for i, p in enumerate(clean)]


    def layout_truth(self, track: Track, k: int):
        """The right layout of page [k] (from 0) of [track]: exact from
        MuseScore's SVG for a synthetic page, or as counted on a scan."""
        from . import layout_truth
        if track.kind == 'synthetic':
            self.pages(track)
            svg = self.work() / f'excerpt-{k + 1}.svg'
            if not svg.exists():
                render.engrave(self.work() / 'excerpt.musicxml', self.work() / 'excerpt.svg')
            return layout_truth.from_svg(svg)
        # A scan track is one page; its layout lists each system's [staves, bars].
        return layout_truth.Counted([tuple(s) for s in track.layout]) if track.layout else None


def load(path: pathlib.Path) -> Case:
    d = yaml.safe_load(path.read_text())
    first, last = d['truth']['bars']
    return Case(id=d['id'], level=int(d['level']), title=d['title'], proves=d.get('proves', ''),
                key=d['truth']['key'], first=int(first), last=int(last),
                tracks=[Track(**t) for t in d['tracks']], masks=frozenset(d.get('masks', [])))


def all_cases(max_level: int | None = None, only: list[str] | None = None) -> list[Case]:
    cases = [load(p) for p in sorted(CASES.glob('*.yaml'))]
    if max_level is not None:
        cases = [c for c in cases if c.level <= max_level]
    if only:
        cases = [c for c in cases if c.id in only]
    return sorted(cases, key=lambda c: (c.level, c.id))
