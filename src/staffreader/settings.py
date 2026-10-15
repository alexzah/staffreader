"""Where staffreader keeps its work and finds its sources, and what it
tells the websites it reads from.

Each setting comes from the environment (STAFFREADER_<NAME>, such as
STAFFREADER_CACHE), else from settings.toml in the config folder
(~/.config/staffreader, or the file STAFFREADER_CONFIG names), else its
default:

    cache = "~/.cache/staffreader"     # renders, data sets, models and runs
    scans = "~/scores/imslp"           # printed scans, as IMSLP names them
    pdmx = "~/datasets/pdmx"           # the PDMX dataset, for rendering
    contact = "you@example.org"        # sent with every web request
    agent = "claude-opus-5-5"          # an AI agent running the tool names its model
    mscore = "mscore"                  # MuseScore 4, for rendering
    audiveris = "flatpak run org.audiveris.audiveris"
    wikimedia = "python /path/to/wikimedia.py"   # a polite Wikimedia client, for Commons scans
    device = "cuda"                    # or "cpu"

A path may start with ~, and may name another path setting in braces,
as "{cache}/scans" does."""

from __future__ import annotations

import os
import pathlib
import tomllib

DEFAULTS: dict[str, str | None] = {
    'cache': '~/.cache/staffreader',
    'scans': '{cache}/scans',
    'pdmx': '{cache}/pdmx',
    'scores': None,  # a folder of score bundles holding answer keys
    'contact': None,
    'agent': None,
    'mscore': 'mscore',
    'audiveris': None,
    'wikimedia': None,
    'device': None,
    'model': None,  # the reader's checkpoint, when not the default
    'text_model': None,
    'triplet_margin': '12',  # nats; a negative margin reads no overfull bar again (baselines/staffreader.py)
}
_FILE: dict | None = None


def config_file() -> pathlib.Path:
    named = os.environ.get('STAFFREADER_CONFIG')
    if named:
        return pathlib.Path(named).expanduser()
    base = pathlib.Path(os.environ.get('XDG_CONFIG_HOME') or pathlib.Path.home() / '.config')
    return base / 'staffreader' / 'settings.toml'


def get(name: str) -> str | None:
    """A setting as written, or None where it is not set."""
    global _FILE
    if name not in DEFAULTS:
        raise KeyError(f'no setting named {name}')
    if os.environ.get(f'STAFFREADER_{name.upper()}'):
        return os.environ[f'STAFFREADER_{name.upper()}']
    if _FILE is None:
        f = config_file()
        _FILE = tomllib.loads(f.read_text()) if f.exists() else {}
    value = _FILE.get(name, DEFAULTS[name])
    return None if value is None else str(value)


def path(name: str) -> pathlib.Path | None:
    """A path setting, with ~ and {other} expanded."""
    value = get(name)
    if value is None:
        return None
    for other in DEFAULTS:
        if '{' + other + '}' in value:
            value = value.replace('{' + other + '}', str(path(other)))
    return pathlib.Path(value).expanduser()
