"""Check that nothing private is about to be published: no secret, no
personal address, no path on someone's machine.

    uv run python scripts/check_clean.py                # the files git tracks
    uv run python scripts/check_clean.py --history      # every commit: files, messages, authors

It looks for:

- secrets by their shape: GitHub, GitLab, AWS, OpenAI, Anthropic,
  Hugging Face and Slack tokens, and private keys;
- email addresses, other than example addresses and GitHub's and
  Anthropic's no-reply addresses;
- absolute paths under a home folder;
- every line of a private deny list (default: private-strings.txt in the
  cache folder, or --deny), such as a real address or a contact alias.
  The list stays outside the repository, so it is never published.

It exits 1 and prints each finding when it finds anything."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

from staffreader import settings

SECRETS = {
    'GitHub token': r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}',
    'GitLab token': r'\bglpat-[A-Za-z0-9_-]{20,}',
    'AWS key': r'\bAKIA[0-9A-Z]{16}\b',
    'OpenAI or Anthropic key': r'\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}',
    'Hugging Face token': r'\bhf_[A-Za-z0-9]{30,}',
    'Slack token': r'\bxox[abprs]-[A-Za-z0-9-]{10,}',
    'private key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
}
EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
ALLOWED_EMAIL = re.compile(r'@(?:example\.(?:org|com|net)|users\.noreply\.github\.com)$|^noreply@anthropic\.com$')
HOME = re.compile(r'(?:/home|/Users)/[A-Za-z0-9._-]+|[A-Z]:\\\\Users\\\\[A-Za-z0-9._-]+')
CACHE = settings.path('cache')


def findings(text: str, where: str, deny: list[str]) -> list[str]:
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        at = f'{where}:{n}'
        for kind, pattern in SECRETS.items():
            if re.search(pattern, line):
                out.append(f'{at}: {kind}')
        for email in EMAIL.findall(line):
            if not ALLOWED_EMAIL.search(email):
                out.append(f'{at}: email address {email}')
        if HOME.search(line):
            out.append(f'{at}: home path {HOME.search(line).group(0)}')
        for word in deny:
            if word.lower() in line.lower():
                out.append(f'{at}: private string from the deny list')
    return out


def _git(*args: str, root: pathlib.Path) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True,
                          errors='replace').stdout


def _text(data: bytes) -> str | None:
    return None if b'\0' in data[:8000] else data.decode('utf-8', 'replace')


def check_tree(root: pathlib.Path, deny: list[str]) -> list[str]:
    out = []
    for name in _git('ls-files', '-z', root=root).split('\0'):
        path = root / name
        if name and path.is_file() and (text := _text(path.read_bytes())) is not None:
            out += findings(text, name, deny)
    return out


def check_history(root: pathlib.Path, deny: list[str]) -> list[str]:
    out = []
    for commit in _git('rev-list', '--all', root=root).split():
        meta = _git('show', '-s', '--format=%an <%ae>%n%cn <%ce>%n%B', commit, root=root)
        out += findings(meta, f'commit {commit[:10]}', deny)
        for line in _git('ls-tree', '-r', '-z', commit, root=root).split('\0'):
            if not line:
                continue
            info, name = line.split('\t', 1)
            blob = info.split()[2]
            data = subprocess.run(['git', '-C', str(root), 'cat-file', 'blob', blob], check=True,
                                  capture_output=True).stdout
            if (text := _text(data)) is not None:
                out += findings(text, f'{commit[:10]}:{name}', deny)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('root', nargs='?', default='.', type=pathlib.Path)
    ap.add_argument('--history', action='store_true', help='check every commit, not only the files tracked now')
    ap.add_argument('--deny', type=pathlib.Path, default=CACHE / 'private-strings.txt')
    args = ap.parse_args()
    # The files tracked under [root]; or, with --history, every commit of its repository.
    root = pathlib.Path(_git('rev-parse', '--show-toplevel', root=args.root).strip()) if args.history else args.root
    deny = [l.strip() for l in args.deny.read_text().splitlines() if l.strip() and not l.startswith('#')] \
        if args.deny.exists() else []
    if not args.deny.exists():
        print(f'note: no deny list at {args.deny}; checking shapes only', file=sys.stderr)
    found = (check_history if args.history else check_tree)(root, deny)
    for f in found:
        print(f)
    print(f"{len(found)} findings in {'every commit' if args.history else 'the tracked files'} of {root}",
          file=sys.stderr)
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
