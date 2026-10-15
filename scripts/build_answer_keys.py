"""Make the answer keys and their release zip (for maintainers).

    uv run python scripts/build_answer_keys.py

Each key in answers/keys.yaml is read from the file its `from` names on
this machine (under the scores, pdmx or cache setting), corrected as it
lists, and written to the cache's answers folder, which is what the
ladder reads. The folder is then zipped with a manifest of every key's
source, license and checksum, ready to attach to a GitHub release. The
script prints the zip's checksum for answers.SHA256."""

from __future__ import annotations

import hashlib
import json
import zipfile

from staffreader import answers, settings

ROOTS = {'scores': 'scores', 'pdmx': 'pdmx', 'cache': 'cache'}


def main():
    answers.FOLDER.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, k in answers.keys().items():
        (where, rel), = k['from'].items()
        source = settings.path(ROOTS[where]) / rel
        data = answers.read_musicxml(source)
        if k.get('corrections'):
            data = answers.correct(data, k['corrections'])
        for old in answers.FOLDER.glob(f'{key}.*'):
            old.unlink()
        out = answers.FOLDER / f'{key}.musicxml'
        out.write_bytes(data)
        manifest[key] = {'file': out.name, 'sha256': hashlib.sha256(data).hexdigest(),
                         **{f: k[f] for f in ('title', 'source', 'url', 'license')},
                         'corrected': len(k.get('corrections', []))}
        print(f'{key}: {len(data) // 1024} KB' + (f", {len(k['corrections'])} corrections" if k.get('corrections') else ''))
    (answers.FOLDER / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    zip_path = answers.FOLDER.parent / f'{answers.RELEASE}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(manifest):
            z.write(answers.FOLDER / manifest[name]['file'], manifest[name]['file'])
        z.write(answers.FOLDER / 'manifest.json', 'manifest.json')
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f'{zip_path}: {zip_path.stat().st_size // 1024} KB, sha256 {digest}')


if __name__ == '__main__':
    main()
