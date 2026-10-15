"""Which IMSLP file, edition and pages each OpenScore Lieder song was
transcribed from, looked up in IMSLP's catalog through imslp.py (api.php
only, five seconds apart, halting on refusal). One lookup per distinct
file, since a song cycle often shares one. Resumes where it stopped."""
import json, pathlib, re
import yaml
from staffreader import imslp
from staffreader.progress import Progress
from staffreader.render import CACHE

HOME = CACHE / 'openscore'
OUT = HOME / 'lieder-editions.jsonl'
songs = yaml.safe_load(open(HOME / 'lieder-scores-pinned.yaml'))
by_file: dict[str, list] = {}
for sid, s in songs.items():
    fid = str(s.get('imslp', '')).strip().lstrip('#')
    if fid:
        by_file.setdefault(fid, []).append({'id': str(sid), **s})
done = {json.loads(l)['imslp'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()


def titles(path: str) -> tuple[list[str], str]:
    comp, group, song = (path.split('/') + ['', ''])[:3]
    work = song if group == '_' else group
    plain = re.sub(r'[„“”"«»]', '', work)
    out = [f'{work}_({comp})', f'{plain}_({comp})']
    catalog = re.search(r'(Op\.?_?\d+[a-z]?|D\.?_?\d+|WoO\.?_?\d+|S\.?_?\d+|MWV_[A-Z]+_?\d+)', work)
    query = f"{comp.split(',')[0]} {catalog.group(0).replace('_', ' ') if catalog else plain.replace('_', ' ')[:50]}"
    return list(dict.fromkeys(out)), query


bar = Progress('lieder-lookup', len(by_file), 'files', 'finding each Lieder song\'s IMSLP file and edition')
found = sum(1 for l in OUT.read_text().splitlines() if json.loads(l).get('work')) if OUT.exists() else 0
with open(OUT, 'a') as out:
    for k, (fid, rows) in enumerate(by_file.items(), start=1):
        if fid in done:
            continue
        bar.update(k - 1, f'{found} found so far; now {rows[0]["path"][:60]}')
        tries, query = titles(rows[0]['path'])
        try:
            f = imslp.find_file(fid, tries, query)
        except imslp.Refused as e:
            bar.update(k - 1, f'halted: IMSLP refused ({e})')
            raise
        rec = {'imslp': fid, 'songs': [r['id'] for r in rows], 'paths': [r['path'] for r in rows]}
        if f:
            rec.update({k2: v for k2, v in f.to_dict().items() if k2 != 'id'})
            found += 1
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out.flush()
bar.finish(f'{found} of {len(by_file)} files found')
