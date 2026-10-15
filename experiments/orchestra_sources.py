"""Find the scans OpenScore's orchestral transcriptions (the Hauptstimme
corpus) were made from: each file looked up in IMSLP's catalog through
imslp.py (api.php only, five seconds apart), then fetched from every
rule-following source that holds the very file (here, Wikimedia Commons,
the Internet Archive's IMSLP snapshot). What none holds is left for the
author. Resumes where it stopped."""
import json, re, zipfile
from staffreader import imslp, sources
from staffreader.progress import Progress
from staffreader.render import CACHE

HOME = CACHE / 'openscore' / 'orchestra'
OUT = HOME / 'orchestra-sources.jsonl'
files: dict[str, dict] = {}
for line in (HOME / 'sources.txt').read_text().splitlines():
    name, ids = line.split('|')
    for fid in ids.split():
        mscx = next(n for n in zipfile.ZipFile(HOME / name).namelist() if n.endswith('.mscx'))
        text = zipfile.ZipFile(HOME / name).read(mscx).decode('utf-8', 'replace')
        tag = lambda t: (re.search(f'<metaTag name="{t}">([^<]*)', text) or [None, ''])[1]
        f = files.setdefault(fid.lstrip('0') or fid, {'movements': [], 'work': tag('workTitle'), 'composer': tag('composer')})
        f['movements'].append(name)
done = {json.loads(l)['want']['imslp'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
bar = Progress('orchestra-sources', len(files), 'files', 'finding the scans OpenScore orchestral transcriptions name')
with open(OUT, 'a') as out:
    for k, (fid, f) in enumerate(sorted(files.items()), start=1):
        if fid in done:
            continue
        bar.update(k - 1, f"{f['composer']}: {f['work']}")
        name = f['composer'] or ''
        surname = name.split(',')[0].strip() if ',' in name else (name.split()[-1] if name else '')
        query = f"{surname} {f['work']}".strip()
        got = imslp.find_file(fid, [], query)
        want = sources.Want(imslp=fid, name=got.name if got else None, megabytes=got.megabytes if got else None,
                            pages=got.pages if got else None, work=got.work if got else query, composer=f['composer'],
                            publisher=got.publisher if got else None)
        found, notes = None, []
        for step in (sources.here, sources.commons, sources.archive_snapshot):
            try:
                found = step(want)
            except Exception as e:
                notes.append(f'{step.__name__}: {e!r}'[:200])
                continue
            if found:
                break
        found = found or sources.author(want)
        found.notes = notes + found.notes
        rec = found.to_dict()
        rec['movements'] = f['movements']
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out.flush()
recs = [json.loads(l) for l in OUT.read_text().splitlines()]
bar.finish(', '.join(f'{n} {s}' for s, n in sorted(__import__('collections').Counter(r['source'] for r in recs).items())))
