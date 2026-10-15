"""Fetch the Lieder songs' source files from every rule-following source
that holds the very file: here, Wikimedia Commons, the Internet
Archive's IMSLP snapshot. The waterfall's candidate steps (other
libraries' scans of the same edition) are left out of this pass: across
the quartets they confirmed nothing and once fetched the wrong work.
What no source holds is left for a person to download. Resumes where it stopped."""
import json
from staffreader import sources
from staffreader.progress import Progress
from staffreader.render import CACHE

HOME = CACHE / 'openscore'
OUT = HOME / 'lieder-sources.jsonl'
rows = [json.loads(l) for l in open(HOME / 'lieder-editions.jsonl')]
rows = [r for r in rows if r.get('work') and r.get('copyright') == 'Public Domain']
done = {json.loads(l)['want']['imslp'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
steps = [sources.here, sources.commons, sources.archive_snapshot]
bar = Progress('lieder-sources', len(rows), 'files', 'fetching Lieder source files from Commons and the Archive snapshot')
got = {}
with open(OUT, 'a') as out:
    for k, r in enumerate(rows, start=1):
        if r['imslp'] in done:
            continue
        bar.update(k - 1, f"{got.get('commons', 0)} from Commons, {got.get('archive-snapshot', 0)} from the Archive; now {r['paths'][0][:50]}")
        want = sources.Want(imslp=r['imslp'], name=r.get('name'), megabytes=r.get('megabytes'), pages=r.get('pages'),
                            work=r['work'], composer=r['paths'][0].split('/')[0], publisher=r.get('publisher'))
        found, notes = None, []
        for step in steps:
            try:
                found = step(want)
            except Exception as e:
                notes.append(f'{step.__name__}: {e!r}'[:200])
                continue
            if found:
                break
        found = found or sources.author(want)
        found.notes = notes + found.notes
        got[found.source] = got.get(found.source, 0) + 1
        rec = found.to_dict()
        rec['songs'] = r['songs']
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out.flush()
bar.finish(f"{got.get('here', 0)} here, {got.get('commons', 0)} from Commons, {got.get('archive-snapshot', 0)} from the Archive, {got.get('author', 0)} to download by hand")
