"""IMSLP's catalog entry for each scan OpenScore's orchestral
transcriptions name: its name, size, pages, edition and copyright tag,
through imslp.py (api.php only, five seconds apart). Work pages are
tried by IMSLP's own naming, then by search."""
import json, re
from staffreader import imslp
from staffreader.progress import Progress
from staffreader.render import CACHE

HOME = CACHE / 'openscore' / 'orchestra'
OUT = HOME / 'orchestra-lookup.jsonl'
recs = [json.loads(l) for l in (HOME / 'orchestra-sources.jsonl').read_text().splitlines()]
bar = Progress('orchestra-lookup', len(recs), 'files', 'looking up the orchestral scans in IMSLP\'s catalog')
with open(OUT, 'w') as out:
    for k, r in enumerate(recs, start=1):
        w = r['want']
        comp = w['composer']
        opus = re.search(r'(Op\.\d+|WAB\.\d+)', r['movements'][0])
        opus = opus.group(1).replace('.', '.') if opus else ''
        work = re.sub(r'\s+', '_', w['work'].split(' (')[0]) if w['work'] else ''
        titles = [f'{work},_{opus}_({comp.replace(" ", "_")})', f'{work}_({comp.replace(" ", "_")})']
        if opus.startswith('WAB'):
            titles.insert(0, f'{work},_{opus.replace(".", "_")}_({comp.replace(" ", "_")})')
        bar.update(k - 1, f'{comp}: {w["work"]}')
        for fid in dict.fromkeys([w['imslp'], w['imslp'].zfill(5)]):
            f = imslp.find_file(fid, titles, f"{comp.split(',')[0]} {w['work']} {opus.replace('.', ' ')}")
            if f:
                break
        row = {'imslp': w['imslp'], 'composer': comp, 'work': w['work'], 'movements': r['movements']}
        if f:
            row.update({'page': f.work, 'name': f.name, 'megabytes': f.megabytes, 'pages': f.pages,
                        'publisher': f.publisher, 'copyright': f.copyright, 'description': f.description})
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
        out.flush()
bar.finish('done')
