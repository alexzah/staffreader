"""Confirm paired real staves by reading them: a trained model reads
every staff a pairing produced, and a system is kept only where its
staves' readings agree with the transcription they were paired with
(mean note F1 at least KEEP). A song paired with the wrong pages of its
volume cannot agree; a right pairing agrees about as well as the model
reads any real page. Writes DATA/<out>/index.jsonl, pointing at the
original pictures, and a report."""
import collections, json, sys
import cv2, torch
from staffreader import metrics, measure, model as M
from staffreader.progress import Progress
from staffreader.render import CACHE

KEEP = 0.6
MODEL = CACHE / 'models/m4-real/last.pt'
sets, out_name = sys.argv[1].split(','), sys.argv[2]
out = CACHE / 'data' / out_name
out.mkdir(parents=True, exist_ok=True)
rows = []
for s in sets:
    for line in open(CACHE / 'data' / s / 'index.jsonl'):
        r = json.loads(line)
        r['image'] = f'../{s}/' + r['image']
        r['set'] = s
        rows.append(r)
net = M.load(MODEL)
bar = Progress(f'check {out_name}', len(rows), 'staves', f'reading {", ".join(sets)} with {MODEL.parent.name} to confirm each pairing')
f1 = {}
order = sorted(range(len(rows)), key=lambda i: rows[i].get('tokens') and len(rows[i]['tokens']))
for k in range(0, len(order), 16):
    idx = order[k:k + 16]
    views = [M.window(cv2.imread(str(out / rows[i]['image']), cv2.IMREAD_GRAYSCALE)) for i in idx]
    x, w = M.batch_images(views, 'cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        reads = net.read(x, w)
    for i, got in zip(idx, reads):
        try:
            f1[i] = metrics.compare(measure._score(rows[i]['tokens']), measure._score(got)).notes['core'].f1
        except Exception:
            f1[i] = 0.0
    if k % 320 == 0:
        bar.update(min(len(rows), k + 16), f'{len(f1)} read')
systems = collections.defaultdict(list)
for i, r in enumerate(rows):
    systems[(r['set'], r['mxl'], r['page'], r['image'].rsplit('-s', 1)[1].split('-')[0])].append(i)
kept = dropped = 0
with open(out / 'index.jsonl', 'w') as index:
    for key, members in systems.items():
        mean = sum(f1[i] for i in members) / len(members)
        if mean >= KEEP:
            kept += 1
            for i in members:
                index.write(json.dumps({**rows[i], 'check_f1': round(f1[i], 3)}) + '\n')
        else:
            dropped += 1
hist = collections.Counter(round(min(0.99, sum(f1[i] for i in m) / len(m)), 1) for m in systems.values())
(out / 'check.json').write_text(json.dumps({'kept_systems': kept, 'dropped_systems': dropped,
                                            'system_f1_histogram': dict(sorted(hist.items()))}, indent=1))
bar.finish(f'{kept} systems kept, {dropped} dropped')
