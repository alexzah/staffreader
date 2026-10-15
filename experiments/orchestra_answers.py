"""Each OpenScore orchestral movement (Hauptstimme, pinned) as the
answer key pairing needs: MuseScore's MusicXML export of the
transcription, with the MuseScore file beside it for its line breaks."""
import pathlib, subprocess, tempfile, zipfile
from staffreader import render
from staffreader.render import CACHE

HOME = CACHE / 'openscore' / 'orchestra'
OUT = HOME / 'answers'
OUT.mkdir(exist_ok=True)
named = {line.split('|')[0] for line in (HOME / 'sources.txt').read_text().splitlines() if line.split('|')[1].strip()}
env = render.mscore_env()
done = 0
for mscz in sorted(HOME.glob('*.mscz')):
    if mscz.name not in named:
        continue
    stem = mscz.stem
    mxl, mscx = OUT / f'{stem}.mxl', OUT / f'{stem}.mscx'
    if not mscx.exists():
        z = zipfile.ZipFile(mscz)
        mscx.write_bytes(z.read(next(n for n in z.namelist() if n.endswith('.mscx'))))
    if not mxl.exists():
        subprocess.run([render.MSCORE, '-o', str(mxl), str(mscz)], env=env, capture_output=True, timeout=600)
    done += mxl.exists()
    print(stem, 'ok' if mxl.exists() else 'NO EXPORT', flush=True)
print(done, 'movements ready')
