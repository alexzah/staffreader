"""Two minutes of the real training step under whatever power limit the
GPU has, to prove a cap works before a long run relies on it.

The real model trains on real batches, read once into memory first, so
the data workers' slow start does not count. Every five seconds it
samples the GPU's power draw, limit, temperature, clock and fan, and at
the end it prints the step rate beside them."""
import json, subprocess, sys, threading, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from staffreader import train, model as M
from staffreader.render import CACHE

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 120
LABEL = sys.argv[2] if len(sys.argv) > 2 else 'pilot'
items = train.samples([CACHE / 'data' / 'm3-a', CACHE / 'data' / 'real-a'], held_out=False)[::40][:6000]
loader = DataLoader(train.Staves(items, augment=True), batch_sampler=train.ByWidth(items, 16000), num_workers=4,
                    collate_fn=train.collate)
batches = []
for b in loader:
    batches.append(b)
    if len(batches) == 40:
        break
net = M.Reader().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
# Each new batch width makes the GPU's convolution library compile its
# kernels, seconds per shape; one untimed pass over the batches does it all
# first, so the timed minutes measure training alone.
warm = time.time()
for b in batches:
    images, widths, ids = (t.cuda() for t in b)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        net(images, widths, ids[:, :-1]).float().sum().backward()
torch.cuda.synchronize()
print(f'warmed up on {len(batches)} batch shapes in {time.time() - warm:.0f}s', flush=True)
samples, stop = [], threading.Event()


def watch():
    while not stop.is_set():
        out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,power.limit,temperature.gpu,clocks.sm,fan.speed',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        samples.append([float(x) for x in out.strip().split(',')])
        stop.wait(5)


threading.Thread(target=watch, daemon=True).start()
start, steps, staves = time.time(), 0, 0
while time.time() - start < SECONDS:
    images, widths, ids = (t.cuda() for t in batches[steps % len(batches)])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = net(images, widths, ids[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=M.PAD)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    steps += 1
    staves += images.shape[0]
torch.cuda.synchronize()
took = time.time() - start
stop.set()
late = samples[len(samples) // 3:] or samples  # after the card has warmed up
mean = lambda k: sum(s[k] for s in late) / len(late)
result = {'label': LABEL, 'seconds': round(took), 'staves_per_second': round(staves / took, 1),
          'power_limit_w': late[-1][1], 'power_draw_w': round(mean(0)), 'power_draw_max_w': max(s[0] for s in late),
          'temperature_c': round(mean(2)), 'temperature_max_c': max(s[2] for s in late),
          'clock_mhz': round(mean(3)), 'fan_percent': round(mean(4))}
print(json.dumps(result))
with open(CACHE / 'runs' / 'power-pilot.jsonl', 'a') as f:
    f.write(json.dumps(result) + '\n')
