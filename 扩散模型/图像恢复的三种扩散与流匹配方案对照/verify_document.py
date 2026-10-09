"""Check formula block format, local links, anchors, images, and README index."""
from pathlib import Path
from urllib.parse import unquote
import re
import os
from PIL import Image

d = Path(__file__).resolve().parent
root = d.parent.parent
p = d / (d.name + '.md')
s = p.read_text(encoding='utf-8')
lines = s.splitlines()
blocks = [i for i,line in enumerate(lines) if line == '$$']
assert len(blocks) % 2 == 0
assert '@@REPORT:' not in s
assert not re.search(r'(?<!\$)\$(?!\$)', s)
assert all(b-a == 2 for a,b in zip(blocks[::2],blocks[1::2]))
anchors = set(re.findall(r'<a id="([^"]+)"></a>',s))
local_count = 0
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s):
    if link.startswith('http'):
        continue
    if link.startswith('#'):
        assert link[1:] in anchors, link
        continue
    target = (d / unquote(link)).resolve()
    if os.name == 'nt':
        target = Path('\\\\?\\' + str(target))
    assert target.is_file(), target
    assert target.stat().st_size > 0, target
    local_count += 1
images = list((d/'images').glob('*.png'))
assert len(images) == 4
for q in images:
    with Image.open(q) as im:
        im.verify()
sample=d/(d.name+'_sample.md')
assert sample.stat().st_size > 0
assert sample.read_text(encoding='utf-8').count('**') == 4
readme=(root/'README.md').read_text(encoding='utf-8')
assert d.name in readme
print(f'Verified {len(blocks)//2} single-line formula blocks, {local_count} local links, {len(anchors)} paper anchors, 4 PNG images, nonempty sample, README index.')
print(f'Report bytes: {p.stat().st_size}')
