"""Control hero and introspection hero side by side, white gap. Built from the narrower, taller versions rendered for the pair
(hero_L28.py --W=1800 --VH=17.5 -> hero_L28_tall.png; hero_inject.py --W=1800 --VH=1295 -> hero_inject_tall.png), which come out the
same height; any small difference is evened out by scaling both to the taller one."""
import os
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__))
a, b = (Image.open(os.path.join(H, f)).convert("RGB") for f in ("hero_L28_tall.png", "hero_inject_tall.png"))
h = max(a.height, b.height)
a, b = (im if im.height == h else im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in (a, b))
gap = 80
out = Image.new("RGB", (a.width + gap + b.width, h), "white"); out.paste(a, (0, 0)); out.paste(b, (a.width + gap, 0))
out.save(os.path.join(H, "hero_pair.png")); out.save(os.path.join(H, "hero_pair.jpg"), quality=90)
print(out.size, "width/height", round(out.width / out.height, 2))
