"""Control hero (hero_L28.png) and introspection hero (hero_inject.png) side by side, unchanged: each scaled to the same height, white gap."""
import os
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__))
a, b = (Image.open(os.path.join(H, f)).convert("RGB") for f in ("hero_L28.png", "hero_inject.png"))
h = max(a.height, b.height)
a, b = (im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in (a, b))
gap = 80
out = Image.new("RGB", (a.width + gap + b.width, h), "white"); out.paste(a, (0, 0)); out.paste(b, (a.width + gap, 0))
out.save(os.path.join(H, "hero_pair.png")); out.save(os.path.join(H, "hero_pair.jpg"), quality=90)
print(out.size)
