"""The injected robot: the plain robot asset with its face redrawn in glowing purple.
The asset's blue eyes and mouth are painted out with the screen colour, then solid purple irises (same size and place,
same two white highlights), a purple smile, and a soft purple bloom clipped to the screen are drawn at 4x and downsampled.
Nothing else changes, so it reads as the same robot as the plain one beside it."""
import io, base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

ASSET = "/Users/user/watching/figures/assets/robot.png"
SCREEN = (38, 54, 80)
EYES = [(164, 260), (284, 260)]            # eye centres in the 451 px asset
R_EYE = 47                                 # iris radius (the asset's eye incl. its ring is ~50)
SCREEN_BOX, SCREEN_RAD = (95, 159, 352, 380), 46


def injected_robot_image(path=ASSET):
    base = Image.open(path).convert("RGBA")
    a = np.asarray(base).astype(int); r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    face = (b > 150) & (r < 170) & (al > 40) & ~(np.abs(a[..., :3] - SCREEN).sum(-1) < 40)
    x0, y0, x1, y1 = SCREEN_BOX; inner = np.zeros_like(face); inner[y0 + 4:y1 - 4, x0 + 4:x1 - 4] = True
    face = Image.fromarray(((face & inner) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    out = base.copy(); out.paste(Image.new("RGBA", base.size, SCREEN + (255,)), mask=face)

    S = 4; W, H = base.size[0] * S, base.size[1] * S
    clip = Image.new("L", (W, H), 0); ImageDraw.Draw(clip).rounded_rectangle([v * S for v in SCREEN_BOX], SCREEN_RAD * S, fill=255)
    def layer(draw_fn, blur=0, alpha=1.0):
        m = Image.new("L", (W, H), 0); draw_fn(ImageDraw.Draw(m))
        if blur: m = m.filter(ImageFilter.GaussianBlur(blur * S))
        if alpha != 1: m = m.point(lambda v: int(v * alpha))
        return m
    def put(img, col, mask):
        img.alpha_composite(Image.composite(Image.new("RGBA", (W, H), col + (255,)), Image.new("RGBA", (W, H), (0, 0, 0, 0)), mask))
    disc = lambda rr: (lambda d: [d.ellipse([(cx - rr) * S, (cy - rr) * S, (cx + rr) * S, (cy + rr) * S], fill=255) for cx, cy in EYES])
    smile = lambda w: (lambda d: d.arc([186 * S, 296 * S, 262 * S, 346 * S], 25, 155, fill=255, width=int(w * S)))

    face_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    put(face_l, (190, 150, 255), ImageChops.multiply(layer(disc(R_EYE + 12), blur=18, alpha=1.0), clip))   # wide bloom
    put(face_l, (215, 185, 255), ImageChops.multiply(layer(disc(R_EYE + 2), blur=7, alpha=1.0), clip))         # tight bloom
    put(face_l, (170, 120, 255), ImageChops.multiply(layer(smile(12), blur=6, alpha=1.0), clip))             # mouth glow
    put(face_l, (190, 150, 255), layer(disc(R_EYE)))                                                        # iris
    # brighter core: a radial ramp toward the centre, so the iris glows rather than sits flat
    put(face_l, (220, 195, 255), layer(disc(R_EYE - 12), blur=10, alpha=0.8))
    put(face_l, (200, 160, 255), layer(smile(7)))                                                        # mouth
    hl = lambda d: [(d.ellipse([(cx - 25) * S, (cy - 31) * S, (cx - 5) * S, (cy - 11) * S], fill=255),
                     d.ellipse([(cx + 13) * S, (cy + 11) * S, (cx + 23) * S, (cy + 21) * S], fill=255)) for cx, cy in EYES]
    put(face_l, (255, 255, 255), layer(hl))                                                                # highlights, as the plain robot
    face_l = face_l.resize(base.size, Image.LANCZOS)
    out.alpha_composite(face_l)
    return out


def injected_robot(path=ASSET):
    buf = io.BytesIO(); injected_robot_image(path).save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    import sys
    injected_robot_image().save(sys.argv[1] if len(sys.argv) > 1 else "robot_injected.png")
