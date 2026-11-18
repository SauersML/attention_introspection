"""Report bars shared by hero_question.py, hero_told.py and privileged_fig.py: one pair of bars per paragraph,
grey = no injection, purple = with injection, log y from 0.1% to 100%; the boosted paragraph's digit sits on an amber chip
and its pair is labelled "a% -> b%". With p1=None only the grey bars are drawn (the no-injection condition alone)."""
import math
from pal import INK, PURPLE
GREY = "#B8B8C0"; AMB = "#F2B53A"

def pair_bars(p0, p1, order, K, bw=44, gw=136, x0=116):
    H, base = 210, 250; o = []; top = 0
    Y = lambda v: base - H * (math.log10(max(v, 1e-3)) + 3) / 3
    for t, lab in ((1.0, "100%"), (0.01, "1%"), (0.001, "0.1%")):
        o.append(f'<text x="{x0 - 16}" y="{Y(t) + 13:.1f}" font-size="38" font-weight="700" text-anchor="end" fill="{INK}">{lab}</text>')
    o.append(f'<line x1="{x0 - 6}" x2="{x0 - 6}" y1="{Y(1):.1f}" y2="{base}" stroke="{INK}" stroke-width="3"/>')
    sets = ((p0, GREY),) if p1 is None else ((p0, GREY), (p1, PURPLE))
    for i, pos in enumerate(order):
        x = x0 + 8 + i * gw
        for j, (pr, c) in enumerate(sets):
            y = Y(pr[pos]); o.append(f'<rect x="{x + j * (bw + 4)}" y="{y:.1f}" width="{bw}" height="{base - y:.1f}" rx="5" fill="{c}"/>')
        cx = x + (bw + 2 if p1 is not None else bw / 2)
        if pos == K:
            o.append(f'<rect x="{cx - 30}" y="{base + 10}" width="60" height="54" rx="10" fill="{AMB}"/>')
            if p1 is None:
                top = min(top, Y(p0[pos]) - 54)
                o.append(f'<text x="{cx}" y="{Y(p0[pos]) - 14:.1f}" font-size="40" font-weight="800" text-anchor="middle" fill="{INK}">{100 * p0[pos]:.1f}%</text>')
            else:
                yt = Y(max(p0[pos], p1[pos])) - 14; top = min(top, yt - 86)
                o.append(f'<text x="{cx}" y="{yt - 46:.1f}" font-size="40" font-weight="800" text-anchor="middle" fill="{INK}">{100 * p0[pos]:.1f}% \u2192</text>')
                o.append(f'<text x="{cx}" y="{yt:.1f}" font-size="40" font-weight="800" text-anchor="middle" fill="{PURPLE}">{100 * p1[pos]:.1f}%</text>')
        o.append(f'<text x="{cx}" y="{base + 52}" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{i + 1}</text>')
    o.append(f'<line x1="{x0 - 6}" x2="{x0 + 4 * gw}" y1="{base}" y2="{base}" stroke="{INK}" stroke-width="3"/>')
    w = x0 + 4 * gw + 10; h = base + 70 - top
    return f'<svg width="{w}" height="{h:.0f}" viewBox="0 {top:.0f} {w} {h:.0f}">{"".join(o)}</svg>'
