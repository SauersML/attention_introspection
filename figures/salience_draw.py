"""Method drawings for salience.py (Qwen3-32B, layers 21-53).
stream(kind, w): token row [P1]..[P4] (paragraph 2 amber) + question and answer tokens, under the band of layers 21-53.
  'att'  purple arrows from the question and answer tokens onto paragraph 2 (attention x20)
  'add'  green: paragraph 2's meaning vector added at every question and answer token
  'rand' slate: a random direction added at the same tokens, same size
recipe(w): how the green vector is made: mean activation over paragraph 2's tokens minus the mean over all four paragraphs' tokens = the
vector (green arrow), scaled to the size of the attention boost's change."""
from pal import INK, PURPLE
import injb_draw as dr
GREEN = "#1A9E77"; SLATE = "#556070"; F = 80


def _arrow(x0, y, x1, col, sw=14, mid="v"):
    return (f'<defs><marker id="{mid}" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto"><path d="M0,0 L4,2 L0,4 Z" fill="{col}"/></marker></defs>'
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{col}" stroke-width="{sw}" marker-end="url(#{mid})"/>')


def stream(kind, w=1440):
    o = []; cw, cg = 50, 8
    groups = [("P1", 3), ("P2", 3), ("P3", 3), ("P4", 3)]
    x = 40; cells = {}
    for g, n in groups:
        cells[g] = [x + j * (cw + cg) for j in range(n)]; x += n * (cw + cg) + 34
    nq = 6; cells["Q"] = [w - 40 - (nq - j) * (cw + cg) + cg for j in range(nq)]
    by0, by1, ty = 120, 360, 380
    col = dict(att=PURPLE, add=GREEN, rand=SLATE)[kind]
    title = {"att": "attention to [P2] boosted \u00d720", "add": "[P2]\u2019s meaning added to activations",
             "rand": "random direction added, same size"}[kind]
    o.append(f'<text x="{w / 2}" y="{F}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{col}">{title}</text>')
    o.append(f'<rect x="20" y="{by0}" width="{w - 40}" height="{by1 - by0}" rx="26" fill="#F4F4F6" stroke="{INK}" stroke-width="4"/>')
    o.append(f'<text x="46" y="{by0 + 76}" font-size="{F}" font-weight="800" fill="{INK}">layers 21\u201353</text>')
    for g, xs in cells.items():
        for cx in xs:
            fill = dr.AMB if g == "P2" else ("#1F8BFF" if g == "Q" else "#9DC9FF")
            o.append(f'<rect x="{cx}" y="{ty}" width="{cw}" height="60" rx="8" fill="{fill}"/>')
            o.append(f'<line x1="{cx + cw / 2}" y1="{ty}" x2="{cx + cw / 2}" y2="{by0 + 100 if g != "Q" else by0 + 4}" stroke="{INK}" stroke-opacity=".18" stroke-width="3"/>')
        if g == "Q":
            for i, l in enumerate(["question", "+ answer"]):
                o.append(f'<text x="{(xs[0] + xs[-1] + cw) / 2}" y="{ty + 138 + 88 * i}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{INK}">{l}</text>')
        else:
            o.append(f'<text x="{(xs[0] + xs[-1] + cw) / 2}" y="{ty + 138}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{INK}">[{g}]</text>')
    p2 = (cells["P2"][0] + cells["P2"][-1] + cw) / 2; qs = [c + cw / 2 for c in cells["Q"]]
    if kind == "att":
        o.append(f'<defs><marker id="ha" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="{PURPLE}"/></marker></defs>')
        for j, qx in enumerate(qs[::2]):
            y0 = by1 - 40 - 22 * j
            o.append(f'<path d="M{qx},{y0} C{qx},{by0 + 40} {p2 + 60},{by0 + 40} {p2 + 8 * j - 12},{by1 - 30}" fill="none" stroke="{PURPLE}" stroke-width="12" stroke-opacity=".8" marker-end="url(#ha)"/>')
    else:
        yp = by0 + 175
        for i, qx in enumerate(qs):
            o.append(f'<circle cx="{qx}" cy="{yp}" r="22" fill="#fff" stroke="{col}" stroke-width="6"/><path d="M{qx - 12},{yp} H{qx + 12} M{qx},{yp - 12} V{yp + 12}" stroke="{col}" stroke-width="6"/>')
        # the added vector, drawn as the same arrow as in the recipe
        ax0 = qs[0] - 330
        o.append(_arrow(ax0, yp, qs[0] - 50, col, mid="m" + kind))
    h = ty + 250
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(o)}</svg>'


def recipe(w=2948):
    """one row, green: [P2] tokens averaged minus all four paragraphs' tokens averaged = the vector, scaled to the attention boost's change"""
    o = []; cw, cg = 40, 7; y = 20; ch = 74
    x = 130
    for j in range(3): o.append(f'<rect x="{x + j * (cw + cg)}" y="{y}" width="{cw}" height="{ch}" rx="7" fill="{dr.AMB}"/>')
    g1 = (x, x + 3 * (cw + cg) - cg)
    x = g1[1] + 120
    o.append(f'<text x="{x}" y="{y + 66}" font-size="120" font-weight="800" text-anchor="middle" fill="{INK}">\u2212</text>')
    x += 120; g2s = x
    for p in range(4):
        for j in range(3):
            o.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" fill="{dr.AMB if p == 1 else "#9DC9FF"}"/>'); x += cw + cg
        x += 16
    g2 = (g2s, x - 16 - cg)
    x += 70
    o.append(f'<text x="{x}" y="{y + 68}" font-size="110" font-weight="800" text-anchor="middle" fill="{INK}">=</text>')
    x += 70
    o.append(_arrow(x, y + ch / 2, x + 300, GREEN, sw=18, mid="rv")); xa = x + 150
    for (a, b), t in ((g1, ["[P2]", "averaged"]), (g2, ["all four paragraphs", "averaged"]), ((xa, xa), ["[P2]\u2019s", "meaning"])):
        for i, l in enumerate(t):
            o.append(f'<text x="{(a + b) / 2}" y="{y + ch + 88 + 90 * i}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{GREEN if a == b else INK}">{l}</text>')
    xt = x + 400
    for i, l in enumerate(["scaled to the", "same size as the", "attention boost\u2019s change"]):
        o.append(f'<text x="{xt}" y="{y + 64 + 90 * i}" font-size="{F}" font-weight="800" fill="{GREEN}">{l}</text>')
    h = y + ch + 200
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(o)}</svg>'
