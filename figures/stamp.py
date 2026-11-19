"""Guards for the hero halves (hero_inject.py, hero_L28.py) and the pair built from them (side_by_side.py).

check(): fails the render if the y axis lost its tick labels or title: at least 3 visible '%' labels and a title, each inside the
figure, none on top of another or under one of the HTML boxes later laid over the violin image (rects given in figure pixels, origin top left).
stamp() / verify(): each half writes OUT.stamp.json with the hash of its script, its helpers and the PNG it wrote; side_by_side
refuses a half whose PNG or script changed since (a stale half, or one rendered by an older script)."""
import hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
HELPERS = ("yaxis.py", "pal.py", "pfmt.py")

def _sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def check(fig, labels, avoid=(), name="", title=None):
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); W, H = fig.canvas.get_width_height()
    good = [t for t in labels if t.get_visible() and t.get_text().strip().endswith("%") and t.get_alpha() in (None, 1)]
    assert len(good) >= 3, f"{name}: y axis has {len(good)} tick labels, need at least 3"
    bb = []
    for t in good:
        b = t.get_window_extent(rd); x0, x1, y0, y1 = b.x0, b.x1, H - b.y1, H - b.y0   # figure px, origin top left
        assert x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H, f"{name}: y label {t.get_text()!r} outside the figure"
        for (l, tp, r, bt) in avoid:
            assert x1 <= l or x0 >= r or y1 <= tp or y0 >= bt, f"{name}: y label {t.get_text()!r} hidden under a box"
        for (a0, a1, b0, b1, s) in bb:
            assert x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1, f"{name}: y labels {s!r} and {t.get_text()!r} overlap"
        bb.append((x0, x1, y0, y1, t.get_text()))
    if title is not None:   # the axis title must be there and clear of the tick labels
        assert title.get_visible() and title.get_text().strip(), f"{name}: y axis title missing"
        b = title.get_window_extent(rd); x0, x1, y0, y1 = b.x0, b.x1, H - b.y1, H - b.y0
        assert x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H, f"{name}: y axis title outside the figure"
        for (a0, a1, b0, b1, s) in bb:
            assert x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1, f"{name}: y axis title overlaps {s!r}"
    print(name, "y axis:", " ".join(s for *_, s in bb))
    return [s for *_, s in bb]

def stamp(png, script, ylabels):
    json.dump(dict(script=os.path.basename(script), script_sha=_sha(script), helpers={h: _sha(os.path.join(HERE, h)) for h in HELPERS},
                   png_sha=_sha(png), ylabels=ylabels), open(png[:-4] + ".stamp.json", "w"), indent=1)

def verify(png, script):
    s = png[:-4] + ".stamp.json"
    assert os.path.exists(s), f"{png}: no stamp, not rendered by the current {script}"
    d = json.load(open(s))
    assert d["png_sha"] == _sha(png), f"{png} changed after its render"
    assert d["script_sha"] == _sha(os.path.join(HERE, script)), f"{png} is stale: {script} changed since it was rendered"
    for h, v in d["helpers"].items(): assert v == _sha(os.path.join(HERE, h)), f"{png} is stale: {h} changed since it was rendered"
    assert len(d["ylabels"]) >= 3, f"{png}: rendered without y tick labels"
    return d
