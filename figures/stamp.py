"""Freshness stamps for the hero halves (hero_inject.py, hero_L28.py) and the pair built from them (side_by_side.py).
stamp() / verify(): each half writes OUT.stamp.json with the hash of its script, its helpers and the PNG it wrote; side_by_side
refuses a half whose PNG or script changed since (a stale half, or one rendered by an older script)."""
import hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
HELPERS = ("stamp.py", "pal.py", "pfmt.py")

def _sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def stamp(png, script):
    json.dump(dict(script=os.path.basename(script), script_sha=_sha(script), helpers={h: _sha(os.path.join(HERE, h)) for h in HELPERS},
                   png_sha=_sha(png)), open(png[:-4] + ".stamp.json", "w"), indent=1)

def verify(png, script):
    s = png[:-4] + ".stamp.json"
    assert os.path.exists(s), f"{png}: no stamp, not rendered by the current {script}"
    d = json.load(open(s))
    assert d["png_sha"] == _sha(png), f"{png} changed after its render"
    assert d["script_sha"] == _sha(os.path.join(HERE, script)), f"{png} is stale: {script} changed since it was rendered"
    for h, v in d["helpers"].items(): assert v == _sha(os.path.join(HERE, h)), f"{png} is stale: {h} changed since it was rendered"
    return d
