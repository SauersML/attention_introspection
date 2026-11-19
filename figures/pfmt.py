"""p-value text for every figure: small p written with a power of ten (p = 2×10⁻⁴, p < 10⁻⁵), larger p as a short decimal."""
import math
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def pow10(e): return "10" + str(e).translate(SUP)
def pstr(p, floor=None):
    """p: the p-value. floor: the smallest p the test can give (e.g. 1/(flips+1)); at the floor, 'p < 10⁻ᵏ' with 10⁻ᵏ the power of ten just above it."""
    if floor is not None and p <= floor * 1.5:
        return "p < " + pow10(math.floor(math.log10(floor * 1.5)) + 1)
    if p >= 0.001:
        return f"p = {p:.1g}" if p < 0.1 else f"p = {p:.2g}"
    e = math.floor(math.log10(p)); m = round(p / 10 ** e)
    if m == 10: m, e = 1, e + 1
    return "p = " + (f"{m}×" if m != 1 else "") + pow10(e)
def pbound(threshold):
    """'p < 10⁻ᵏ' for a script that already says p < threshold (threshold a power of ten)."""
    return "p < " + pow10(round(math.log10(threshold)))
