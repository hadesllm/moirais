# moirais.fn — function file (hadesllm/moirais)
"""He who would learn to fly one day must first learn to stand and walk. — Friedrich Nietzsche"""

from moirais.sampling import compute_design_weights as _fn

dsgwt = _fn
compute_design_weights = _fn


def cheatsheet() -> str:
    return "dsgwt() -> Inverse-probability design weights for stratified samples. '"
