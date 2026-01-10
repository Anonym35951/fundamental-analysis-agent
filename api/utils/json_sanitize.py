import math
from datetime import date, datetime
from typing import Any

def make_json_safe(obj: Any):
    """
    Recursively convert values that are not JSON-compliant:
    - float('inf'), float('-inf') -> "inf" / "-inf"
    - float('nan') -> None
    - pandas/numpy types (if they show up) -> python native (best effort)
    - datetime/date -> isoformat string
    """
    # primitives
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # float handling
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj

    # datetime handling
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # dict
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    # fallback: try to cast numpy/pandas scalars etc.
    try:
        # e.g. numpy.float64 -> float
        if hasattr(obj, "item"):
            return make_json_safe(obj.item())
    except Exception:
        pass

    # last resort
    return str(obj)