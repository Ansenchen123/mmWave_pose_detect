import os
from typing import Optional

def getEnv_int(key: str, default: Optional[int] = None) -> Optional[int]:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(v, 0)
    except ValueError:
        return int(float(v))