from __future__ import annotations

import numpy as np


def price_decimals(price: float) -> int:
    """Decimals that keep ~5 significant digits (FX 1.0842 -> 5, indices 5800 -> 2)."""
    if not np.isfinite(price) or price <= 0:
        return 2
    return int(max(2, 5 - np.floor(np.log10(price))))


def fmt_price(price: float) -> str:
    return f"{price:,.{price_decimals(price)}f}"
