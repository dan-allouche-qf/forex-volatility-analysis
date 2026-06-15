"""Number-provenance guard (reproducibility-as-a-result).

Recomputes the headline numbers from the package and fails if any of them no
longer appears in the committed prose (README.md / REPORT.md / report/report.tex).
The project's claim is "every number traceable to a computed value" -- this makes
that literally enforceable in CI.

Run: ``python scripts/check_numbers.py``  (also ``make report-guard``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from compute_results import compute_key  # same directory

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "REPORT.md", "report/report.tex"]


def _checks(key: dict) -> list[tuple[str, str]]:
    """(label, expected substring) pairs that must appear in at least one doc."""
    out: list[tuple[str, str]] = []
    for p in ("EURUSD", "GBPUSD", "USDJPY"):
        out.append((f"{p} vol peak", f"{key['vol_peak_%'][p]:.1f}"))
        out.append((f"{p} GARCH persistence", f"{key['garch_persistence'][p]:.3f}"))
    out.append(("PC1 explained", f"{key['pc1_explained_%']:.1f}"))
    out.append(("USDJPY ES p-value", f"{key['es99_garch_t_pvalue']['USDJPY']:.3f}"))
    return out


def main() -> int:
    key = compute_key()
    blob = "\n".join((ROOT / d).read_text(encoding="utf-8") for d in DOCS if (ROOT / d).exists())
    failures = [label for label, sub in _checks(key) if sub not in blob]
    if failures:
        print("NUMBER-PROVENANCE GUARD FAILED — committed prose drifted from computed values:")
        for f in failures:
            print(f"  - missing/stale: {f}")
        return 1
    print(f"Number-provenance guard PASSED — {len(_checks(key))} headline numbers traced to docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
