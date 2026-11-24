"""Writing paper artifacts: LaTeX tables and figures under `paper_artifacts/<version>/`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rasping.config import ARTIFACTS_DIR


def artifact_path(version: str, kind: str, name: str) -> Path:
    """`paper_artifacts/<version>/<kind>/<name>`; kind is "tables" or "figures"."""
    p = ARTIFACTS_DIR / version / kind / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def latex_table(header: Sequence[str], rows: Sequence[Sequence], align: str | None = None,
                midrule_before_last: bool = False, caption: str | None = None,
                label: str | None = None) -> str:
    """A booktabs tabular (optionally wrapped in a table float)."""
    align = align or "l" + "r" * (len(header) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if midrule_before_last and i == len(rows) - 1:
            lines.append(r"\midrule")
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if caption is None and label is None:
        return body + "\n"
    out = [r"\begin{table}[t]", r"\centering"]
    if caption:
        out.append(rf"\caption{{{caption}}}")
    if label:
        out.append(rf"\label{{{label}}}")
    return "\n".join(out + [body, r"\end{table}"]) + "\n"


def write_table(version: str, name: str, tex: str) -> Path:
    p = artifact_path(version, "tables", name)
    p.write_text(tex)
    print(f"wrote {p.relative_to(ARTIFACTS_DIR.parent)}")
    return p


def save_figure(fig, version: str, name: str) -> Path:
    import matplotlib.pyplot as plt

    p = artifact_path(version, "figures", name)
    # No CreationDate, so regenerated PDFs are byte-identical to the committed ones.
    fig.savefig(p, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {p.relative_to(ARTIFACTS_DIR.parent)}")
    return p


def tt(s: str) -> str:
    """Typewriter font with escaped underscores (program names)."""
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}\\%"
