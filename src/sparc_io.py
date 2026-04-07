from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]


@dataclass(frozen=True)
class GalaxyCurve:
    name: str
    df: pd.DataFrame


def list_rotmod_files(rotmod_dir: Path) -> list[Path]:
    files = sorted(rotmod_dir.glob("*.dat"))
    return files


def load_rotmod_file(path: Path) -> GalaxyCurve | None:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=COLS).dropna()
    df = df[df["errV"] > 0].sort_values("Rad")
    if len(df) == 0:
        return None
    return GalaxyCurve(name=path.name, df=df)