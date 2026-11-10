"""Explicit parquet -> pickle fallback when optional local parquet engine is missing."""
from pathlib import Path
import pandas as pd

def write_table(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        frame.to_parquet(path,index=False)
        return path
    except ImportError:
        dest=path.with_suffix('.pkl')
        frame.to_pickle(dest)
        (path.parent/'PARQUET_ENGINE_NOT_INSTALLED.txt').write_text('pyarrow not installed; fallback artifacts are clearly named .pkl, not fake Parquet. Colab requirements install pyarrow.\n')
        return dest

def read_table(path):
    path=Path(path)
    if path.exists():return pd.read_parquet(path)
    if path.with_suffix('.pkl').exists():return pd.read_pickle(path.with_suffix('.pkl'))
    raise FileNotFoundError(str(path))

def table_exists(path):
    path=Path(path)
    return path.exists() or path.with_suffix('.pkl').exists()
