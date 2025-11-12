# datasets/__init__.py
# Only expose classes if their deps are available; avoid hard-failing on unused datasets.

try:
    from .endoslam_dataset import EndoSLAMDataset
except Exception:
    EndoSLAMDataset = None

try:
    from .scared_dataset import SCAREDRAWDataset
except Exception:
    SCAREDRAWDataset = None

__all__ = ["EndoSLAMDataset", "SCAREDRAWDataset"]
