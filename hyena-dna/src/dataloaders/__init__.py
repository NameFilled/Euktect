try:
    from . import et, genomics
    from .base import SequenceDataset
except ImportError:
    pass  # optional heavy dataloaders (genomic_benchmarks etc.) not required for inference
