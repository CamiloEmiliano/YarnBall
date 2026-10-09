"""
SFT Dataset Generation & Manifold Synthesis Pipeline for YarnBall.

Modules:
- builder: End-to-end multi-task SFT dataset generation orchestrator integrated with Cold-Start & Active Learning curation.
- sampler: Manifold-targeted stratified sampler with rare relation floors and active learning steering.
- annotator: 5-Axis financial taxonomy annotation engine.
- exporter: Multi-task formatting, train/val/test 80/10/10 split partitioning with zero leakage.
- hf_sync: Hugging Face Hub synchronization utility.
"""

from .sampler import (
    DOMINANT_CLASS_CAP,
    ManifoldSample,
    ManifoldTargetedSampler,
    RARE_RELATION_FLOORS,
)
from .annotator import (
    AnnotatedTriple,
    FinancialTaxonomyAnnotator,
)
from .exporter import (
    SFTDatasetExporter,
    SFTRecord,
)
from .hf_sync import (
    upload_to_huggingface,
)
from .builder import (
    FullSFTDatasetBuilder,
)

__all__ = [
    # Builder
    "FullSFTDatasetBuilder",
    # Sampler
    "ManifoldTargetedSampler",
    "ManifoldSample",
    "RARE_RELATION_FLOORS",
    "DOMINANT_CLASS_CAP",
    # Exporter
    "SFTDatasetExporter",
    "SFTRecord",
    # Annotator
    "FinancialTaxonomyAnnotator",
    "AnnotatedTriple",
    # Hugging Face Sync
    "upload_to_huggingface",
]
