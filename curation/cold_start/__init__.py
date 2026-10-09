"""
Cold-Start Pre-Training Data Curation Pipeline.
Implements:
- MinHash LSH (Broder 1997, Lee et al. ACL 2022)
- SemDeDup (Song et al. NeurIPS 2020, Abbas et al. 2023)
- Submodular Facility-Location Coreset Selection (Mirzasoleiman et al. ICML 2020, Sener & Savarese ICLR 2018)
"""

from curation.cold_start.minhash_lsh import MinHashLSHDeduplicator
from curation.cold_start.semdedup import SemanticDeduplicator
from curation.cold_start.coreset import FacilityLocationCoresetSelector
from curation.cold_start.pipeline import ColdStartCurationPipeline

__all__ = [
    "MinHashLSHDeduplicator",
    "SemanticDeduplicator",
    "FacilityLocationCoresetSelector",
    "ColdStartCurationPipeline",
]
