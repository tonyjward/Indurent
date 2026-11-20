"""Thresholds and settings for cleaning and matching.

Each value cites the EDA section (``notebooks/01_eda.ipynb``) where the evidence for it is shown.
"""

# --- Size (EDA 2.1) ---------------------------------------------------------------------------------------------
# Indurent's smallest unit band starts at 500 sq ft; smaller units are flagged as doubtful.
MIN_PLAUSIBLE_SQFT = 500
# Units over this are rare even among big UK logistics sheds, but real: noted, not flagged (EDA 2.1, 2.9).
VERY_LARGE_SQFT = 600_000
