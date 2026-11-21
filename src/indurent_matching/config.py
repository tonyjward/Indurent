"""Thresholds and settings for cleaning and matching.

Each value cites the EDA section (``notebooks/01_eda.ipynb``) where the evidence for it is shown.
"""

# --- Size (EDA 2.1) ---------------------------------------------------------------------------------------------
# Indurent's smallest unit band starts at 500 sq ft; smaller units are flagged as doubtful.
MIN_PLAUSIBLE_SQFT = 500
# Units over this are rare even among big UK logistics sheds, but real: noted, not flagged (EDA 2.1, 2.9).
VERY_LARGE_SQFT = 600_000

# --- Location (EDA 2.2) -----------------------------------------------------------------------------------------
# Natural Earth's 1:10m coastline is generalised (it smooths over docks, estuaries and reclaimed land), so points
# under this distance offshore count as coastal, not wrong. The cutoff sits in a clear gap: 8 units are 1-5 km out
# against 270 at 5 km+; 500 m would wrongly drop two coastal estates (Stone Mews 81, Ruby Mill 13).
COAST_TOLERANCE_KM = 1.0
# The same allowance for region boundaries, which come from the same generalised source.
REGION_TOLERANCE_KM = 1.0

# --- Distance to major cities (EDA section 5) -------------------------------------------------------------------
# (latitude, longitude) of each city centre. The list and any radius are parameters, and answers should state them.
MAJOR_CITIES = {
    "London": (51.5074, -0.1278),
    "Birmingham": (52.4862, -1.8904),
    "Manchester": (53.4808, -2.2426),
    "Leeds": (53.8008, -1.5491),
    "Glasgow": (55.8642, -4.2518),
    "Liverpool": (53.4084, -2.9916),
    "Newcastle": (54.9783, -1.6178),
    "Sheffield": (53.3811, -1.4701),
    "Bristol": (51.4545, -2.5879),
    "Edinburgh": (55.9533, -3.1883),
    "Leicester": (52.6369, -1.1398),
    "Nottingham": (52.9548, -1.1581),
    "Cardiff": (51.4816, -3.1791),
    "Southampton": (50.9097, -1.4044),
    "Coventry": (52.4068, -1.5197),
    "Bradford": (53.7960, -1.7594),
}

# --- Doors (EDA 2.6) --------------------------------------------------------------------------------------------
# A typical standalone shed has about one door per 9,000 sq ft. Fewer than this many sq ft per door is implausibly
# dense (Opal Forge 42 has one per ~1,200; listing M4 one per ~40), so the door count is flagged as doubtful.
# Applied only where the count is the unit's own, not a copied estate total.
MIN_SQFT_PER_DOOR = 2_000
# A few doors on a small unit is normal (Granite Depot 69: 3 doors on ~4,000 sq ft, judged plausible in EDA 2.6), so
# the density check applies only above this many doors.
MAX_DOORS_ALWAYS_PLAUSIBLE = 3

# --- Known issues (EDA 2.8, 2.9) --------------------------------------------------------------------------------
# Findings tied to particular Property Ids in this file, which no general rule can express: these fields are flagged
# as doubtful for these units, where a value is recorded. Review this list whenever the portfolio data is refreshed.
KNOWN_ISSUES = [
    {
        # Small units carry a big shed's eaves, doors and yard depth, and big units a small one's: the eaves-size
        # correlation reverses inside the block (-0.67 vs +0.62 elsewhere). Size is kept: its total matches
        # Indurent's published figure (EDA 2.1).
        "property_ids": range(1173, 1188),
        "fields": ["doors", "min_eaves_m", "max_eaves_m", "yard_depth_m"],
        "reason": "in a block of records (Property Ids 1173-1187) whose specs look attached to the wrong units",
    },
    {
        # Granite Mews 59: 2.1-2.4 m eaves on every unit, build years 1939-1953, EPC A on three of them.
        "property_ids": range(259, 264),
        "fields": ["min_eaves_m", "max_eaves_m", "build_year"],
        "reason": "Granite Mews 59's record doesn't hang together (house-height eaves, 1940s build, EPC A)",
    },
]

# --- EPC (EDA 2.9, 4.2) -----------------------------------------------------------------------------------------
# Valid ratings; anything else (the 7 Excel FALSE cells) becomes missing.
EPC_RATINGS = ["A", "B", "C", "D", "E", "F", "G"]

# --- Similarity: age (notebook 02_similarity_build_year) ---------------------------------------------------------
# Homogeneity: units in the listing's build-year band / units in the most frequent band. Build years are flat from 1970 to
# 2021, so fixed bands work; 5 years averages out year-to-year noise and stops the 2022 spike setting the scale.
AGE_BAND_YEARS = 5
# Pairwise: similarity halves for every this many years between build years (roughly one era of building spec).
AGE_HALF_LIFE_YEARS = 10
