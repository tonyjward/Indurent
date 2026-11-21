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
        # Granite Mews 59: 2.1-2.4 m eaves on every unit, a garage's ceiling height (the next lowest max eaves in the
        # portfolio is 5.2 m). Its build years (1939-1953) are not flagged: a 1940s industrial building is plausible,
        # and the EPC A ratings that seemed to contradict them can't be relied on (EPC is unreliable throughout).
        "property_ids": range(259, 264),
        "fields": ["min_eaves_m", "max_eaves_m"],
        "reason": "Granite Mews 59's eaves (2.1-2.4 m) are house height, not warehouse height",
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

# --- Similarity: location (notebook 03_similarity_location) -------------------------------------------------------
# Homogeneity: mean miles to this many nearest other estates; the best-placed portfolio estate scores 1. A single
# nearest estate can mislead (M1: one estate 4.4 miles away, then 30+ miles to the rest).
LOCATION_NEIGHBOUR_ESTATES = 5
# A listing "sits on" an estate if its nearest usable portfolio unit is within this distance (about 400 m). The
# listings are 9-71 m from their estate's units, and no portfolio unit is more than 156 m from its estate's centre
# (EDA 2.2).
SITS_ON_ESTATE_MILES = 0.25
# Pairwise: similarity halves for every this many miles apart (roughly a local industrial market).
LOCATION_HALF_LIFE_MILES = 10

# --- Similarity: physical, i.e. size (notebook 04_similarity_physical) --------------------------------------------
# Physical means size only: the one physical field every listing has. Eaves, doors and volume are context.
# Homogeneity: count the portfolio units of a similar size to the listing, and divide by the most units any
# portfolio unit has of a similar size to it. "Similar size" is a range around the listing's size in which the
# largest size is this many times the smallest: with 2, a 4,000 sq ft listing counts units from about 2,830 to
# 5,660 sq ft (4,000 / 1.41 to 4,000 x 1.41). The range moves with the listing, which avoids the jumps that fixed
# size bands made at their edges.
SIZE_WINDOW_RATIO = 2
# Pairwise: similarity halves each time one size is this many times the other. With 2, a unit twice or half the
# listing's size scores 0.5, and one four times or a quarter of it 0.25. A ratio rather than a fixed sq ft gap
# (unlike years and miles) because size differences are relative: 1,000 vs 2,000 sq ft is a different class of
# unit, while 100,000 vs 101,000 sq ft is effectively the same.
SIZE_HALF_LIFE_RATIO = 2
