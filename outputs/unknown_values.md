# What to do with unknown values

By default a filter leaves out units whose value is unknown, and the answer says how many. Asked to, the agent keeps them. M5 is compared on age and size only (its coordinates are offshore), so units with no location can still match it well: keeping them changes the list.

## Turn 1

> **User:** Find the 10 units most similar to Chancery Depot (M5), keeping only units within 20 miles of a major city.

**Tool calls:**

- `get_listing({"listing": "M5"})`
  - M5: Chancery Depot, Unit 2
- `filter_rows({"conditions": [{"field": "miles_to_city", "op": "<=", "value": 20}], "include_unknown": false, "refine": false})`
  - Kept 213 of 1,250 units (miles_to_city <= 20.0). 358 had no usable value for miles_to_city and were excluded. (miles_to_city is unknown where a unit has no usable location.)
- `find_similar({"listing": "M5", "use_filter": true, "k": 10})`
  - Top 10 matches for M5 (Chancery Depot, Unit 2) from 213 units, scored on age and physical (size), weighted equally

**Answer:**

I found 10 matches for Chancery Depot, Unit 2 (M5), but **they're matched on age and size only, not location**. M5's coordinates are 27 km offshore, so its location can't be used. That also means I can't tell whether it sits on an estate Indurent already owns.

**About the filter:** the 20-mile limit applies to portfolio units. M5 has no usable location, so it doesn't have a distance to a city of its own. The filter kept 213 of 1,250 units. Another 358 units have no usable location, so their distance to a city is unknown and they were left out. I can include them if you'd like.

M5 is 16,614 sq ft and was built in 1966.

| # | Property Id | Estate / Unit | Overall | Age | Size | Built | Sq ft |
|---|---|---|---|---|---|---|---|
| 1 | 169 | Jade Bridge 6, Unit 3 | 0.761 | 0.574 | 0.948 | 1958 | 17,520 |
| 2 | 383 | Steel Bridge 36, Unit 2 | 0.757 | 0.933 | 0.581 | 1967 | 28,591 |
| 3 | 168 | Jade Bridge 6, Unit 2 | 0.683 | 0.871 | 0.496 | 1968 | 8,239 |
| 4 | 902 | Sable Crescent 13, Unit 9 | 0.644 | 0.500 | 0.787 | 1976 | 13,076 |
| 5 | 425 | Cobalt Grove 82, Unit 7 | 0.642 | 0.933 | 0.352 | 1967 | 5,843 |
| 6 | 384 | Steel Bridge 36, Unit 3 | 0.633 | 0.660 | 0.606 | 1972 | 27,432 |
| 7 | 854 | Velvet Park 60, Unit 9 | 0.605 | 0.287 | 0.923 | 1984 | 17,997 |
| 8 | 225 | Steel Harbour 45, Unit 1 | 0.571 | 0.406 | 0.735 | 1979 | 12,217 |
| 9 | 1023 | Velvet Bridge 96, Unit 1 | 0.564 | 0.189 | 0.939 | 1990 | 15,593 |
| 10 | 1009 | Crimson Tower 56, Unit 1 | 0.563 | 0.660 | 0.467 | 1972 | 7,760 |

**What drives the ranking:**
- **Property Id 169** ranks first mainly because its size is very close to M5's (size score 0.948).
- **Property Id 383** is close behind (0.757) mainly because of its age (1967, age score 0.933), even though it's much larger.
- No unit scores highest on both age and size, so none is the best match on every dimension.

The top three include two units on Jade Bridge 6 (Property Ids 169 and 168), and two units on Steel Bridge 36 also appear (Property Ids 383 and 384).

## Turn 2

> **User:** Now include the units whose location is unknown as well.

**Tool calls:**

- `filter_rows({"conditions": [{"field": "miles_to_city", "op": "<=", "value": 20}], "include_unknown": true, "refine": false})`
  - Kept 571 of 1,250 units (miles_to_city <= 20.0). 358 had no usable value for miles_to_city and were kept. (miles_to_city is unknown where a unit has no usable location.)
- `find_similar({"listing": "M5", "use_filter": true, "k": 10})`
  - Top 10 matches for M5 (Chancery Depot, Unit 2) from 571 units, scored on age and physical (size), weighted equally

**Answer:**

Adding the units with unknown location changes the top 10: **eight of the ten are new**. Property Id 169 (Jade Bridge 6, Unit 3) and Property Id 383 (Steel Bridge 36, Unit 2) are the only two left from the earlier list.

**What limits this:**
- **No location matching.** M5's coordinates are 27 km offshore, so units are still matched on age and size only.
- **The filter is no longer strict.** It now covers 571 of 1,250 units. Besides the 213 known to be within 20 miles of a major city, it includes the 358 units with no usable location. Those 358 could be any distance from a city, so they aren't confirmed to meet the 20-mile condition.
- **Where the new units come from:** the eight new units weren't in the earlier top 10, but they score higher than its lowest entry (0.563). So they must be among the units whose distance to a city is unknown.

M5 is 16,614 sq ft and was built in 1966.

| # | Property Id | Estate / Unit | Overall | Age | Size | Built | Sq ft |
|---|---|---|---|---|---|---|---|
| 1 | 656 | Onyx Beacon 83, Unit 4 | 0.801 | 0.933 | 0.668 | 1965 | 11,101 |
| 2 | 918 | Crimson Mill 42, Unit 1 | 0.783 | 0.707 | 0.859 | 1971 | 19,332 |
| 3 | 211 | Scarlet Depot 92, Unit 7 | 0.783 | 0.758 | 0.808 | 1970 | 13,429 |
| 4 | 210 | Scarlet Depot 92, Unit 6 | 0.778 | 1.000 | 0.555 | 1966 | 9,221 |
| 5 | 1235 | Marble Brook 22, Unit 1 | 0.777 | 0.707 | 0.848 | 1971 | 14,083 |
| 6 | 169 | Jade Bridge 6, Unit 3 | 0.761 | 0.574 | 0.948 | 1958 | 17,520 |
| 7 | 1191 | Crystal Terrace 36, Unit 1 | 0.758 | 0.660 | 0.856 | 1972 | 14,224 |
| 8 | 383 | Steel Bridge 36, Unit 2 | 0.757 | 0.933 | 0.581 | 1967 | 28,591 |
| 9 | 678 | Ruby Junction 67, Unit 12 | 0.735 | 0.758 | 0.713 | 1970 | 11,839 |
| 10 | 279 | Cedar Works 93, Unit 1 | 0.722 | 1.000 | 0.444 | 1966 | 37,377 |

**How the scores compare:**
- **Property Id 656** has the highest overall score (0.801), helped by a close build year (1965, age score 0.933).
- **Property Id 169** still has the best size score (0.948).
- **Property Ids 210 and 279** are the only units built in the same year as M5 (age score 1.000), but they score lower on size.
- No unit is the best match on both age and size.

Two Scarlet Depot 92 units are in the list (Property Ids 210 and 211).

If you'd rather stick to units confirmed to be within 20 miles of a city, the earlier list is the one to use.

*claude-opus-5-5, 23 November 2026. Tokens: 31,346 in, 2,942 out; $0.18.*
