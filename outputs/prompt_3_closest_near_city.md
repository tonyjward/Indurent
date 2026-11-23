# Prompt 3: the closest properties near a major city

The brief's third prompt, word for word: a filter (keep units within 10 miles of a major city), then the nearest units on the map from what's left, with the units of unknown location counted.

> **User:** Find the closest properties to the marketed property, after excluding any property in the portfolio that is more than 10 miles from a major city.

**Tool calls:**

- `list_listings({})`
  - 5 marketed listings: M1, M2, M3, M4, M5. 2 can't be compared on every dimension: M2 (no age or location), M5 (no location).
- `filter_rows({"conditions": [{"field": "miles_to_city", "op": "<=", "value": 10}], "include_unknown": false, "refine": false})`
  - Kept 50 of 1,250 units (miles_to_city <= 10.0). 358 had no usable value for miles_to_city and were excluded. (miles_to_city is unknown where a unit has no usable location.)
- `find_nearest({"listing": "M1", "use_filter": true})`
  - The 10 units nearest to M1 (Cherry Lane, Unit 10) on the map, from 50 units with a usable location
- `find_nearest({"listing": "M3", "use_filter": true})`
  - The 10 units nearest to M3 (Spitfire Park, Unit 42) on the map, from 50 units with a usable location
- `find_nearest({"listing": "M4", "use_filter": true})`
  - The 10 units nearest to M4 (Stable Lane, Unit 14) on the map, from 50 units with a usable location

**Answer:**

You didn't say which marketed listing you meant, and there are five, so I ran the search for every listing that has a usable location.

**What limits the answer**
- **Only 3 of the 5 listings could be searched.** Nearest-on-map needs coordinates. M2 (Tech Hub, Unit 8) has none. M5 (Chancery Depot, Unit 2) has coordinates 27 km offshore, so its location can't be trusted.
- **The filter leaves a small set.** Keeping units within 10 miles of a major city left **50 of 1,250 units**. Another **358 units were left out because their distance to a city is unknown.** I can include them if you'd like.
- **Every listing that could be searched is on an estate Indurent already owns, and the filter removed that estate.** M1 is on Scarlet Pavilion 78, M3 on Velvet Park 60 and M4 on Sable Mill 91. None of those estates' units are within 10 miles of a city, so the matches below are all further away.
- **Data-quality notes:**
  - M1's location is medium confidence: it's 87 km outside its labelled region (North West) and falls in Wales.
  - M4's location is medium confidence: it's 15 km outside its labelled region (South East) and falls in East of England. Its door count (16 doors on 617 sq ft) is implausibly dense, so doubtful.

**M1: Cherry Lane, Unit 10 (16,860 sq ft, built 1977, 72.8 mi from Cardiff)**
| Property Id | Estate / Unit | Miles from listing | Built | Sq ft | Nearest city (mi) |
|---|---|---|---|---|---|
| 348 | Marble Works 29 / U1 | 74.6 | 1987 | 887,038 | Birmingham (7.9) |
| 400, 399, 398 | Sapphire Junction 78 / U3, U2, U1 | 88.1 | 2020, 2015, 1987 | 27,960; 42,332; 31,721 | Birmingham (8.6) |
| 401 | Sapphire Junction 78 / U4 | 88.1 | 2000 | 38,568 | Birmingham (8.7) |
| 168 | Jade Bridge 6 / U2 | 97.0 | 1968 | 8,239 | Manchester (7.1) |
| 167, 169, 170 | Jade Bridge 6 / U1, U3, U4 | 97.1 | 1977, 1958, 1989 | 7,542; 17,520; 7,908 | Manchester (7.1) |
| 483 | Onyx Oaks 5 / U1 | 103.1 | 1987 | 173,913 | Leicester (9.0) |

**M3: Spitfire Park, Unit 42 (2,294 sq ft, built 1976, 14.0 mi from Manchester)**
| Property Id | Estate / Unit | Miles from listing | Built | Sq ft | Nearest city (mi) |
|---|---|---|---|---|---|
| 170, 169, 167 | Jade Bridge 6 / U4, U3, U1 | 9.1 | 1989, 1958, 1977 | 7,908; 17,520; 7,542 | Manchester (7.1) |
| 168 | Jade Bridge 6 / U2 | 9.2 | 1968 | 8,239 | Manchester (7.1) |
| 503 | Scarlet Arc 73 / U1 | 16.2 | 2005 | 134,579 | Sheffield (9.0) |
| 1162 | Topaz Harbour 22 / U1 | 51.5 | 2004 | 15,268 | Nottingham (8.2) |
| 483 | Onyx Oaks 5 / U1 | 62.7 | 1987 | 173,913 | Leicester (9.0) |
| 348 | Marble Works 29 / U1 | 64.8 | 1987 | 887,038 | Birmingham (7.9) |
| 401, 400 | Sapphire Junction 78 / U4, U3 | 68.8 | 2000, 2020 | 38,568; 27,960 | Birmingham (8.7, 8.6) |

12 units are within 68.8 miles of M3, so the last places in this top 10 are effectively tied with units not shown.

**M4: Stable Lane, Unit 14 (617 sq ft, built 2000, 40.6 mi from London)**
| Property Id | Estate / Unit | Miles from listing | Built | Sq ft | Nearest city (mi) |
|---|---|---|---|---|---|
| 350 | Onyx Pointe 12 / U1 | 33.2 | unknown | 284,391 | London (8.6) |
| 894, 899, 896, 900, 897, 901, 895 | Sable Crescent 13 / U1–U4, U6–U8 | 39.9 | 1992, 2005, 2013, 1977, 1982, 2018, 2022 | 8,053; 14,818; 8,391; 4,712; 4,555; 20,251; 5,944 | London (6.9) |
| 898, 902 | Sable Crescent 13 / U5, U9 | 40.0 | 1972, 1976 | 5,614; 13,076 | London (6.9) |

If you meant one listing in particular, tell me which and I'll focus on it. I can also rerun the search with the 358 units whose distance to a city is unknown.

*claude-opus-5-5, 23 November 2026. Tokens: 19,538 in, 2,356 out; $0.13.*
