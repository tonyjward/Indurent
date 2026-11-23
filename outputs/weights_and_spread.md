# Weights, and a spread across estates

Similarity weights age, location and size equally unless asked; here location counts double. Then a cap of 2 units per estate, for a spread rather than one estate's units.

## Turn 1

> **User:** Find the 10 units most similar to M1, with location counting twice as much as age and size.

**Tool calls:**

- `get_listing({"listing": "M1"})`
  - M1: Cherry Lane, Unit 10
- `find_similar({"listing": "M1", "k": 10, "use_filter": false, "weights": {"location": 2, "age": 1, "physical": 1}})`
  - Top 10 matches for M1 (Cherry Lane, Unit 10) from 1,250 units, scored on age, location and physical (size), weighted as requested: {'location': 2.0, 'age': 1.0, 'physical': 1.0}

**Answer:**

I compared M1 (Cherry Lane, Unit 10: 16,860 sq ft, built 1977) with all 1,250 portfolio units, with location counting twice as much as age and size.

**Limits on the answer:**
- **The location is uncertain.** M1's location has medium confidence. Its point is 87 km outside its labelled region (North West) and falls in Wales. The tool gives its nearest city as Cardiff, 72.8 miles away.
- **Indurent already owns the estate M1 sits on.** That estate is Scarlet Pavilion 78, and 3 of the 10 matches are on it.

**Top 10 matches**

| # | Property Id | Estate / Unit | Overall | Age | Location | Size | Built | Sq ft | Miles from M1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 586 | Quartz Wharf 16, Unit 5 | 0.843 | 0.933 | 0.734 | 0.972 | 1976 | 17,340 | 4.5 |
| 2 | 1157 | Scarlet Pavilion 78, Unit 7 | 0.817 | 0.758 | 0.995 | 0.519 | 1981 | 8,752 | 0.1 |
| 3 | 584 | Quartz Wharf 16, Unit 3 | 0.791 | 0.871 | 0.737 | 0.820 | 1975 | 20,565 | 4.4 |
| 4 | 1153 | Scarlet Pavilion 78, Unit 3 | 0.731 | 0.812 | 0.997 | 0.119 | 1974 | 2,011 | 0.0 |
| 5 | 606 | Quartz Wharf 16, Unit 25 | 0.725 | 0.467 | 0.737 | 0.961 | 1988 | 16,208 | 4.4 |
| 6 | 612 | Quartz Wharf 16, Unit 31 | 0.699 | 0.758 | 0.734 | 0.570 | 1981 | 9,603 | 4.5 |
| 7 | 642 | Quartz Wharf 16, Unit 61 | 0.689 | 0.758 | 0.733 | 0.531 | 1973 | 8,945 | 4.5 |
| 8 | 1151 | Scarlet Pavilion 78, Unit 1 | 0.654 | 0.406 | 0.994 | 0.220 | 1990 | 3,712 | 0.1 |
| 9 | 633 | Quartz Wharf 16, Unit 52 | 0.652 | 0.707 | 0.734 | 0.434 | 1982 | 38,889 | 4.5 |
| 10 | 634 | Quartz Wharf 16, Unit 53 | 0.645 | 1.000 | 0.737 | 0.105 | 1977 | 1,768 | 4.4 |

**What the scores show:**
- **Two estates make up the whole list:** Scarlet Pavilion 78, the estate M1 is on, and Quartz Wharf 16, about 4.4–4.5 miles away.
- **The Scarlet Pavilion units have the highest location scores (0.994–0.997)**, but lower size scores.
- **Property Id 586 still ranks first** even with the extra weight on location. Its age (0.933) and size (0.972) scores are high. Its location score is 0.734, versus 0.995 for Property Id 1157 in second place.

None of the matches has fields flagged as doubtful.

## Turn 2

> **User:** Same again, but at most 2 units from any one estate.

**Tool calls:**

- `find_similar({"listing": "M1", "k": 10, "use_filter": false, "weights": {"location": 2, "age": 1, "physical": 1}, "max_per_estate": 2})`
  - Top 10 matches for M1 (Cherry Lane, Unit 10) from 1,250 units, scored on age, location and physical (size), weighted as requested: {'location': 2.0, 'age': 1.0, 'physical': 1.0}

**Answer:**

Here's the same search (location weighted ×2) with at most 2 units from any one estate. The limits from before still apply, and one new one matters:

- **Three matches have no usable location.** Property Ids 673, 418 and 1233 have no location data, so they score 0 on location and have no distance from M1. Their rank comes from their age and size scores only.
- **M1's own location is uncertain.** It has medium confidence: the point is 87 km outside its labelled region (North West) and falls in Wales.
- **M1 is on Scarlet Pavilion 78, an estate Indurent already owns.** 2 of the 10 matches are on it.

| # | Property Id | Estate / Unit | Overall | Age | Location | Size | Built | Sq ft | Miles from M1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 586 | Quartz Wharf 16, Unit 5 | 0.843 | 0.933 | 0.734 | 0.972 | 1976 | 17,340 | 4.5 |
| 2 | 1157 | Scarlet Pavilion 78, Unit 7 | 0.817 | 0.758 | 0.995 | 0.519 | 1981 | 8,752 | 0.1 |
| 3 | 584 | Quartz Wharf 16, Unit 3 | 0.791 | 0.871 | 0.737 | 0.820 | 1975 | 20,565 | 4.4 |
| 4 | 1153 | Scarlet Pavilion 78, Unit 3 | 0.731 | 0.812 | 0.997 | 0.119 | 1974 | 2,011 | 0.0 |
| 5 | 673 | Ruby Junction 67, Unit 7 | 0.487 | 1.000 | no location | 0.948 | 1977 | 17,793 | no location |
| 6 | 418 | Verdant Oaks 7, Unit 9 | 0.476 | 1.000 | no location | 0.906 | 1977 | 15,273 | no location |
| 7 | 1233 | Opal Meadow 55, Unit 1 | 0.475 | 1.000 | no location | 0.901 | 1977 | 15,194 | no location |
| 8 | 904 | Topaz Arc 37, Unit 1 | 0.469 | 0.933 | 0.000 | 0.945 | 1978 | 15,926 | 163.7 |
| 9 | 796 | Crimson Forge 9, Unit 36 | 0.465 | 0.933 | 0.000 | 0.927 | 1976 | 15,626 | 117.5 |
| 10 | 71 | Onyx Harbour 29, Unit 2 | 0.453 | 1.000 | 0.001 | 0.812 | 1977 | 20,775 | 99.0 |

**What changed:**
- **The top 4 are the same as before**, two each from Quartz Wharf 16 and Scarlet Pavilion 78.
- **Below that, overall scores drop from 0.731 to 0.487 and below.** The other estates' units score 1.000 or 0.933 on age and 0.812–0.948 on size, but 0 (or near it) on location.
- **Places 8–10 are 99 to 164 miles from M1.**
- **Places 5–7 are the units with no location data**, so their distance from M1 is unknown.

So with the limit of 2 per estate, only the top 4 match M1 on location.

*claude-opus-5-5, 23 November 2026. Tokens: 21,622 in, 2,794 out; $0.14.*
