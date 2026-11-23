# Prompt 2: homogeneity scores

The brief's second prompt, word for word: a score for each dimension, separately, with the reading of 'correlation score for homogeneity' stated.

> **User:** Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the estate, scoring each by physical characteristics, location, and age separately.

**Tool calls:**

- `list_listings({})`
  - 5 marketed listings: M1, M2, M3, M4, M5. 2 can't be compared on every dimension: M2 (no age or location), M5 (no location).
- `get_homogeneity_scores({"listing": "M1"})`
  - Homogeneity scores for M1 (Cherry Lane, Unit 10)
- `get_homogeneity_scores({"listing": "M2"})`
  - Homogeneity scores for M2 (Tech Hub, Unit 8)
- `get_homogeneity_scores({"listing": "M3"})`
  - Homogeneity scores for M3 (Spitfire Park, Unit 42)
- `get_homogeneity_scores({"listing": "M4"})`
  - Homogeneity scores for M4 (Stable Lane, Unit 14)
- `get_homogeneity_scores({"listing": "M5"})`
  - Homogeneity scores for M5 (Chancery Depot, Unit 2)

**Answer:**

You didn't name a listing, so I scored all five marketed listings. I read "correlation score for homogeneity" as **how typical each listing is of Indurent's portfolio**. Each dimension is scored from 0 to 1, where 1 means the most typical value in the portfolio.

**Limits first:**
- **M2 (Tech Hub, Unit 8)** can only be scored on physical. It has no coordinates and no build year.
- **M5 (Chancery Depot, Unit 2)** has no location score. Its coordinates are 27 km offshore, so they can't be used.
- **M1 (Cherry Lane, Unit 10)** has a medium-confidence location. It is labelled North West, but its point is 87 km outside that region, in Wales.
- **M4 (Stable Lane, Unit 14)** also has a medium-confidence location. It is labelled South East, but its point is 15 km outside that region, in the East of England. Its door count is also doubtful: 16 doors on 617 sq ft (about 39 sq ft per door) is implausibly dense.

**Scores**

| Listing | Age | Location | Physical (size) | Where it sits |
|---|---|---|---|---|
| M1 Cherry Lane, Unit 10 | 0.86 (1977; 14.8% of units older, 82.7% newer) | 0.23 (better placed than 6.6% of units) | 0.41 (16,860 sq ft; larger than 78.5% of units) | **On Indurent's Scarlet Pavilion 78 estate** |
| M2 Tech Hub, Unit 8 | n/a | n/a | 0.38 (906 sq ft; smaller than 95.2% of units) | Unknown |
| M3 Spitfire Park, Unit 42 | 0.86 (1976; 12.9% older, 85.2% newer) | 0.62 (better placed than 62.9% of units) | 0.87 (2,294 sq ft; 26.5% of units smaller, 73.5% larger) | **On Indurent's Velvet Park 60 estate** |
| M4 Stable Lane, Unit 14 | 0.86 (2000; 58.0% older, 40.0% newer) | 0.73 (better placed than 85.4% of units) | 0.17 (617 sq ft; smaller than 98.7% of units) | **On Indurent's Sable Mill 91 estate** |
| M5 Chancery Depot, Unit 2 | 0.20 (1966; only 2.0% of units older) | n/a | 0.41 (16,614 sq ft; larger than 78.2% of units) | Unknown |

**Summary**
- **M3** is the only listing that scores fairly high on all three dimensions: 0.86 for age, 0.62 for location and 0.87 for physical. M4 does score higher on location (0.73 against 0.62).
- **M4** is typical on age and location but very atypical on size, because it is smaller than almost every unit. Its eaves height is 7.0 m, against a median of 4.5 m for 6 similar-sized units.
- **M1** is typical on age but has the lowest location score of the three listings that could be scored on location (0.23).
- **M5** is atypical on age because it is older than almost all units. Its eaves height is 5.3 m, against a median of 6.3 m for 45 similar-sized units.
- **M2** can only be judged on size, and it's smaller than most units.

*claude-opus-5-5, 23 November 2026. Tokens: 13,892 in, 1,525 out; $0.09.*
