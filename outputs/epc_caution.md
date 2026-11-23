# A filter on EPC rating, with its warning

The portfolio's EPC ratings look unreliable (newer buildings are mostly rated E), so the agent applies the filter as asked but warns that the result should be treated with caution.

> **User:** Find the units most similar to M3 that have an EPC rating of C or better.

**Tool calls:**

- `get_listing({"listing": "M3"})`
  - M3: Spitfire Park, Unit 42
- `filter_rows({"conditions": [{"field": "epc_rating", "op": "in", "value": ["A+", "A", "B", "C"]}], "include_unknown": false, "refine": false})`
  - Kept 575 of 1,250 units (epc_rating in ['A+', 'A', 'B', 'C']). 326 had no usable value for epc_rating and were excluded. (epc_rating is unknown where none is recorded.)
- `find_similar({"listing": "M3", "use_filter": true})`
  - Top 10 matches for M3 (Spitfire Park, Unit 42) from 575 units, scored on age, location and physical (size), weighted equally

**Answer:**

M3 (Spitfire Park, Unit 42) **sits on Velvet Park 60, an estate Indurent already owns**, and all 10 matches below are units on that estate.

**Limits on this answer:**
- **EPC ratings:** Treat this result with caution. The portfolio's EPC ratings look unreliable, because newer buildings are mostly rated E.
- **Units left out:** 575 of 1,250 units have a rating of C or better (A+, A, B or C). Another 326 have no EPC rating recorded, so they weren't searched. I can include them if you'd like.
- **Data quality:** M3 can be compared on age, location and size, and the tools report no data-quality notes for it.

**Top 10 similar units with EPC rating C or better** (each dimension weighted equally; M3 is 2,294 sq ft, built in 1976):

| Property Id | Unit | Overall | Age | Location | Size | Build year | Size (sq ft) | Miles from M3 |
|---|---|---|---|---|---|---|---|---|
| 869 | Unit 24 | 0.84 | 0.812 | 0.993 | 0.714 | 1973 | 1,638 | 0.1 |
| 865 | Unit 20 | 0.815 | 0.758 | 0.998 | 0.689 | 1980 | 3,332 | 0.0 |
| 863 | Unit 18 | 0.799 | 1.0 | 0.996 | 0.402 | 1976 | 923 | 0.1 |
| 855 | Unit 10 | 0.797 | 0.871 | 0.996 | 0.523 | 1978 | 1,199 | 0.1 |
| 878 | Unit 33 | 0.772 | 0.66 | 0.993 | 0.663 | 1970 | 1,520 | 0.1 |
| 849 | Unit 4 | 0.771 | 0.435 | 0.999 | 0.877 | 1988 | 2,013 | 0.0 |
| 851 | Unit 6 | 0.74 | 0.933 | 0.994 | 0.294 | 1977 | 675 | 0.1 |
| 852 | Unit 7 | 0.732 | 0.406 | 0.999 | 0.79 | 1989 | 2,904 | 0.0 |
| 858 | Unit 13 | 0.729 | 0.5 | 0.998 | 0.689 | 1986 | 3,331 | 0.0 |
| 876 | Unit 31 | 0.719 | 0.871 | 0.999 | 0.288 | 1974 | 661 | 0.0 |

All 10 units score between 0.993 and 0.999 on location, so the ranking comes down to their age and size scores. Property Id 869 has the highest overall score (0.84), but others beat it on single dimensions:
- **Age:** Property Id 863 scores 1.0, with the same build year as M3.
- **Size:** Property Id 849 scores 0.877.

*claude-opus-5-5, 23 November 2026. Tokens: 11,278 in, 1,349 out; $0.07.*
