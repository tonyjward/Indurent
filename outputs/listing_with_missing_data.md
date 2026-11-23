# Listings with missing data

M2 has no build year or coordinates, so it can only be compared on size, and many units tie. M5's coordinates are offshore, so it can't be placed on the map at all.

## Turn 1

> **User:** Which portfolio units are most like Tech Hub, Unit 8?

**Tool calls:**

- `get_listing({"listing": "Tech Hub, Unit 8"})`
  - M2: Tech Hub, Unit 8
- `find_similar({"listing": "Tech Hub, Unit 8", "use_filter": false})`
  - Top 10 matches for M2 (Tech Hub, Unit 8) from 1,250 units, scored on physical (size) alone

**Answer:**

**Tech Hub, Unit 8 (M2) can only be matched on size.** It has no coordinates, so location can't be used, and no build year, so age can't be compared. Because there's no usable location, I also can't tell whether it sits on an estate Indurent already owns. One more caveat from the tool: 11 units score 0.98 or more on size alone, so this top 10 is one of several equally good lists. The ranking below comes down to small differences in floor area.

The listing is **906 sq ft**. Top 10 matches by size, out of 1,250 units:

| Property Id | Estate, Unit | Size (sq ft) | Build year | Physical score |
|---|---|---|---|---|
| 602 | Quartz Wharf 16, Unit 21 | 905 | 1979 | 1.000 |
| 689 | Ruby Junction 67, Unit 23 | 905 | 2006 | 0.999 |
| 221 | Opal Hub 4, Unit 2 | 905 | 2022 | 0.999 |
| 874 | Velvet Park 60, Unit 29 | 902 | 2004 | 0.997 |
| 698 | Ruby Junction 67, Unit 32 | 901 | 1998 | 0.995 |
| 24 | Sable Mill 91, Unit 6 | 898 | 2016 | 0.992 |
| 1101 | Stone Mews 81, Unit 4 | 889 | 1986 | 0.982 |
| 601 | Quartz Wharf 16, Unit 20 | 923 | 2017 | 0.981 |
| 863 | Velvet Park 60, Unit 18 | 923 | 1976 | 0.981 |
| 102 | Indigo Square 80, Unit 6 | 886 | 2017 | 0.979 |

The build years are only there for reference. They didn't affect the scores, because the listing has no build year to compare them with.

To get a more useful shortlist, I could filter the portfolio first, for example by build year, size range or region, and then match within that set.

## Turn 2

> **User:** Which of our units are nearest to M5 on the map?

**Tool calls:**

- `get_listing({"listing": "M5"})`
  - M5: Chancery Depot, Unit 2
- `find_nearest({"listing": "M5", "use_filter": false})`
  - Error: M5 can't be placed on the map: coordinates are offshore, so can't be trusted. Nearest properties can't be found for it; say so in the answer.

**Answer:**

I can't find the nearest units to **M5 (Chancery Depot, Unit 2)** because it can't be placed on the map. Its coordinates are 27 km offshore, so they can't be trusted, and the nearest-unit search won't run without a usable location. For the same reason I can't tell whether it sits on an estate Indurent already owns. The listing also has no region or nearest city recorded, so I can't narrow the area another way.

M5 can still be compared on **age** (built 1966) and **size** (16,614 sq ft). If it helps, I can find the portfolio units most similar to it on those two measures, but that would be a similarity match, not a distance one. If you can get corrected coordinates for Chancery Depot, the nearest-unit search would work.

*claude-opus-5-5, 23 November 2026. Tokens: 18,482 in, 1,244 out; $0.10.*
