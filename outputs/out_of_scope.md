# Questions the data can't answer

The agent never estimates a value or fills in missing data: asked for a rent figure, or a guess at a build year that isn't recorded, it says what the data doesn't hold and offers what it can do.

## Turn 1

> **User:** What rent per sq ft should we offer for M3?

**Tool calls:**

- `get_listing({"listing": "M3"})`
  - M3: Spitfire Park, Unit 42

**Answer:**

I can't suggest a rent per sq ft for M3. None of my tools hold rent or pricing data for the listing or for Indurent's portfolio, and I won't estimate a figure.

**What the tools do show for M3 (Spitfire Park, Unit 42):**
- **It sits on an estate Indurent already owns: Velvet Park 60.** That's the most useful location fact here.
- Size: 2,294 sq ft. Built in 1976. EPC rating B.
- Nearest city: Manchester, 14.0 miles away. Region label: Midlands. Location confidence: high.
- It can be compared on age, location and size. Nothing is missing and the tool reported no data-quality issues.

**What I can do to help you set a rent:**
- **Find the most similar units** in the portfolio, scored on age, location and size. You could then look up their rents in your own systems.
- **Find the nearest units on the map**, such as other units on Velvet Park 60.
- **Score how typical M3 is** of the portfolio on age, location and size. This shows how closely your current rents are likely to apply to it.

Would you like me to run any of these?

## Turn 2

> **User:** M2 has no build year recorded. What's your best guess for it?

**Tool calls:**

- `get_listing({"listing": "M2"})`
  - M2: Tech Hub, Unit 8

**Answer:**

I can't guess M2's build year. The tool has no build year recorded for it, and I won't fill in a missing value.

**What limits any comparison of M2 (Tech Hub, Unit 8):**
- **No build year**, so age can't be compared.
- **No coordinates**, so location can't be used. Nearest city, distance to a city and whether it sits on an estate we own are all unknown, and location confidence is "unknown".
- **No EPC rating** is recorded either.
- It can only be matched on **size**: 906 sq ft. The region label says South East, but the tool doesn't use that as a location.

**What I can still do:**
- **Find portfolio units of a similar size**, which is the only basis M2 can be scored on.
- **Score how typical M2's size is** of the portfolio.

To get an actual build year, the agent or the marketing details would be the best source. Once you have it, I can include age in the comparison.

*claude-opus-5-5, 23 November 2026. Tokens: 13,545 in, 1,023 out; $0.07.*
