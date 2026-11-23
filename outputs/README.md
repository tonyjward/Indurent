# Example outputs

Real conversations with the agent (claude-opus-5-5, 23 November 2026), made by `scripts/run_examples.py`. Each file shows the questions, the tool calls Claude made (with their arguments and a one-line outcome) and its answers.

| File | What it shows |
|---|---|
| [`prompt_1_most_similar.md`](prompt_1_most_similar.md) | The brief's first prompt, word for word. It doesn't say which listing, so the agent has to ask or answer for all five, and say what each listing can be compared on. |
| [`prompt_2_homogeneity.md`](prompt_2_homogeneity.md) | The brief's second prompt, word for word: a score for each dimension, separately, with the reading of 'correlation score for homogeneity' stated. |
| [`prompt_3_closest_near_city.md`](prompt_3_closest_near_city.md) | The brief's third prompt, word for word: a filter (keep units within 10 miles of a major city), then the nearest units on the map from what's left, with the units of unknown location counted. |
| [`refining_filters.md`](refining_filters.md) | The filtered set is held in the conversation, so each follow-up narrows it further (refine), a search can run on the result, and a new question starts again from the whole portfolio. |
| [`unknown_values.md`](unknown_values.md) | By default a filter leaves out units whose value is unknown, and the answer says how many. Asked to, the agent keeps them. M5 is compared on age and size only (its coordinates are offshore), so units with no location can still match it well: keeping them changes the list. |
| [`listing_with_missing_data.md`](listing_with_missing_data.md) | M2 has no build year or coordinates, so it can only be compared on size, and many units tie. M5's coordinates are offshore, so it can't be placed on the map at all. |
| [`weights_and_spread.md`](weights_and_spread.md) | Similarity weights age, location and size equally unless asked; here location counts double. Then a cap of 2 units per estate, for a spread rather than one estate's units. |
| [`epc_caution.md`](epc_caution.md) | The portfolio's EPC ratings look unreliable (newer buildings are mostly rated E), so the agent applies the filter as asked but warns that the result should be treated with caution. |
| [`out_of_scope.md`](out_of_scope.md) | The agent never estimates a value or fills in missing data: asked for a rent figure, or a guess at a build year that isn't recorded, it says what the data doesn't hold and offers what it can do. |
