# Sample Datasets Index

This folder contains small, representative samples for each dataset family described in Chapter 3. Use these as templates for schema, formatting, and ingestion tests.

- nutrition_fssai.md — Normalized nutrient values (per 100g), standard portion sizes, and a combined JSON record with citations.
- cultural_documents.md — Cultural notes (cuisines, fasting, preparation styles) and alias maps for regional names.
- age_style_guidance.md — Age-group tone/structure guidance and example templates for generation.
- availability_by_state.md — State-wise availability for commonly available vs. scarce foods.
- canonical_entities.md — Canonicalized entities (FoodItem, Ingredient, Nutrient, Culture, AgeGroup) in CSV/JSON.
- text_chunks.md — Segmented retrievable text chunks with citations and ABOUT links.
- graph_edges.md — KAG nodes and relationships as CSV edge lists and a JSON subgraph example.

Notes

- IDs like F001 (FoodItem), I_Peanut (Ingredient), N_SODIUM (Nutrient), C_MH (Culture), A_ELDER (AgeGroup), and T001 (TextChunk) are consistent across files.
- CSVs are designed to be import-friendly; JSON examples show how to represent richer nested structures.
- Extend these samples with additional fields (e.g., rda_by_age_gender, dietary_tags, citation line ranges) as needed.
