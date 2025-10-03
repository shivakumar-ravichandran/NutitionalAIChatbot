# Sample Dataset: Canonical Entities

Reflects resolved FoodItem, Ingredient, Nutrient, Culture, AgeGroup entities (3.2, 3.3).

CSV (FoodItem)

```
id,name,alt_names,fssai_code,dietary_tags
F001,Poha,"Kande Pohe|Flattened Rice",FSSAI_123,vegetarian
F002,Idli,"Idly|Steamed Rice Cake",FSSAI_456,vegetarian
```

CSV (Ingredient)

```
id,name,alt_names,allergen,dietary_tags
I_Peanut,Peanut,"Groundnut",true,vegan
I_Rice,Rice,,false,vegan
```

CSV (Nutrient)

```
id,name,unit
N_ENERGY,Energy,kcal
N_PROTEIN,Protein,g
N_SODIUM,Sodium,mg
```

JSON (Culture)

```json
{
  "id": "C_MH",
  "name": "Maharashtrian",
  "regions": ["Maharashtra"],
  "practices": ["breakfast: poha"],
  "language_hints": ["Marathi"]
}
```

JSON (AgeGroup)

```json
{
  "id": "A_CHILD",
  "tone_style": "Simple and encouraging",
  "nutrition_focus": ["iron", "calcium"]
}
```
