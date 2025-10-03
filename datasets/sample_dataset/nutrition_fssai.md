# Sample Dataset: FSSAI Nutrition Content

This sample illustrates normalized nutrition values and portion sizes per food, as described in Chapter 3 (3.2, 3.3).

CSV (nutrition values per 100g)

```
food_id,food_name,energy_kcal,protein_g,fat_g,carb_g,sodium_mg,calcium_mg,iron_mg
F001,Poha (Flattened Rice),361,6.7,1.0,80.7,5,20,1.2
F002,Idli (Steamed Rice Cake),146,4.0,0.9,31.3,198,10,0.4
F003,Chana (Chickpeas, boiled),164,8.9,2.6,27.4,7,49,2.9
```

CSV (standard portion sizes)

```
food_id,portion_label,grams
F001,one cup,80
F002,one idli,50
F003,one katori,60
```

JSON (combined record with citations)

```json
{
  "food_id": "F001",
  "food_name": "Poha (Flattened Rice)",
  "nutrients_per_100g": {
    "energy_kcal": 361,
    "protein_g": 6.7,
    "fat_g": 1.0,
    "carb_g": 80.7,
    "sodium_mg": 5,
    "calcium_mg": 20,
    "iron_mg": 1.2
  },
  "portion_sizes": [{ "label": "one cup", "grams": 80 }],
  "citations": [{ "source": "FSSAI Handbook", "page": 42 }]
}
```
