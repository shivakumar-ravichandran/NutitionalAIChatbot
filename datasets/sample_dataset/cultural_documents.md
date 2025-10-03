# Sample Dataset: Cultural Documents

Illustrates regional cuisines, fasting practices, preparation styles, and local aliases (3.2, 3.3).

CSV (cultural notes)

```
culture_id,culture_name,region,practice,description,language
C_MH,Maharashtrian,Maharashtra,breakfast,"Poha is a common breakfast, often tempered with peanuts.",Marathi
C_TN,Tamil,Tamil Nadu,breakfast,"Idli is popular; often paired with sambar/chutney.",Tamil
C_JA,Jain,Pan-India,restriction,"Avoids root vegetables and certain after-sunset meals.",Hindi
```

CSV (aliases)

```
entity_id,entity_type,alias,language
F001,FoodItem,Kande Pohe,Marathi
F002,FoodItem,Idly,Tamil
I_Peanut,Ingredient,Groundnut,English
```

JSON (cultural entry)

```json
{
  "culture_id": "C_MH",
  "culture_name": "Maharashtrian",
  "regions": ["Maharashtra"],
  "practices": [{ "type": "breakfast", "note": "Poha is a common breakfast" }],
  "language_hints": ["Marathi"],
  "aliases": [
    { "entity_id": "F001", "alias": "Kande Pohe", "language": "Marathi" }
  ]
}
```
