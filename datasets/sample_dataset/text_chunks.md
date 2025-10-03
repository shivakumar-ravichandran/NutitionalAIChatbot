# Sample Dataset: TextChunks with Citations

Segmented, retrievable chunks tied to entities (3.2, 3.3, 3.4).

CSV (chunks)

```
id,source_type,text,language,citations,scope
T001,FSSAI,"Poha provides ~361 kcal/100g with ~6.7g protein.",English,"FSSAI:42","foods|nutrients"
T002,CULTURE,"Poha is a common Maharashtrian breakfast.",English,"CulturalGuide:12","culture|foods"
T003,AGE_STYLE,"For elderly, keep sodium modest and explain portions simply.",English,"AgeStyle:5","age|safety"
T004,AVAILABILITY,"Chickpeas are commonly available in Maharashtra.",English,"AvailList:3","availability|foods"
```

JSON (chunk with ABOUT edges)

```json
{
  "id": "T001",
  "source_type": "FSSAI",
  "text": "Poha provides ~361 kcal/100g with ~6.7g protein.",
  "language": "English",
  "citations": [{ "source": "FSSAI Handbook", "page": 42 }],
  "scope": ["foods", "nutrients"],
  "about": ["F001", "N_ENERGY", "N_PROTEIN"]
}
```
