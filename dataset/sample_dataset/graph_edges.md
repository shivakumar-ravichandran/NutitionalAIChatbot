# Sample Dataset: Graph Edges (Neo4j KAG)

Nodes and relationships used by the KAG (3.5.2, 3.5.3). Represented as edge lists suitable for CSV import.

CSV (nodes)

```
id,label,name
F001,FoodItem,Poha
F002,FoodItem,Idli
I_Peanut,Ingredient,Peanut
N_SODIUM,Nutrient,Sodium
C_MH,Culture,Maharashtrian
A_ELDER,AgeGroup,Elderly
T001,TextChunk,Chunk: Poha nutrients
```

CSV (relationships)

```
start_id,rel_type,end_id,props
F001,HAS_INGREDIENT,I_Peanut,"{\"amount_g\": 5}"
F001,POPULAR_IN,C_MH,"{}"
F001,RICH_IN,N_SODIUM,"{\"per_100g_mg\": 5}"
A_ELDER,FOCUSES_ON,N_SODIUM,"{\"note\": \"moderation\"}"
T001,ABOUT,F001,"{}"
```

JSON (example subgraph)

```json
{
  "nodes": [
    { "id": "F001", "label": "FoodItem", "name": "Poha" },
    { "id": "C_MH", "label": "Culture", "name": "Maharashtrian" },
    { "id": "T001", "label": "TextChunk", "name": "Chunk: Poha nutrients" }
  ],
  "edges": [
    { "start": "F001", "type": "POPULAR_IN", "end": "C_MH", "props": {} },
    { "start": "T001", "type": "ABOUT", "end": "F001", "props": {} }
  ]
}
```
