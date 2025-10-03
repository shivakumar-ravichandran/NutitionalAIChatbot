# Sample Dataset: State-wise Availability

Indicates commonly available vs. scarce foods by region (3.2).

CSV (availability)

```
state,food_id,availability
Maharashtra,F001,common
Maharashtra,F003,common
Tamil Nadu,F001,common
Tamil Nadu,F003,scarce
```

JSON (availability list)

```json
{
  "state": "Maharashtra",
  "commonly_available": ["F001", "F003"],
  "scarce": []
}
```
