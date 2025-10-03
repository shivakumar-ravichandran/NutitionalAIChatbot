# Sample Dataset: Age-Specific Guidance

Examples of tone, structure, and nutrition focus by age group (3.2, 3.7).

CSV (age groups and style)

```
age_group_id,label,tone_style,nutrition_focus
A_CHILD,6-12,Simple and encouraging,"iron, calcium"
A_ADULT,18-60,Structured and motivational,"balanced macros, sodium moderation"
A_ELDER,60+,Polite and calm,"protein quality, easy digestion, sodium moderation"
```

JSON (age group templates)

```json
{
  "age_group_id": "A_ELDER",
  "tone_style": "Polite and calm",
  "template": {
    "intro": "Let's keep it gentle and easy to prepare.",
    "sections": [
      "2-3 meal ideas",
      "why it works (1-2 lines)",
      "safety reminders (short)",
      "citations"
    ]
  }
}
```
