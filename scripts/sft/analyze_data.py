import json
import pandas as pd

with open("data_processed/medec_cot/sft_cot_training_data_1.jsonl") as f:
    rows = [json.loads(line) for line in f]

df = pd.DataFrame(rows)
df["note_len"] = df["note"].str.len()
df["reasoning_len"] = df["reasoning"].str.len()

summary = (
    df.groupby(["role", "task"])
      [["note_len", "reasoning_len"]]
      .mean()
      .round(1)
)

print(summary)