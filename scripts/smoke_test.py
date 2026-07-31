"""Smoke test: load all three released products and check expected row counts."""
from load import load_dataset, load_gold, load_cleaned_leaderboards

c = load_dataset(); g = load_gold(); b = load_cleaned_leaderboards()
assert len(c) == 3058, len(c)
assert len(g) == 200, len(g)
assert len(b) == 4438, len(b)
model = [r for r in c if r.get("label_source") == "model"]
assert len(model) == 523, len(model)
print(f"OK  candidates={len(c)}  reference(human)={len(g)}  leaderboards={len(b)}  model-labeled={len(model)}")
