"""Minimal loader for the cross-paper result comparability resource."""
import json, os
_HERE = os.path.dirname(os.path.abspath(__file__))
def _jsonl(p):
    with open(os.path.join(_HERE, p)) as f:
        return [json.loads(line) for line in f if line.strip()]
def load_dataset():              return _jsonl("dataset/comparekg_candidates.jsonl")
def load_gold():                 return _jsonl("dataset/comparekg_gold.jsonl")
def load_cleaned_leaderboards(): return _jsonl("cleaned_leaderboards/cleaned_leaderboards.jsonl")
def load_census():
    with open(os.path.join(_HERE, "census/census.json")) as f: return json.load(f)
if __name__ == "__main__":
    print("candidate pairs:", len(load_dataset()))
    print("human-labeled pairs:", len(load_gold()))
    print("cleaned leaderboard entries:", len(load_cleaned_leaderboards()))
