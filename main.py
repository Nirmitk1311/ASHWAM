import json
import argparse
import os
from typing import List, Dict

from scorer import Scorer # Import the Scorer class
from ashwam_types import SemanticObject # Import SemanticObject


def extract_semantic_objects(journal_entry: str) -> List[SemanticObject]:
    """
    Placeholder for the extraction logic. In a real scenario, this would use an LLM or a pipeline
    to extract semantic objects from the journal entry.
    """
    return []

def load_data_from_jsonl(file_path: str) -> Dict[str, dict]:
    """
    Loads data from a JSONL file, indexed by journal_id.
    """
    data_by_journal = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_by_journal[data["journal_id"]] = data
    return data_by_journal

def load_semantic_objects(data_by_journal: Dict[str, dict], key: str = "annotations") -> Dict[str, List[SemanticObject]]:
    """
    Converts raw dictionary data into a dictionary of lists of SemanticObject instances.
    """
    all_semantic_objects: Dict[str, List[SemanticObject]] = {}
    for journal_id, journal_data in data_by_journal.items():
        objects = []
        for item in journal_data.get(key, []):
            obj = SemanticObject(
                domain=item["domain"],
                evidence_span=item["evidence_span"],
                polarity=item["polarity"],
                intensity_bucket=item.get("intensity_bucket", "unknown"),
                arousal_bucket=item.get("arousal_bucket", "unknown"),
                time_bucket=item["time_bucket"]
            )
            objects.append(obj)
        all_semantic_objects[journal_id] = objects
    return all_semantic_objects

def main():
    parser = argparse.ArgumentParser(description="Ashwam Evaluation Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to the data directory (e.g., ./data)")
    parser.add_argument("--out", type=str, required=True, help="Path to the output directory (e.g., ./out)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)

    journals_path = os.path.join(args.data, "journals.jsonl")
    gold_path = os.path.join(args.data, "gold.jsonl")
    sample_predictions_path = os.path.join(args.data, "sample_predictions.jsonl")

    # Load all datasets
    journals_data = load_data_from_jsonl(journals_path)
    gold_data = load_data_from_jsonl(gold_path)
    predicted_raw_data = load_data_from_jsonl(sample_predictions_path)

    # Convert raw data into SemanticObject instances
    gold_objects_by_journal = load_semantic_objects(gold_data, key="items")
    predicted_objects_by_journal = load_semantic_objects(predicted_raw_data, key="items")

    scorer = Scorer()
    all_journal_scores = []
    per_journal_scores_output = []

    for journal_id, journal_entry_data in journals_data.items():
        journal_text = journal_entry_data["text"]
        gold_objs = gold_objects_by_journal.get(journal_id, [])
        predicted_objs = predicted_objects_by_journal.get(journal_id, [])

        score = scorer.score_journal(journal_text, gold_objs, predicted_objs)
        all_journal_scores.append(score)
        
        # Prepare per-journal output, excluding detailed matches for the JSONL file
        per_journal_output = {
            "journal_id": journal_id,
            "tp": score["tp"],
            "fp": score["fp"],
            "fn": score["fn"],
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "polarity_accuracy": score["polarity_accuracy"],
            "bucket_accuracy": score["bucket_accuracy"],
            "evidence_coverage_rate": score["evidence_coverage_rate"],
        }
        per_journal_scores_output.append(per_journal_output)

    # Calculate overall scores
    overall_summary = scorer.overall_scores(all_journal_scores)

    # Write outputs
    with open(os.path.join(args.out, "score_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=4)

    with open(os.path.join(args.out, "per_journal_scores.jsonl"), 'w', encoding='utf-8') as f:
        for entry in per_journal_scores_output:
            f.write(json.dumps(entry) + '\n')

    print("Evaluation complete. Results saved to {args.out}/")

if __name__ == "__main__":
    main()
