import json
from typing import List, Dict, Tuple
from ashwam_types import SemanticObject # Assuming SemanticObject is in ashwam_types.py

def calculate_jaccard_similarity(span1: str, span2: str) -> float:
    """
    Calculates the Jaccard similarity between two text spans.
    """
    set1 = set(span1.lower().split())
    set2 = set(span2.lower().split())
    if not set1 and not set2:
        return 1.0  # Both empty, considered a perfect match
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

class Scorer:
    def __init__(self, jaccard_threshold: float = 0.5):
        self.jaccard_threshold = jaccard_threshold

    def _match_objects(self, gold_objects: List[SemanticObject], predicted_objects: List[SemanticObject]) -> Tuple[List[Tuple[SemanticObject, SemanticObject]], List[SemanticObject], List[SemanticObject]]:
        """
        Matches predicted objects to gold objects based on domain and evidence span overlap.
        Returns a list of matched pairs (TPs), unmatched predicted objects (FPs), and unmatched gold objects (FNs).
        """
        matched_pairs: List[Tuple[SemanticObject, SemanticObject]] = []
        unmatched_predicted = list(predicted_objects)
        unmatched_gold = list(gold_objects)

        # Create a copy of predicted objects to avoid modifying during iteration
        remaining_predicted_indices = list(range(len(unmatched_predicted)))
        remaining_gold_indices = list(range(len(unmatched_gold)))

        # Store potential matches with their scores
        potential_matches: List[Tuple[float, int, int]] = [] # (jaccard_score, pred_idx, gold_idx)

        for i, gold_obj in enumerate(unmatched_gold):
            for j, pred_obj in enumerate(unmatched_predicted):
                if gold_obj.domain == pred_obj.domain:
                    jaccard = calculate_jaccard_similarity(gold_obj.evidence_span, pred_obj.evidence_span)
                    if jaccard >= self.jaccard_threshold:
                        potential_matches.append((jaccard, j, i))
        
        # Sort potential matches by Jaccard score in descending order
        potential_matches.sort(key=lambda x: x[0], reverse=True)

        # Greedily match based on the highest Jaccard score
        matched_gold_indices = set()
        matched_predicted_indices = set()

        for jaccard, pred_idx, gold_idx in potential_matches:
            if pred_idx not in matched_predicted_indices and gold_idx not in matched_gold_indices:
                matched_pairs.append((unmatched_gold[gold_idx], unmatched_predicted[pred_idx]))
                matched_predicted_indices.add(pred_idx)
                matched_gold_indices.add(gold_idx)

        # Identify FPs and FNs
        fps = [pred_obj for i, pred_obj in enumerate(unmatched_predicted) if i not in matched_predicted_indices]
        fns = [gold_obj for i, gold_obj in enumerate(unmatched_gold) if i not in matched_gold_indices]

        return matched_pairs, fps, fns


    def score_journal(self, journal_text: str, gold_objects: List[SemanticObject], predicted_objects: List[SemanticObject]):
        matched_pairs, fps, fns = self._match_objects(gold_objects, predicted_objects)

        tp_count = len(matched_pairs)
        fp_count = len(fps)
        fn_count = len(fns)

        # Object-level Precision, Recall, F1
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Polarity Accuracy
        correct_polarity_count = 0
        for gold_obj, pred_obj in matched_pairs:
            if gold_obj.polarity == pred_obj.polarity:
                correct_polarity_count += 1
        polarity_accuracy = correct_polarity_count / tp_count if tp_count > 0 else 0.0

        # Bucket Accuracy
        correct_bucket_count = 0
        for gold_obj, pred_obj in matched_pairs:
            if gold_obj.domain == "emotion":
                if gold_obj.arousal_bucket == pred_obj.arousal_bucket:
                    correct_bucket_count += 1
            else:
                if gold_obj.intensity_bucket == pred_obj.intensity_bucket:
                    correct_bucket_count += 1
        bucket_accuracy = correct_bucket_count / tp_count if tp_count > 0 else 0.0

        # Evidence Coverage Rate
        valid_evidence_spans = 0
        for pred_obj in predicted_objects:
            if pred_obj.evidence_span in journal_text:
                valid_evidence_spans += 1
        evidence_coverage_rate = valid_evidence_spans / len(predicted_objects) if len(predicted_objects) > 0 else 0.0

        return {
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "polarity_accuracy": polarity_accuracy,
            "bucket_accuracy": bucket_accuracy,
            "evidence_coverage_rate": evidence_coverage_rate,
            "matched_pairs": [(g.to_dict(), p.to_dict()) for g, p in matched_pairs],
            "false_positives": [fp.to_dict() for fp in fps],
            "false_negatives": [fn.to_dict() for fn in fns],
        }
    
    @staticmethod
    def overall_scores(all_journal_scores: List[Dict]) -> Dict:
        total_tp = sum(s["tp"] for s in all_journal_scores)
        total_fp = sum(s["fp"] for s in all_journal_scores)
        total_fn = sum(s["fn"] for s in all_journal_scores)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        # For accuracies, average the per-journal accuracies (this assumes each journal contributes equally)
        # A more robust approach might be to sum correct predictions across all TPs and divide by total TPs
        total_polarity_correct = sum(s["polarity_accuracy"] * s["tp"] for s in all_journal_scores)
        overall_polarity_accuracy = total_polarity_correct / total_tp if total_tp > 0 else 0.0

        total_bucket_correct = sum(s["bucket_accuracy"] * s["tp"] for s in all_journal_scores)
        overall_bucket_accuracy = total_bucket_correct / total_tp if total_tp > 0 else 0.0

        total_valid_evidence_spans = sum(s["evidence_coverage_rate"] * (s["tp"] + s["fp"]) for s in all_journal_scores) # tp + fp is total predicted
        total_predicted_objects = sum(s["tp"] + s["fp"] for s in all_journal_scores)
        overall_evidence_coverage_rate = total_valid_evidence_spans / total_predicted_objects if total_predicted_objects > 0 else 0.0

        return {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_polarity_accuracy": overall_polarity_accuracy,
            "overall_bucket_accuracy": overall_bucket_accuracy,
            "overall_evidence_coverage_rate": overall_evidence_coverage_rate,
        }

