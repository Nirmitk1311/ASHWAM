# Ashwam Exercise A: Evidence-Grounded Extraction & Evaluation

This project implements a mini evaluation pipeline for evidence-grounded extraction from Ashwam journal entries, as per the requirements of Exercise A.

## 1) Extraction Schema Design

We define a `SemanticObject` to represent extracted entities. This schema balances the need for objective evaluation with the inherent messiness and non-canonical nature of Ashwam journal data.

### SemanticObject Structure:

```json
{
  "domain": "symptom" | "food" | "emotion" | "mind",
  "evidence_span": "string" (exact quote from journal),
  "polarity": "present" | "absent" | "uncertain",
  "intensity_bucket": "low" | "medium" | "high" | "unknown" (for symptom, food, mind),
  "arousal_bucket": "low" | "medium" | "high" | "unknown" (for emotion),
  "time_bucket": "today" | "last_night" | "past_week" | "unknown"
}
```

### Field Constraints:

*   **Constrained Fields:** `domain`, `polarity`, `intensity_bucket`, `arousal_bucket`, `time_bucket` are strictly defined enums. This ensures consistency and facilitates objective evaluation.
*   **Free-Text Fields:** `evidence_span` is a free-text string, capturing the exact relevant phrase from the journal entry. This is crucial for evidence grounding and allows for the non-canonical nature of the journal content.

### Why this schema supports:

*   **Safety (no hallucinations):** The `evidence_span` field *forces* the extraction system to ground every extracted semantic object in the original text. If an entity cannot be directly evidenced in the text, it should not be extracted, or its `polarity` should be marked as `uncertain` if there\'s ambiguous textual evidence. This directly mitigates hallucination by requiring direct textual support.

*   **Evaluation (objective scoring):** Constrained fields like `domain`, `polarity`, and the `_bucket` fields allow for direct, objective comparison against gold labels. The `evidence_span` enables overlap-based matching, which is essential for non-canonical labels. We can compare the extracted `evidence_span` with the gold `evidence_span` for overlap, and then check for exact matches on the constrained fields. This allows us to score without relying on fixed vocabularies for the concepts themselves.

*   **Extensibility (future attributes):** The schema is flexible enough to accommodate additional attributes for `SemanticObject`s in the future. New constrained fields can be added (e.g., `severity_scale` for symptoms, `meal_type` for food) without altering the fundamental structure. Free-text fields could also be added for more nuanced observations if needed.

## 2) Proposed Extraction Approach

For this exercise, we will simulate the extraction process by loading the provided `sample_predictions.jsonl` dataset. In a real-world scenario, the extraction would likely follow an **LLM + Rules-based Pipeline** approach:

### Pipeline Steps:

1.  **Initial LLM Extraction:** A Large Language Model would process the journal entry, identifying semantic objects (symptoms, food, emotions, mind concepts) and extracting their `domain`, `evidence_span`, `polarity`, and relevant `_bucket` values based on a meticulously crafted prompt.

2.  **Post-processing and Validation (Rules-based):** A subsequent rules-based component would:
    *   **Evidence Span Validation:** Verify that each `evidence_span` is an *exact substring* of the original journal text.
    *   **Schema Conformance:** Ensure all extracted fields adhere to the predefined types and constrained enum values in the `SemanticObject` schema.
    *   **Domain-Specific Bucket Assignment:** Correctly assign `intensity_bucket` for `symptom`, `food`, `mind` domains and `arousal_bucket` for `emotion`.

### How Evidence Grounding is Enforced:

*   **Strict Prompting:** The LLM would be explicitly instructed to provide `evidence_span` values that are direct, unaltered quotes from the journal text.
*   **Post-Extraction Validation:** The rules-based post-processing acts as a critical safeguard, programmatically checking the validity of each `evidence_span` against the source text. Any `evidence_span` that is not a perfect substring would be flagged or the entire object potentially discarded or marked as uncertain.

### How Uncertainty / Abstention is Handled Safely:

*   **`polarity: "uncertain"`:** The LLM would be guided to use `"uncertain"` for `polarity` when the evidence in the journal text is ambiguous or vague.
*   **Abstention:** The system is designed to abstain from extracting an object entirely if there is no clear and direct textual evidence to support it. This is reinforced by the `evidence_span` validation: if no valid `evidence_span` can be found, no object will be extracted, preventing hallucinations.

## 3) Designed Evaluation Method

The evaluation method is designed to objectively score the extraction performance without relying on canonical labels for the extracted concepts themselves. Instead, it leverages evidence spans and constrained categorical fields.

### A) How Predicted Objects are Matched to Gold Objects

1.  **Primary Matching Criteria:** A predicted `SemanticObject` is considered a match for a gold `SemanticObject` if **both** of the following conditions are met:
    *   **Domain Match:** The `domain` field of the predicted object must be identical to the `domain` field of the gold object (e.g., both are `"symptom"`).
    *   **Evidence Span Overlap:** The `evidence_span` of the predicted object and the gold object must have a Jaccard similarity of \(\ge 0.5\). Jaccard similarity is calculated on the word sets of the lowercased evidence spans.

2.  **One-to-One Greedy Matching:** The matching process proceeds greedily. For each journal entry, we consider all possible predicted-gold object pairs that meet the primary matching criteria. These potential matches are then sorted by their Jaccard similarity (highest first). We then iterate through this sorted list, making matches. Once a predicted object or a gold object is part of a match, it cannot be matched again. This ensures that each gold object is matched by at most one predicted object, and vice-versa.

### B) What Counts as TP / FP / FN without Canonical Labels

*   **True Positive (TP):** A gold object is counted as a True Positive if it is successfully matched with a predicted object based on the defined matching criteria.
*   **False Positive (FP):** A predicted object is counted as a False Positive if it does not find any matching gold object.
*   **False Negative (FN):** A gold object is counted as a False Negative if it does not find any matching predicted object.

### C) How Scoring Works for Polarity and Bucket Attributes

These attribute scores are computed *only for True Positive (matched) object pairs*.

*   **Polarity Accuracy:** For each TP pair, we compare the `polarity` values of the gold and predicted objects. If they are identical, it's considered a correct polarity prediction for that object. The polarity accuracy is the percentage of TP pairs with correct polarity.

*   **Bucket Accuracy (Intensity/Arousal/Time):** Similar to polarity, for each TP pair, we compare the relevant bucket attribute. This means `intensity_bucket` for `symptom`, `food`, and `mind` domains, and `arousal_bucket` for the `emotion` domain, along with `time_bucket` for all domains. An exact string match is required for a correct bucket prediction. The bucket accuracy is the percentage of TP pairs with correct bucket attributes.
    *   **Clearly Defined Rule for Bucket Accuracy:** An exact string comparison is used. If a bucket is `"unknown"` in both gold and predicted, it counts as a match. If it\'s `"unknown"` in one and a specific value in the other (and vice-versa), it\'s a mismatch.

### D) Evidence Coverage Rate

This metric measures the proportion of predicted objects whose `evidence_span` is a valid substring within the original journal text. This is a direct measure of the system\'s ability to avoid hallucinations.

*   **Calculation:**
    \[
    \\text{Evidence Coverage Rate} = \\frac{\\text{Number of Predicted Objects with Valid Evidence Spans}}{\\text{Total Number of Predicted Objects}}
    \]
    A "valid evidence span" means the `evidence_span` string exists verbatim in the journal entry.
