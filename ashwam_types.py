from typing import List, Dict, Literal, Union

class SemanticObject:
    def __init__(self, domain: Literal["symptom", "food", "emotion", "mind"],
                 evidence_span: str,
                 polarity: Literal["present", "absent", "uncertain"],
                 intensity_bucket: Literal["low", "medium", "high", "unknown"] = "unknown",
                 arousal_bucket: Literal["low", "medium", "high", "unknown"] = "unknown",
                 time_bucket: Literal["today", "last_night", "past_week", "unknown"] = "unknown"):
        self.domain = domain
        self.evidence_span = evidence_span
        self.polarity = polarity
        self.intensity_bucket = intensity_bucket
        self.arousal_bucket = arousal_bucket
        self.time_bucket = time_bucket

    def to_dict(self):
        obj_dict = {
            "domain": self.domain,
            "evidence_span": self.evidence_span,
            "polarity": self.polarity,
            "time_bucket": self.time_bucket,
        }
        if self.domain == "emotion":
            obj_dict["arousal_bucket"] = self.arousal_bucket
        else:
            obj_dict["intensity_bucket"] = self.intensity_bucket
        return obj_dict

