
from typing import List, Dict, Optional
from rouge_score import rouge_scorer


def rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1) / len(r1) if r1 else 0.0,
        "rouge2": sum(r2) / len(r2) if r2 else 0.0,
        "rougeL": sum(rL) / len(rL) if rL else 0.0,
    }


def length_metrics(answers: List[str]) -> Dict[str, float]:
    if not answers:
        return {"avg_len": 0.0, "min_len": 0.0, "max_len": 0.0}
    lens = [len(a.split()) for a in answers]
    return {"avg_len": sum(lens) / len(lens), "min_len": float(min(lens)), "max_len": float(max(lens))}


def source_metrics(results: List[Dict]) -> Dict[str, float]:
    nums = [r.get("num_sources", 0) for r in results if "error" not in r]
    if not nums:
        return {"avg_sources": 0.0, "min_sources": 0.0, "max_sources": 0.0}
    return {"avg_sources": sum(nums) / len(nums), "min_sources": float(min(nums)), "max_sources": float(max(nums))}


def calculate_metrics(results: List[Dict], expected_answers: Optional[List[str]] = None) -> Dict[str, float]:
    if not results:
        return {"success_rate": 0.0, "overall_score": 0.0}

    ok = [r for r in results if "error" not in r]
    success_rate = len(ok) / len(results)

    answers = [(r.get("answer") or "").strip() for r in ok]
    answers = [a for a in answers if a]

    lm = length_metrics(answers)
    sm = source_metrics(results)

    r = {}
    if expected_answers and len(expected_answers) == len(results) and len(ok) == len(results):
        r = rouge(answers, expected_answers)

    overall = success_rate * 0.4
    if r:
        overall += r["rouge1"] * 0.2 + r["rouge2"] * 0.2 + r["rougeL"] * 0.2
    else:
        # heuristic if no ground truth
        if 20 <= lm["avg_len"] <= 200:
            overall += 0.6
        elif 10 <= lm["avg_len"] <= 300:
            overall += 0.4
        else:
            overall += 0.2

    out = {"success_rate": success_rate, "overall_score": overall, **lm, **sm}
    if r:
        out.update(r)
    return out
