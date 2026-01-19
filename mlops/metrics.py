from typing import List, Dict, Optional
from rouge_score import rouge_scorer


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # (target, prediction) is the common convention
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }


def calculate_answer_length_metrics(answers: List[str]) -> Dict[str, float]:
    if not answers:
        return {"avg_length": 0.0, "min_length": 0.0, "max_length": 0.0}

    lengths = [len(a.split()) for a in answers]
    return {
        "avg_length": sum(lengths) / len(lengths),
        "min_length": float(min(lengths)),
        "max_length": float(max(lengths)),
    }


def calculate_source_metrics(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {"avg_sources": 0.0, "min_sources": 0.0, "max_sources": 0.0}

    nums = [r.get("num_sources", 0) for r in results if "error" not in r]
    if not nums:
        return {"avg_sources": 0.0, "min_sources": 0.0, "max_sources": 0.0}

    return {
        "avg_sources": sum(nums) / len(nums),
        "min_sources": float(min(nums)),
        "max_sources": float(max(nums)),
    }


def calculate_metrics(results: List[Dict], expected_answers: Optional[List[str]] = None) -> Dict[str, float]:
    if not results:
        return {"success_rate": 0.0, "overall_score": 0.0}

    successful = [r for r in results if "error" not in r]
    success_rate = len(successful) / len(results)

    # Robust extraction (supports "answer" or old "answers")
    answers = [(r.get("answer") or r.get("answers") or "").strip() for r in successful]
    answers = [a for a in answers if a]

    length_metrics = calculate_answer_length_metrics(answers)
    source_metrics = calculate_source_metrics(results)

    rouge_scores = {}
    # Only compute ROUGE when we can safely align
    if expected_answers and len(expected_answers) == len(results) and len(successful) == len(results):
        rouge_scores = calculate_rouge_scores(answers, expected_answers)

    # Overall score: success is always important
    overall_score = success_rate * 0.4

    if rouge_scores:
        overall_score += (
            rouge_scores.get("rouge1", 0.0) * 0.2 +
            rouge_scores.get("rouge2", 0.0) * 0.2 +
            rouge_scores.get("rougeL", 0.0) * 0.2
        )
    else:
        # Heuristic when no ground truth is provided
        avg_length = length_metrics["avg_length"]
        if 20 <= avg_length <= 200:
            overall_score += 0.6
        elif 10 <= avg_length <= 300:
            overall_score += 0.4
        else:
            overall_score += 0.2

    metrics = {
        "success_rate": success_rate,
        "overall_score": overall_score,
        **length_metrics,
        **source_metrics,
    }

    if rouge_scores:
        metrics.update(rouge_scores)

    return metrics
