from typing import List, Dict, Optional

from rouge_score import rouge_scorer

def calculate_rouge_source(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer= rouge_scorer.RougeScorer(["rouge-1", "rouge-2", "rougeL"],use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rouge3_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(pred, ref)
        rouge1_scores.append(scores["rouge-1"].fmeasure)
        rouge2_scores.append(scores["rouge-2"].fmeasure)
        rouge3_scores.append(scores["rougeL"].fmeasure)
        
    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rouge3_scores) / len(rouge3_scores) if rouge3_scores else 0.0,
            }

def calculate_answer_length_metrics(answers: List[str]) -> Dict[str, float]:
    if not answers:
        return {
            "avg_length":0.0,
            "min_length":0.0,
            "max_length":0.0,
        }

    lengths = [len(answer.split()) for answer in answers]
    return {
        "avg_length":sum(lengths) / len(lengths),
        "min_length":min(lengths),
        "max_length":max(lengths),
    }

def calculate_source_metrics(results: List[Dict])-> Dict[str, float]:
    if not results:
        return {
            "avg_sources":0.0,
            "min_sources":0.0,
            "max_sources":0.0,
        }
    num_sources = [r.get("num_sources",0) for r in results if "error" not in r]
    if not num_sources:
        return {
            "avg_sources":0.0,
            "min_sources":0.0,
            "max_sources":0.0,
        }
    return {
        "avg_sources":sum(num_sources) / len(num_sources),
        "min_sources":min(num_sources),
        "max_sources":max(num_sources),
    }

def calculate_metrics(results: List[Dict], expected_answers: Optional[List[str]] = None) -> Dict[str, float]:
    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        return {
            "accuracy":0.0,
            "success_rate":0.0,
            "overall_score":0.0,
        }

    success_rate = len(successful_results) / len(results) if results else 0.0

    answers = [r["answer"] for r in successful_results]

    length_metrics = calculate_answer_length_metrics(answers)
    source_metrics = calculate_source_metrics(results)
    rouge_scores = {}
    if expected_answers and len(expected_answers) == len(answers):
        rouge_scores = calculate_rouge_source(answers, expected_answers)

    overall_score = success_rate * 0.4

    if rouge_scores:
        overall_score += (
            rouge_scores.get("rouge1", 0.0) *0.2 +
            rouge_scores.get("rouge2", 0.0) *0.2 +
            rouge_scores.get("rougeL", 0.0) *0.2
        )
    else:
        avg_length = length_metrics["avg_length"]
        if 20 <= avg_length <= 200:
            overall_score += 0.3
        elif 10 <= avg_length <= 300:
            overall_score += 0.2
        else:
            overall_score += 0.1

    metrics = {
        "success_rate": success_rate,
        "accuracy": success_rate,
        "overall_score": overall_score,
        **length_metrics,
        **source_metrics,
    }

    if rouge_scores:
        metrics.update(rouge_scores)
    return metrics

