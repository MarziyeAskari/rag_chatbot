# mlops/cli.py
import argparse
import json
from pathlib import Path

from mlops.train_rag import train
from mlops.evaluate_rag import evaluate
from mlops.finetune_embedding import finetune
from mlops.tune_optuna import tune


def main():
    p = argparse.ArgumentParser("rag-mlops")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--documents", type=str, default="./data/documents")
    p_train.add_argument("--clear", action="store_true")
    p_train.add_argument("--no-mlflow", action="store_true")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--questions", type=str, default="mlops/questions.json")
    p_eval.add_argument("--expected", type=str, default="")
    p_eval.add_argument("--no-mlflow", action="store_true")

    p_ft = sub.add_parser("finetune-emb")
    p_ft.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_ft.add_argument("--output-dir", type=str, default="./models/embedding_finetuned")
    p_ft.add_argument("--epochs", type=int, default=1)
    p_ft.add_argument("--batch-size", type=int, default=16)
    p_ft.add_argument("--max-sessions", type=int, default=200)
    p_ft.add_argument("--max-pairs", type=int, default=50)
    p_ft.add_argument("--no-mlflow", action="store_true")

    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--questions", type=str, default="mlops/questions.json")
    p_tune.add_argument("--expected", type=str, default="")
    p_tune.add_argument("--trials", type=int, default=20)
    p_tune.add_argument("--no-mlflow", action="store_true")

    args = p.parse_args()

    if args.cmd == "train":
        train(args.documents, clear_existing=args.clear, use_mlflow=not args.no_mlflow)

    elif args.cmd == "evaluate":
        q = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
        evaluate(q, expected_answers=expected, use_mlflow=not args.no_mlflow)

    elif args.cmd == "finetune-emb":
        finetune(
            base_model=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_sessions=args.max_sessions,
            max_pairs_per_session=args.max_pairs,
            use_mlflow=not args.no_mlflow,
        )

    elif args.cmd == "tune":
        q = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
        tune(q, expected_answers=expected, n_trials=args.trials, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
