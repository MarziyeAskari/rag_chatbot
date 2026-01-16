import argparse
import datetime
from pathlib import Path
from typing import List

import mlflow
from torch.utils.data import DataLoader

from mlops.mlflow_utils import MlflowTracker
from src.config_loader import get_setting
from src.database import DatabaseManager, MessageModel
from sentence_transformers import SentenceTransformer, InputExample, losses
import logging

logger = logging.getLogger(__name__)
settings = get_setting()

def build_conversation_pairs(
        db_manager: DatabaseManager,
        max_session: int,
        max_pairs_per_session: int, ) -> List[InputExample]:
    db = db_manager.get_session()
    try:
        sessions_rows = (
            db.query(MessageModel.session_id).distinct().limit(max_session).all()
        )
        sessions_ids = [row[0] for row in sessions_rows]
        examples: List[InputExample] = []
        for sessions_id in sessions_ids:
            messages = (db.query(MessageModel).filter(MessageModel.session_id == sessions_id).order_by(
                MessageModel.timestamp.asc()).all())
            pairs_add = 0
            for idx in range(len(messages) - 1):
                current_message = messages[idx]
                next_message = messages[idx + 1]
                if current_message.role == "user" and next_message.role == "assistant":
                    examples.append(InputExample(texts=[current_message.content, next_message.content]))
                    pairs_add += 1
                if pairs_add >= max_pairs_per_session:
                    break
        return examples
    finally:
        db.close()


def finetune_embedding(
        base_model: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        max_session: int,
        max_pairs_per_session: int,
        use_mlflow: bool = True,
        register_name: str = "conversation_embeddings", ):

    if not settings.use_database_session:
        logger.error("Database session are required for access conversation history.")
        return

    db_manager = DatabaseManager(
        database_url=settings.session_database_url or None,
        database_path=settings.session_database_path,
    )
    examples = build_conversation_pairs(
        db_manager=db_manager,
        max_session=max_session,
        max_pairs_per_session=max_pairs_per_session,
    )
    if len(examples) < 10:
        logger.error("Not enough conversation pairs to fine-tune (need at least 10).")
        return
    model = SentenceTransformer(base_model)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    mlflow_tracker = MlflowTracker(experiment_name="embedding_finetune") if use_mlflow else None
    run_name = f"embedd_finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with mlflow_tracker.start_run(run_name=run_name) if mlflow_tracker else nullcontext():
        logger.info(f"Starting embedding fine-tune on conversation history")
        logger.info(f"Pairs: {len(examples)}, base model: {base_model}")

        if mlflow_tracker:
            mlflow_tracker.log_params(
                {
                    "base_model": base_model,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "max_session": max_session,
                    "max_pairs_per_session": max_pairs_per_session,
                    "total_pairs": len(examples),
                }
            )
        warmup_steps = max(10, int(len(examples) * epochs * 0.1))
        model.fit(
            dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path= str(output_path),
            show_progress_bar=True,
        )
        logger.info(f"Finished embedding fine-tune on conversation history")
        logger.info(f"Save fine-tuned embedding to {output_path}")
        if mlflow_tracker:
            model.save(output_dir)
            mlflow_tracker.log_artifacts(str(output_path),artifact_path="embedding_model_files")


class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="finetune embedding model with conversation history")
    parser.add_argument("--base-model", type=str, required=True, help="Base sentence transformer model id")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--max-session", type=int, required=True, help="Limit session number to sample")
    parser.add_argument("--max-pairs-per-session", type=int, required=True, help="Limit pairs to session")
    parser.add_argument("--output_dir", type="str", help="Directory to save fine-tune model", default=None)
    parser.add_argument("--register-name", action="store", default="conversation_embeddings")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging", default=False)
    parser.add_argument(
        "--register-mlflow",
        type=str,
        default="conversation_embeddings",
        help="Register MLflow logging with MLflow logging",
    )
    args = parser.parse_args()
    settings = get_setting()

    finetune_embedding(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_session=args.max_session,
        max_pairs_per_session=args.max_pairs_per_session,
        use_mlflow=not args.no_mlflow,
        register_name=args.register_name,
    )