# mlops/finetune_embedding.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Dict

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from mlops.common import load_settings, safe_run_ctx, ensure_dir
from mlops.mlflow_utils import MlflowTracker
from src.database import DatabaseManager, MessageModel

logger = logging.getLogger(__name__)


def build_pairs(db: DatabaseManager, max_sessions: int, max_pairs_per_session: int) -> List[InputExample]:
    sess = db.get_session()
    try:
        rows = sess.query(MessageModel.session_id).distinct().limit(max_sessions).all()
        session_ids = [r[0] for r in rows]

        examples: List[InputExample] = []
        for sid in session_ids:
            msgs = (
                sess.query(MessageModel)
                .filter(MessageModel.session_id == sid)
                .order_by(MessageModel.timestamp.asc())
                .all()
            )

            added = 0
            for i in range(len(msgs) - 1):
                cur, nxt = msgs[i], msgs[i + 1]
                # strict user -> assistant pairing
                if cur.role == "user" and nxt.role == "assistant":
                    examples.append(InputExample(texts=[cur.content, nxt.content]))
                    added += 1
                if added >= max_pairs_per_session:
                    break

        return examples
    finally:
        sess.close()


def finetune(
    *,
    base_model: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    max_sessions: int,
    max_pairs_per_session: int,
    use_mlflow: bool = True,
) -> None:
    s = load_settings()
    if not s.use_database_session:
        raise RuntimeError("use_database_session must be True to finetune from conversation DB.")

    db = DatabaseManager(database_url=s.session_database_url or None, database_path=s.session_database_path)
    pairs = build_pairs(db, max_sessions=max_sessions, max_pairs_per_session=max_pairs_per_session)

    if len(pairs) < 10:
        raise RuntimeError("Not enough conversation pairs (need at least 10).")

    tracker = MlflowTracker("embedding_finetune") if use_mlflow else None
    ensure_dir(output_dir)

    with safe_run_ctx(tracker, "finetune_embedding"):
        if tracker:
            tracker.log_params({
                "base_model": base_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_sessions": max_sessions,
                "max_pairs_per_session": max_pairs_per_session,
                "num_pairs": len(pairs),
            })

        model = SentenceTransformer(base_model)
        dl = DataLoader(pairs, shuffle=True, batch_size=batch_size)
        loss_fn = losses.MultipleNegativesRankingLoss(model)

        warmup_steps = max(10, int(len(pairs) * epochs * 0.1))
        model.fit(train_objectives=[(dl, loss_fn)], epochs=epochs, warmup_steps=warmup_steps, show_progress_bar=True)

        model.save(output_dir)
        logger.info(f"Saved finetuned embedding model to {output_dir}")

        if tracker:
            tracker.log_artifacts(output_dir, artifact_path="embedding_model")
