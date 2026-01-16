import argparse
import datetime
import json
import logging
from pathlib import Path



from mlops.finetune_embedding import nullcontext
from mlops.metrics import calculate_metrics
from mlops.mlflow_utils import MlflowTracker
from src.config_loader import get_setting
from src.rag_chain import RagChain
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

def evaluate_rag(
        test_questions: list,
        expected_answers: list = None,
        use_mlflow: bool = True):
    settings = get_setting()
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MlflowTracker(experiment_name="rag_evaluation")

    run_name = f"evaluation_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow_tracker.start_run( run_name=run_name) if mlflow_tracker else nullcontext():
        logger.info("Initializing RAG system for evaluation.")

        vector_store = VectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.collection_name,
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
            api_key=settings.openai_api_key if settings.embedding_provider =="openai" else None,
        )

        size = vector_store.get_collection_size()
        if size == 0:
            logger.info("RAG system is empty, Please train the model first.")
            return

        logger.info(f"Vector store contains {size} documents.")

        rag_chain = RagChain(
            vector_store=vector_store,
            llm_provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key if settings.llm_provider =="openai" else None,

        )

        if mlflow_tracker:
            mlflow_tracker.log_params({
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "top_k": settings.top_k,
                "embedding_model": settings.embedding_model,
                "vector_store_size": size,
            } )

        results =[]
        for i, question in enumerate(test_questions):
            logger.info(f"Evaluating question {i+1} of {len(test_questions)}: {question}.")

            try:
                result = rag_chain.query(question, top_key=settings.top_k)
                results.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "num_sources": len(result["source_documents"]),
                        "sources": [
                            {
                                "content_preview": doc["content"][:100]+"...",
                                "metadata": doc["metadata"],
                            }
                            for doc in result["source_documents"]
                        ],
                    })
                logger.info(f"Answer: {result['answer'][:100]}...")
            except Exception as e:
                logger.error(f"Error evaluating question: {str(e)}")
                results.append({
                    "question": question,
                    "error": str(e),
                })
        metrics = calculate_metrics(results, expected_answers=expected_answers)

        if mlflow_tracker:
            mlflow_tracker.log_metrics(metrics)
            mlflow_tracker.log_model(
                rag_chain,
                artifact_path="rag_model"
            )

        output_file = Path("mlops/evaluation_output.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "metrics": metrics,
            },f, indent=2, ensure_ascii=False)

        if mlflow_tracker:
            mlflow_tracker.log_artifact(str(output_file))

        logger.info(f"Evaluation complete. Results saved to {output_file}.")
        logger.info(f"Metrics: {metrics}")

        successful = sum(1 for r in results if "error" not in r)
        logger.info(f"Successful answered: {successful}/{len(results)} questions.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot")
    parser.add_argument(
        "--no--mlflow",
        action="store_true",
        help="Disable Mlflow logging"
    )
    args = parser.parse_args()
    test_questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
    ]
    test_file = Path("mlops/evaluation_output.json")
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_questions = data.get("questions", test_questions)
    evaluate_rag(test_questions=test_questions, use_mlflow=not args.no_mlflow)