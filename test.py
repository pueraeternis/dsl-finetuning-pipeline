import torch

from src.config import Config
from src.inference import AuraInference
from src.logger import get_logger

logger = get_logger(__name__)


def run_debug_test() -> None:
    """
    Runs a detailed inference test showing the RAG results,
    the full prompt, and the model's response.
    """
    model_path = str(Config.BASE_DIR / "models" / "phi-4-auradsl-20251223_0845")

    logger.info("Initializing AuraInference engine...")
    engine = AuraInference(model_path)

    test_query = "What is the average humidity in the Living Room from the last 24 hours?"

    logger.info("--- STARTING DEBUG TEST ---")
    logger.info("User Input: %s", test_query)

    context, full_prompt = engine.get_full_context_and_prompt(test_query)

    print("\n" + "=" * 30 + " RAG CONTEXT " + "=" * 30)
    print(context)
    print("=" * 73 + "\n")

    print("=" * 30 + " FULL PROMPT SENT TO MODEL " + "=" * 30)
    print(full_prompt)
    print("=" * 87 + "\n")

    logger.info("Generating response from Phi-4...")
    result_dsl = engine.predict(test_query)

    print("=" * 30 + " MODEL RESPONSE (DSL) " + "=" * 30)
    print(f"Result: {result_dsl}")
    print("=" * 83 + "\n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        run_debug_test()
    except Exception as e:
        logger.exception("Test failed: %s", e)
