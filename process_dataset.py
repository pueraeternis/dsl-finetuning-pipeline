import json
import logging
from typing import Any

from src.config import Config
from src.data_gen.generate import SkeletonGenerator
from src.data_gen.prompt_factory import PromptFactory
from src.llm_client import LLMClient
from src.schema import AETHERIS_DB

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict[str, Any] | None:
    """
    Extracts JSON from a string that might contain markdown blocks.
    """
    try:
        cleaned_text = text.strip()
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()

        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s. Raw text: %s", e, text)
        return None


def process_single_item(
    generator: SkeletonGenerator,
    client: LLMClient,
    factory: PromptFactory,
) -> dict[str, Any] | None:
    """
    Workflow for a single skeleton-to-sample conversion with full type hints.
    """
    # Create a logical skeleton
    skeleton = generator.generate_skeleton()

    # Prepare prompts
    system_p = factory.get_system_prompt()
    user_p = factory.get_user_prompt(skeleton)

    # Request LLM
    raw_response = client.generate(system_p, user_p)
    if not raw_response:
        return None

    # Extract and format data
    data = extract_json(raw_response)
    if not data:
        return None

    return {
        "instruction": (
            "You are an expert in AuraDSL. Translate the natural language request into a valid AuraDSL query based on the provided schema."
        ),
        "context": {
            "table": skeleton["table_name"],
            "description": skeleton["table_description"],
            "columns": skeleton["columns_info"],
        },
        "nl_variants": data.get("nl_variants", []),
        "dsl": data.get("final_dsl", ""),
    }


def main() -> None:
    """Main execution loop for testing."""
    # Initialize from Config
    client = LLMClient(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_URL,
        model_name=Config.LLM_MODEL_NAME,
    )
    generator = SkeletonGenerator(AETHERIS_DB)
    factory = PromptFactory()

    print(f"--- STARTING LLM INFILLING (Model: {Config.LLM_MODEL_NAME}) ---")

    test_results = []
    for i in range(2):
        print(f"Generating sample {i + 1}...")
        sample = process_single_item(generator, client, factory)
        if sample:
            test_results.append(sample)
            print(f"Successfully generated DSL: {sample['dsl']}")

    # Save to data directory defined in Config
    output_path = Config.DATA_DIR / "dataset_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(test_results)} test samples to {output_path}")


if __name__ == "__main__":
    main()
