from unsloth import FastLanguageModel

from src.config import Config
from src.logger import get_logger
from src.retrieval.vector_store import SchemaRetriever
from src.schema import AETHERIS_DB, TableSchema

logger = get_logger(__name__)


class AuraInference:
    """End-to-end inference pipeline: RAG + Fine-tuned LLM."""

    def __init__(self, model_path: str) -> None:
        """Initialize the model and the schema retriever."""
        # Load the model for inference
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        # Initialize RAG component
        self.retriever = SchemaRetriever(AETHERIS_DB)

    def _format_context(self, tables: list[TableSchema]) -> str:
        """Formats the retrieved tables into a clean context string."""
        context_parts: list[str] = []
        for table in tables:
            cols = [{"name": c.name, "description": c.description} for c in table.columns]
            part = f"Table '{table.name}': {table.description}. Columns: {cols}"
            context_parts.append(part)
        return "\n".join(context_parts)

    def predict(self, nl_query: str) -> str:
        """Executes the full Text-to-DSL pipeline."""
        # Semantic Search (RAG)
        relevant_tables = self.retriever.get_relevant_tables(nl_query, top_k=2)

        if not relevant_tables:
            logger.warning("No tables found for query: %s", nl_query)
            return "ERROR: Context not found"

        context = self._format_context(relevant_tables)

        # Format prompt using the centralized style
        full_prompt = Config.PROMPT_STYLE.format(context, nl_query, "")

        # LLM Generation
        inputs = self.tokenizer([full_prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract the assistant's response from the formatted block
        if "### Response:" in decoded:
            return decoded.split("### Response:")[1].strip()

        return decoded


def main() -> None:
    """Main execution entry point for inference tests."""
    model_path = str(Config.BASE_DIR / "models" / "phi-4-auradsl-20251223_0845")

    logger.info("Initializing inference engine with model at %s", model_path)
    engine = AuraInference(model_path)

    test_queries: list[str] = [
        "How much electricity did the fridge use today?",
        "Check all climate sensors in the kitchen and garage.",
        "List top 5 security logs with high severity from yesterday.",
        "Show me water leaks detected in the last 24 hours.",
    ]

    logger.info("Starting test queries execution...")

    for query in test_queries:
        logger.info("Processing NL: %s", query)
        try:
            dsl = engine.predict(query)
            logger.info("Resulting AuraDSL: %s", dsl)
        except Exception as e:
            logger.error("Failed to generate DSL for query '%s': %s", query, e)


if __name__ == "__main__":
    main()
