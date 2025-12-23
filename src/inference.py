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
        logger.info("Loading model for inference from: %s", model_path)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.retriever = SchemaRetriever(AETHERIS_DB)

    def _format_context(self, tables: list[TableSchema]) -> str:
        """Formats the retrieved tables into a clean context string."""
        context_parts: list[str] = []
        for table in tables:
            cols = [{"name": c.name, "description": c.description} for c in table.columns]
            part = f"Table '{table.name}': {table.description}. Columns: {cols}"
            context_parts.append(part)
        return "\n".join(context_parts)

    def get_full_context_and_prompt(self, nl_query: str) -> tuple[str, str]:
        """Returns the formatted context and the final prompt for debugging."""
        relevant_tables = self.retriever.get_relevant_tables(nl_query, top_k=2)
        context = self._format_context(relevant_tables)
        full_prompt = Config.PROMPT_STYLE.format(context, nl_query, "")
        return context, full_prompt

    def predict(self, nl_query: str) -> str:
        """Executes the full Text-to-DSL pipeline."""
        _, full_prompt = self.get_full_context_and_prompt(nl_query)

        inputs = self.tokenizer([full_prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if "### Response:" in decoded:
            return decoded.split("### Response:")[1].strip()

        return decoded
