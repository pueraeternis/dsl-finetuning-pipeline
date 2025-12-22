import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from tqdm.asyncio import tqdm

from src.config import Config
from src.data_gen.generate import SkeletonGenerator
from src.data_gen.prompt_factory import PromptFactory
from src.schema import AETHERIS_DB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MassGenerator:
    def __init__(self, concurrency: int = 10):
        self.generator = SkeletonGenerator(AETHERIS_DB)
        self.factory = PromptFactory()
        self.concurrency = concurrency
        self.results: list[dict[str, Any]] = []

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extracts JSON from text, handling potential Markdown or noise."""
        try:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return None
        except (json.JSONDecodeError, ValueError):
            return None

    async def fetch_sample(self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, pbar: tqdm) -> None:
        async with semaphore:
            skeleton = self.generator.generate_skeleton()
            payload = {
                "model": Config.LLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": self.factory.get_system_prompt()},
                    {"role": "user", "content": self.factory.get_user_prompt(skeleton)},
                ],
                "temperature": 0.8,
                "max_tokens": 1024,
            }

            try:
                response = await client.post(
                    f"{Config.OPENAI_API_URL}/chat/completions",
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                data = self._extract_json(content)

                if data and "nl_variants" in data and "final_dsl" in data:
                    self.results.append(
                        {
                            "table": skeleton["table_name"],
                            "context": skeleton,
                            "nl_variants": data["nl_variants"],
                            "dsl": data["final_dsl"],
                        },
                    )
                else:
                    tqdm.write(f" [!] JSON mismatch in table: {skeleton['table_name']}")

            except Exception as e:
                tqdm.write(f" [X] Request failed: {type(e).__name__}")
            finally:
                pbar.update(1)

    async def run(self, total_skeletons: int):
        semaphore = asyncio.Semaphore(self.concurrency)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        async with httpx.AsyncClient() as client:
            with tqdm(total=total_skeletons, desc="Generating Dataset", unit="skel") as pbar:
                tasks = [self.fetch_sample(client, semaphore, pbar) for _ in range(total_skeletons)]
                await asyncio.gather(*tasks)

    def save_raw(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)


def flatten_dataset(raw_data_path: Path, output_path: Path):
    """
    Transforms nested LLM responses into flat training pairs.
    Removed redundant 'instruction' field as it should be handled by the trainer.
    """
    if not raw_data_path.exists():
        return

    with open(raw_data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    flattened = []
    for item in raw_data:
        for nl in item["nl_variants"]:
            clean_nl = re.sub(r"^(Casual|Formal|Indirect|Variant \d|Short|Detailed|Question):\s*", "", nl, flags=re.IGNORECASE)

            flattened.append(
                {
                    "input": clean_nl.strip(),
                    "context": (
                        f"Table '{item['table']}': {item['context']['table_description']}. Columns: {item['context']['columns_info']}"
                    ),
                    "output": item["dsl"],
                },
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flattened, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Dataset cleaned and flattened. Final size: {len(flattened)} samples.")


async def main():
    Config.ensure_dirs()
    total_skeletons = 3340

    mg = MassGenerator(concurrency=50)

    await mg.run(total_skeletons)

    raw_path = Config.DATA_DIR / "dataset_raw.json"
    mg.save_raw(raw_path)

    final_path = Config.DATA_DIR / "dataset_final.json"
    flatten_dataset(raw_path, final_path)


if __name__ == "__main__":
    asyncio.run(main())
