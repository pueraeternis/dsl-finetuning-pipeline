import json

from tqdm import tqdm

from src.config import Config
from src.engine.db import DBManager
from src.engine.transpiler import AuraTranspiler
from src.inference import AuraInference
from src.schema import AETHERIS_DB

MODEL_PATH = str(Config.BASE_DIR / "models" / "phi-4-auradsl-20251223_0845")
DATASET_PATH: str = "data/dataset_test.json"


class DSLValidator:
    def __init__(self, model_path: str):
        self.inference = AuraInference(model_path)
        self.transpiler = AuraTranspiler(AETHERIS_DB)
        self.db = DBManager(AETHERIS_DB)

    def compare_results(self, expected_dsl: str, predicted_dsl: str) -> bool:
        """Executes both queries and compares resulting data sets."""
        try:
            sql_exp, params_exp = self.transpiler.translate(expected_dsl)
            sql_pred, params_pred = self.transpiler.translate(predicted_dsl)

            res_exp = self.db.execute_query(sql_exp, params_exp)
            res_pred = self.db.execute_query(sql_pred, params_pred)

            return res_exp == res_pred
        except Exception:
            return False

    def get_component_score(self, expected: str, predicted: str) -> float:
        """Simple token-based similarity for components."""
        exp_set = set(expected.replace("|>", "").split())
        pred_set = set(predicted.replace("|>", "").split())
        if not exp_set:
            return 0.0
        return len(exp_set.intersection(pred_set)) / len(exp_set)


def run_evaluation(test_data_path: str, model_path: str):
    with open(test_data_path) as f:
        test_data = json.load(f)

    validator = DSLValidator(model_path)

    exec_matches = 0
    total_comp_score = 0.0

    results = []

    for item in tqdm(test_data[:100], desc="Evaluating"):
        expected_dsl = item["output"]
        predicted_dsl = validator.inference.predict(item["input"])

        # Execution Match
        is_match = validator.compare_results(expected_dsl, predicted_dsl)
        if is_match:
            exec_matches += 1

        # Component Score
        score = validator.get_component_score(expected_dsl, predicted_dsl)
        total_comp_score += score

        results.append(
            {
                "input": item["input"],
                "expected": expected_dsl,
                "predicted": predicted_dsl,
                "exec_match": is_match,
                "comp_score": score,
            },
        )

    print("\n--- EVALUATION RESULTS ---")
    print(f"Execution Accuracy: {exec_matches / 100:.2%}")
    print(f"Average Component Match: {total_comp_score / 100:.2%}")


if __name__ == "__main__":
    run_evaluation(DATASET_PATH, MODEL_PATH)
