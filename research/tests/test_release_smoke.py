from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.common.response_parser import parse_probability_json


class ReleaseSmokeTest(unittest.TestCase):
    def test_probability_contract(self) -> None:
        self.assertEqual(
            parse_probability_json('{"1m": 0.1, "1y": null, "5y": 0.4}'),
            {"1m": 0.1, "1y": None, "5y": 0.4},
        )
        self.assertIsNone(parse_probability_json("```json\n{\"1m\": 0.1}\n```"))
        self.assertIsNone(parse_probability_json('{"1m": null, "1y": null, "5y": null}'))

    def test_synthetic_shapes_and_no_real_identity(self) -> None:
        raw = json.loads((ROOT / "llm/examples/patient_raw_example.json").read_text(encoding="utf-8"))
        self.assertTrue(str(raw["id"]).startswith("SYNTHETIC_"))
        self.assertEqual(json.loads((ROOT / "machine_learning/examples/input_shapes.json").read_text(encoding="utf-8"))["static_shape"], [142])
        self.assertFalse((ROOT / "examples").exists())

    def test_public_metric_contract(self) -> None:
        forbidden_metric = "AUP" + "RC"
        for path in (ROOT / "evaluation" / "metrics").glob("*.csv"):
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn(forbidden_metric, header)
        from evaluation.statistical_significance.secondary.significance_common import METRICS

        self.assertEqual(METRICS, ("auroc", "brier"))


if __name__ == "__main__":
    unittest.main()
