from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "evaluation" / "calibration"
sys.path.insert(0, str(MODULE))

try:
    from calibration_common import EPS, fit_platt

    CALIBRATION_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    EPS = 1e-6
    fit_platt = None
    CALIBRATION_DEPENDENCIES_AVAILABLE = False


class CalibrationReleaseTest(unittest.TestCase):
    def test_platt_parameters_are_well_formed(self) -> None:
        parameters = json.loads((MODULE / "parameters" / "platt_parameters.json").read_text(encoding="utf-8"))
        self.assertEqual(set(parameters["targets"]), {"1m", "1y", "5y"})
        self.assertNotIn("threshold", json.dumps(parameters).lower())
        for record in parameters["targets"].values():
            self.assertTrue(np.isfinite(record["intercept"]))
            self.assertGreater(record["slope"], 0)

    @unittest.skipUnless(CALIBRATION_DEPENDENCIES_AVAILABLE, "Install research/requirements.txt to run calibration tests")
    def test_fit_platt_uses_a_bounded_probability_scale(self) -> None:
        parameters, values = fit_platt(
            np.asarray([0, 0, 0, 1, 1, 1], dtype=int),
            np.asarray([0.01, 0.05, 0.20, 0.70, 0.90, 0.99], dtype=float),
        )
        self.assertEqual(parameters["probability_clip"], [EPS, 1 - EPS])
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_public_module_has_no_private_artifacts(self) -> None:
        forbidden_suffixes = {".joblib", ".log", ".docx", ".pdf", ".xlsx", ".zip"}
        self.assertFalse(any(path.suffix.lower() in forbidden_suffixes for path in MODULE.rglob("*") if path.is_file()))
        for table in (MODULE / "results" / "tables").glob("*.csv"):
            header = table.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("Patient_ID", header)
            self.assertNotIn("Sample_ID", header)
        text_files = [path for path in MODULE.rglob("*") if path.is_file() and path.suffix in {".md", ".py", ".json", ".csv"}]
        text = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
        self.assertNotIn("/root/nas", text)
        self.assertNotIn("D:\\pxc", text)

    @unittest.skipUnless(CALIBRATION_DEPENDENCIES_AVAILABLE, "Install research/requirements.txt to run calibration tests")
    def test_synthetic_end_to_end_run_and_verification(self) -> None:
        rows: list[dict[str, object]] = []
        event_sets = {
            "1m": {0, 2, 4, 6, 8, 10},
            "1y": {0, 3, 4, 7, 8, 11},
            "5y": {1, 3, 5, 7, 9, 11},
        }
        for patient_index in range(12):
            for target, events in event_sets.items():
                label = int(patient_index in events)
                rows.append(
                    {
                        "Patient_ID": f"SYNTHETIC_{patient_index:03d}",
                        "Sample_ID": f"SYNTHETIC_{target}_{patient_index:03d}",
                        "Target": target,
                        "True_Label": label,
                        "Pred_Prob": 0.80 - patient_index * 0.01 if label else 0.10 + patient_index * 0.01,
                        "Stage": "Synthetic",
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary)
            predictions = private / "synthetic_predictions.csv"
            pd.DataFrame(rows).to_csv(predictions, index=False)
            manifest = private / "calibration.private.json"
            manifest.write_text(
                json.dumps(
                    {
                        "calibration_source_predictions": predictions.name,
                        "retained_evaluation_predictions": predictions.name,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = private / "output"
            command = [
                sys.executable,
                str(MODULE / "run_calibration_analysis.py"),
                "--input-manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--calibration-patients",
                "6",
                "--split-candidates",
                "100",
                "--min-positive-patients",
                "1",
                "--bootstrap-n",
                "5",
                "--skip-model-verification",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            verification_run = subprocess.run(
                [
                    sys.executable,
                    str(MODULE / "verify_calibration_outputs.py"),
                    "--input-manifest",
                    str(manifest),
                    "--output-dir",
                    str(output_dir),
                    "--skip-model-verification",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if verification_run.returncode:
                self.fail(f"verification failed:\n{verification_run.stdout}\n{verification_run.stderr}")
            verification = json.loads((output_dir / "independent_verification_results.json").read_text(encoding="utf-8"))
            self.assertEqual(verification["status"], "PASS")
            self.assertFalse(any(output_dir.rglob("*.joblib")))


if __name__ == "__main__":
    unittest.main()
