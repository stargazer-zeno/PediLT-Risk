from __future__ import annotations

import argparse

from significance_common import (
    RESULTS_DIR,
    ensure_output_dirs,
    get_model_pairs,
    load_predictions,
    make_inventory,
    paired_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare inventories for paired model comparisons.")
    parser.add_argument("--pairs", choices=["primary", "all"], default="all")
    args = parser.parse_args()

    ensure_output_dirs()
    predictions = load_predictions()
    model_inventory = make_inventory(predictions)
    pairs = get_model_pairs(args.pairs)
    pair_inventory = paired_inventory(predictions, pairs)

    model_path = RESULTS_DIR / "model_prediction_inventory.csv"
    pair_path = RESULTS_DIR / "paired_prediction_inventory.csv"
    model_inventory.to_csv(model_path, index=False)
    pair_inventory.to_csv(pair_path, index=False)

    print(f"Wrote {model_path}")
    print(f"Wrote {pair_path}")


if __name__ == "__main__":
    main()
