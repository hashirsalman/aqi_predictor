"""Run Phase 4 feature engineering and target construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.features.engineering import build_features
from aqi_predictor.features.feature_contract import (
    CANONICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_eligibility_table,
)
from aqi_predictor.features.leakage_checks import assert_canonical_features_are_safe
from aqi_predictor.features.target_generation import add_aqi_targets


def main() -> int:
    input_path = ROOT / "data" / "processed" / "karachi_open_meteo_hourly_backfill.csv"
    output_path = ROOT / "data" / "processed" / "karachi_features_targets.csv"
    metrics_dir = ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(input_path, parse_dates=["event_time_utc", "event_time_local"])
    features = build_features(observations)
    targets = add_aqi_targets(observations)
    dataset = features.merge(targets, on=["event_time_utc", "event_time_local"], validate="one_to_one")

    complete_feature_rows = dataset[list(CANONICAL_FEATURE_COLUMNS)].notna().all(axis=1)
    complete_target_rows = dataset[list(TARGET_COLUMNS)].notna().all(axis=1)
    supervised = dataset.loc[complete_feature_rows & complete_target_rows].reset_index(drop=True)
    supervised.to_csv(output_path, index=False)

    eligibility = build_feature_eligibility_table()
    assert_canonical_features_are_safe()

    report = {
        "feature_schema_version": 1,
        "input_rows": int(len(observations)),
        "engineered_rows": int(len(dataset)),
        "supervised_rows_after_history_and_target_drop": int(len(supervised)),
        "dropped_initial_rows_for_lags_rolls": int((~complete_feature_rows).sum()),
        "dropped_final_rows_for_targets": int((~complete_target_rows).sum()),
        "feature_count": len(CANONICAL_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "first_supervised_event_time_utc": supervised["event_time_utc"].iloc[0].isoformat(),
        "last_supervised_event_time_utc": supervised["event_time_utc"].iloc[-1].isoformat(),
        "leakage_policy": "Canonical features contain no target/future/forecast columns; lags use positive shifts and rolling windows are backward-looking.",
        "feature_eligibility_table": eligibility,
        "local_artifact": str(output_path.relative_to(ROOT)),
        "local_artifact_role": "development/staging only; not the production Feature Store",
    }
    report_path = metrics_dir / "feature_target_build_report.json"
    eligibility_path = metrics_dir / "feature_eligibility_table.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    eligibility_path.write_text(json.dumps(eligibility, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
