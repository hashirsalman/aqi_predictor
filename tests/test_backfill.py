"""Tests for historical backfill helpers."""

from __future__ import annotations

from datetime import date
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.data.backfill import build_backfill_quality_report, build_date_chunks


class BackfillHelperTest(unittest.TestCase):
    def test_build_date_chunks_covers_inclusive_range_without_overlap(self) -> None:
        chunks = build_date_chunks(date(2026, 1, 1), date(2026, 1, 10), chunk_days=4)

        self.assertEqual(
            [(chunk.start_date, chunk.end_date) for chunk in chunks],
            [
                (date(2026, 1, 1), date(2026, 1, 4)),
                (date(2026, 1, 5), date(2026, 1, 8)),
                (date(2026, 1, 9), date(2026, 1, 10)),
            ],
        )

    def test_build_date_chunks_rejects_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            build_date_chunks(date(2026, 1, 2), date(2026, 1, 1))

    def test_quality_report_marks_local_artifact_as_non_production(self) -> None:
        frame = pd.DataFrame(
            {
                "event_time_utc": pd.to_datetime(["2026-01-01T00:00Z"]),
                "event_time_local": pd.to_datetime(["2026-01-01T05:00"]).tz_localize(
                    "Asia/Karachi"
                ),
                "us_aqi": [100.0],
            }
        )
        validation_summary = {
            "start_utc": "2026-01-01T00:00:00+00:00",
            "end_utc": "2026-01-01T00:00:00+00:00",
            "start_local": "2026-01-01T05:00:00+05:00",
            "end_local": "2026-01-01T05:00:00+05:00",
            "missing_percent": {"us_aqi": 0.0},
        }

        report = build_backfill_quality_report(
            frame,
            validation_summary,
            chunk_results=[],
            output_path=ROOT / "data" / "processed" / "sample.csv",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 1),
        )

        self.assertIn("not the production Feature Store", report["local_artifact_role"])
        self.assertEqual(report["numeric_distribution"]["us_aqi"]["median"], 100.0)


if __name__ == "__main__":
    unittest.main()
