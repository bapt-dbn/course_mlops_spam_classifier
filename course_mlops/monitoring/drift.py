import logging

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from course_mlops.monitoring.enums import MonitoredColumn
from course_mlops.monitoring.exceptions import DriftDetectionError
from course_mlops.monitoring.schemas import ColumnDrift
from course_mlops.monitoring.schemas import DriftResult

logger = logging.getLogger(__name__)

_MONITORED_COLUMNS = list(MonitoredColumn)


class DriftDetector:
    def __init__(self, reference_df: pd.DataFrame) -> None:
        self._reference_df = reference_df[_MONITORED_COLUMNS]

    def detect(self, current_df: pd.DataFrame) -> DriftResult:
        try:
            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(reference_data=self._reference_df, current_data=current_df[_MONITORED_COLUMNS])
            result = snapshot.dict()
            metrics = result["metrics"]

            drift_share = metrics[0]["value"]["share"]
            drift_threshold = metrics[0]["config"].get("drift_share", 0.5)
            dataset_drift = drift_share >= drift_threshold

            column_drifts = {}
            for metric in metrics[1:]:
                config = metric["config"]
                col_name = config.get("column")
                if col_name is None:
                    continue
                column_drifts[col_name] = ColumnDrift(
                    drift_detected=metric["value"] < config.get("threshold", 0.05),
                    drift_score=metric["value"],
                    stattest_name=config.get("method", "unknown"),
                )

            return DriftResult(
                dataset_drift=dataset_drift,
                drift_share=drift_share,
                column_drifts=column_drifts,
            )
        except Exception as e:
            raise DriftDetectionError(f"Drift detection failed: {e}") from e
