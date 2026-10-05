"""
Backtest adapter for the production model.

Calls the real ``forecast_future_sales_direct`` on training data truncated at the
cutoff, with ``as_of_date`` set to the cutoff, so the code being scored is exactly the
code that runs in production. Series are processed in parallel, one per task.
"""

import logging
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from estimation.model_forecast import forecast_future_sales_direct

logger = logging.getLogger(__name__)


def _forecast_one(series_features: pd.DataFrame, horizon: int, cutoff: pd.Timestamp):
    """Run the production forecaster for a single series; return (mlb, values) or None."""
    logging.getLogger("estimation.model_forecast").setLevel(logging.ERROR)
    forecasts, _ = forecast_future_sales_direct(
        series_features, horizon, as_of_date=cutoff
    )
    if not forecasts:
        return None
    mlb, (forecast_df, _sku) = next(iter(forecasts.items()))
    return mlb, forecast_df["prediction"].astype(float)


def forecast_current_model(
    features: pd.DataFrame,
    mlbs: List[str],
    cutoff: pd.Timestamp,
    horizon: int,
    n_jobs: int = -1,
) -> Dict[str, pd.Series]:
    """
    Production-model forecasts for the given series.

    Args:
        features: Training features (output of ``build_training_features``).
        mlbs: Series to forecast.
        cutoff: Forecast origin; predictions cover cutoff + 1 .. cutoff + horizon.
        horizon: Forecast length in days.
        n_jobs: Parallel worker processes (-1 uses all cores).

    Returns:
        Dict mapping MLB to a forecast Series indexed by date. Series the production
        code skips (inactive or too little data) are absent.
    """
    tasks = [features[features["mlb"] == mlb] for mlb in mlbs]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_forecast_one)(task, horizon, cutoff) for task in tasks
        )
    forecasts = dict(r for r in results if r is not None)
    skipped = len(mlbs) - len(forecasts)
    if skipped:
        logger.warning(f"Production model skipped {skipped} of {len(mlbs)} series")
    return forecasts
