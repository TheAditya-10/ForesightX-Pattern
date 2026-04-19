from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from foresightx_pattern.ml.utils.config import AppSettings, load_settings


DataProvider = Callable[[str, AppSettings, pd.Timestamp | None], pd.DataFrame]


@dataclass(slots=True)
class ServiceContainer:
    settings: AppSettings
    data_provider: DataProvider


def get_container(settings: AppSettings | None = None, data_provider: DataProvider | None = None) -> ServiceContainer:
    loaded = settings or load_settings()
    if data_provider is None:
        from foresightx_pattern.app.services.feature_service import download_latest_market_data

        data_provider = download_latest_market_data
    return ServiceContainer(settings=loaded, data_provider=data_provider)
