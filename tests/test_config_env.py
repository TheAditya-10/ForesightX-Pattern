from __future__ import annotations

import os

from foresightx_pattern.ml.utils.config import load_yaml_config


def test_load_yaml_config_respects_env_override(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("artifacts:\n  root_dir: artifacts\n  raw_data_path: artifacts/raw\n  processed_data_path: artifacts/processed\n  features_path: artifacts/features\n  reports_path: artifacts/reports\n  mlruns_path: artifacts/mlruns\ndata:\n  tickers: [TCS.NS]\nfeatures: {}\nmodel: {}\ntraining: {}\ntracking: {}\nservice: {}\ncache: {}\n", encoding="utf-8")
    old = os.environ.get("FORESIGHTX_CONFIG_PATH")
    os.environ["FORESIGHTX_CONFIG_PATH"] = str(config_path)
    try:
        loaded = load_yaml_config(None)
        assert loaded["data"]["tickers"] == ["TCS.NS"]
        assert loaded["artifacts"]["root_dir"] == "artifacts"
    finally:
        if old is None:
            os.environ.pop("FORESIGHTX_CONFIG_PATH", None)
        else:
            os.environ["FORESIGHTX_CONFIG_PATH"] = old
