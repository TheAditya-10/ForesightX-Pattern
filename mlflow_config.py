from foresightx_pattern.ml.utils.config import load_settings
from foresightx_pattern.ml.utils.mlflow_utils import configure_mlflow


settings = load_settings()
configure_mlflow(settings)
