# em_prediction_service - Configuration
import os
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://em_user:em_pass_2026@localhost:5432/em_prediction"
    database_url_sync: str = "postgresql+psycopg2://em_user:em_pass_2026@localhost:5432/em_prediction"

    # Data sources (historical import)
    grid_data_source: str = ""
    weather_data_source: str = ""
    base_table_source: str = ""

    # Grid API (real-time data from lingfeng-saas)
    grid_api_url: str = "https://lingfeng-saas.tradingthink.cn/api/data-analysis/intraProvincialSpotMarketData/PXNSC/marketSupport"
    grid_api_token: str = ""
    grid_api_trade_unit_id: int = 5101000
    # Grid API login credentials (encrypted, for auto token refresh)
    grid_api_username: str = ""
    grid_api_password: str = ""
    grid_api_platform: str = ""
    grid_login_url: str = "https://lingfeng-saas.tradingthink.cn/api/auth/login/v3"

    # Model
    model_dir: str = str(PROJECT_ROOT / "models")
    feature_version: str = "v14"

    # Training
    dry_season_months: tuple = (1, 2, 3, 4)
    wet_season_months: tuple = (5, 6)
    feature_dim: int = 177  # A-P groups: 12+14+10+20+4+14+6+4+1+3+19+16+2+18+16+18
    n_periods_per_day: int = 96

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
