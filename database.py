# em_prediction_service - Database (SQLAlchemy async + ORM models)
from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Numeric, Date, Boolean,
    DateTime, Text, ForeignKey, UniqueConstraint, Index, text
)
from sqlalchemy.dialects.postgresql import JSONB, BYTEA, TIMESTAMPTZ
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from config import settings


class Base(DeclarativeBase):
    pass


# ========== ORM Models (7 tables) ==========

class GridData(Base):
    """电网实况数据 — 15分钟粒度，宽表"""
    __tablename__ = "grid_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    datetime = Column(TIMESTAMPTZ, nullable=False, unique=True, index=True)
    price = Column(Numeric(10, 2))            # 出清价格 元/MWh
    load = Column(Numeric(10, 2))             # 省内负荷 MW
    solar = Column(Numeric(10, 2))            # 光伏出力 MW
    wind = Column(Numeric(10, 2))             # 风电出力 MW
    hydro = Column(Numeric(10, 2))            # 水电出力 MW
    renewable_total = Column(Numeric(10, 2))  # 新能源总出力 MW
    bidspace = Column(Numeric(10, 2))         # 竞价空间 MW
    reserve = Column(Numeric(10, 2))          # 系统备用 MW
    nonmarket = Column(Numeric(10, 2))        # 非市场机组 MW
    tieline = Column(Numeric(10, 2))          # 联络线 MW
    load_tie = Column(Numeric(10, 2))         # 负荷联络线 MW
    day_type = Column(String(20))             # 工作日/周末/节假日
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))

    __table_args__ = (
        Index("idx_grid_datetime", "datetime"),
    )


class WeatherForecast(Base):
    """气象预报数据 — JSONB 模式，一行=一个时间点的全部气象变量"""
    __tablename__ = "weather_forecast"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    fetch_time = Column(TIMESTAMPTZ, nullable=False)   # API 调用时间
    target_time = Column(TIMESTAMPTZ, nullable=False)  # 预报有效时间
    variables = Column(JSONB, nullable=False)          # {"temp_chengdu":22.5,...}
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("fetch_time", "target_time"),
        Index("idx_wf_target", "target_time"),
    )


class WeatherObs(Base):
    """气象实况数据 — JSONB 模式，用于训练"""
    __tablename__ = "weather_obs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    datetime = Column(TIMESTAMPTZ, nullable=False, unique=True)
    variables = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class ModelVersion(Base):
    """模型版本注册"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_name = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(30), nullable=False)    # stage1_solar / stage2_dry_valley / ...
    file_path = Column(String(500))
    metrics = Column(JSONB)                            # {"R2": 0.96, "MAE": 310}
    status = Column(String(20), default="shadow")       # active / shadow / archived
    feature_cache = Column(BYTEA)                      # 序列化的特征矩阵（增量推理用）
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Prediction(Base):
    """预测记录"""
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))
    target_time = Column(TIMESTAMPTZ, nullable=False)
    predicted_price = Column(Numeric(10, 2))
    actual_price = Column(Numeric(10, 2))
    model_version = Column(String(50))
    season = Column(String(10))
    period = Column(Integer)

    __table_args__ = (Index("idx_pred_target", "target_time"),)


class DataQualityLog(Base):
    """数据质量日志"""
    __tablename__ = "data_quality_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    check_date = Column(Date, nullable=False)
    status = Column(String(20), nullable=False)    # ok / warning / critical
    completeness_pct = Column(Numeric(5, 2))
    anomaly_count = Column(Integer, default=0)
    details = Column(JSONB)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class ShadowPrediction(Base):
    """Shadow 预测（A/B 测试用，与生产预测隔离）"""
    __tablename__ = "shadow_predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))
    target_time = Column(TIMESTAMPTZ, nullable=False)
    predicted_price = Column(Numeric(10, 2))
    model_version = Column(String(50))
    season = Column(String(10))
    period = Column(Integer)

    __table_args__ = (Index("idx_shadow_version", "model_version", "target_time"),)


# ========== Async Engine + Session ==========

engine = create_async_engine(settings.database_url, echo=False, pool_size=5, max_overflow=10)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session


async def init_db():
    """Create all tables (run on startup). Does NOT create partitions — see init_db.sql."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    await engine.dispose()
