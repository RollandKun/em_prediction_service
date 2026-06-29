-- em_prediction_service - Database initialization
-- Runs on first Docker Compose start (docker-entrypoint-initdb.d)

-- Grid data table (partitioned by month)
CREATE TABLE IF NOT EXISTS grid_data (
    id BIGSERIAL,
    datetime TIMESTAMPTZ NOT NULL,
    price NUMERIC(10,2),
    load NUMERIC(10,2),
    solar NUMERIC(10,2),
    wind NUMERIC(10,2),
    hydro NUMERIC(10,2),
    renewable_total NUMERIC(10,2),
    bidspace NUMERIC(10,2),
    reserve NUMERIC(10,2),
    nonmarket NUMERIC(10,2),
    tieline NUMERIC(10,2),
    load_tie NUMERIC(10,2),
    day_type VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, datetime)
) PARTITION BY RANGE (datetime);

-- Monthly partitions (2026-01 to 2026-12)
CREATE TABLE IF NOT EXISTS grid_data_2026_01 PARTITION OF grid_data
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_02 PARTITION OF grid_data
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_03 PARTITION OF grid_data
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_04 PARTITION OF grid_data
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_05 PARTITION OF grid_data
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_06 PARTITION OF grid_data
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_07 PARTITION OF grid_data
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_08 PARTITION OF grid_data
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_09 PARTITION OF grid_data
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_10 PARTITION OF grid_data
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_11 PARTITION OF grid_data
    FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
CREATE TABLE IF NOT EXISTS grid_data_2026_12 PARTITION OF grid_data
    FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_grid_dt ON grid_data(datetime);

-- Weather forecast (JSONB)
CREATE TABLE IF NOT EXISTS weather_forecast (
    id BIGSERIAL PRIMARY KEY,
    fetch_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    variables JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fetch_time, target_time)
);
CREATE INDEX IF NOT EXISTS idx_wf_target ON weather_forecast(target_time);

-- Weather observations (JSONB)
CREATE TABLE IF NOT EXISTS weather_obs (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMPTZ NOT NULL UNIQUE,
    variables JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_wo_dt ON weather_obs(datetime);

-- Model versions
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version_name VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(30) NOT NULL,
    file_path VARCHAR(500),
    metrics JSONB,
    status VARCHAR(20) DEFAULT 'shadow' CHECK (status IN ('active', 'shadow', 'archived')),
    feature_cache BYTEA,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    target_time TIMESTAMPTZ NOT NULL,
    predicted_price NUMERIC(10,2),
    actual_price NUMERIC(10,2),
    model_version VARCHAR(50),
    season VARCHAR(10),
    period INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_time);

-- Data quality log
CREATE TABLE IF NOT EXISTS data_quality_log (
    id BIGSERIAL PRIMARY KEY,
    check_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,
    completeness_pct NUMERIC(5,2),
    anomaly_count INTEGER DEFAULT 0,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Shadow predictions (A/B testing)
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    target_time TIMESTAMPTZ NOT NULL,
    predicted_price NUMERIC(10,2),
    model_version VARCHAR(50),
    season VARCHAR(10),
    period INTEGER
);
CREATE INDEX IF NOT EXISTS idx_shadow_version ON shadow_predictions(model_version, target_time);
