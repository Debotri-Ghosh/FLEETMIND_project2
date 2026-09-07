-- FleetMind schema
CREATE TABLE IF NOT EXISTS depots (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    load_hp      SMALLINT NOT NULL,
    nominal_rpm  SMALLINT
);

CREATE TABLE IF NOT EXISTS experiments (
    id                 SERIAL PRIMARY KEY,
    depot_id           INTEGER NOT NULL REFERENCES depots(id),
    fault_type         TEXT NOT NULL CHECK (fault_type IN ('Normal', 'InnerRace', 'OuterRace', 'Ball')),
    fault_diameter_in  NUMERIC(5,3),
    sensor_position    TEXT NOT NULL CHECK (sensor_position IN ('DE', 'FE', 'BA')),
    sampling_rate_hz   INTEGER NOT NULL DEFAULT 12000,
    source_file        TEXT NOT NULL,
    source_variable    TEXT NOT NULL,
    n_samples          INTEGER NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signal_windows (
    id             BIGSERIAL PRIMARY KEY,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    window_index   INTEGER NOT NULL,
    start_sample   INTEGER NOT NULL,
    end_sample     INTEGER NOT NULL,
    raw_values     DOUBLE PRECISION[] NOT NULL,
    label          TEXT NOT NULL,
    UNIQUE (experiment_id, window_index)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id               BIGSERIAL PRIMARY KEY,
    window_id        BIGINT NOT NULL REFERENCES signal_windows(id),
    model_version    TEXT NOT NULL,
    predicted_label  TEXT NOT NULL,
    confidence       REAL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id             BIGSERIAL PRIMARY KEY,
    prediction_id  BIGINT NOT NULL REFERENCES model_predictions(id),
    agent_name     TEXT NOT NULL,
    decision_text  TEXT NOT NULL,
    risk_score     REAL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiments_depot ON experiments(depot_id);
CREATE INDEX IF NOT EXISTS idx_windows_experiment ON signal_windows(experiment_id);
CREATE INDEX IF NOT EXISTS idx_windows_label ON signal_windows(label);
CREATE INDEX IF NOT EXISTS idx_predictions_window ON model_predictions(window_id);

INSERT INTO depots (name, load_hp, nominal_rpm) VALUES
    ('depot_0hp', 0, 1797),
    ('depot_1hp', 1, 1772),
    ('depot_2hp', 2, 1750),
    ('depot_3hp', 3, 1730)
ON CONFLICT (name) DO NOTHING;