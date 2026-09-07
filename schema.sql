-- FleetMind schema
-- Design rationale (read this before you run it):
--
-- CWRU's raw files are NOT one-row-per-reading tables — each .mat file is one
-- experiment: a continuous vibration recording (tens of thousands of samples
-- at 12 kHz) from one bearing condition at one load. Storing every individual
-- sample as a Postgres row would mean 100k+ rows per file and tens of millions
-- of rows total, which is both wasteful and useless for ML (models train on
-- fixed-length windows, not on the raw stream). So the schema stores:
--   experiments  -> metadata about each recording (one row per .mat file)
--   signal_windows -> the recording chopped into fixed-length windows, each
--                     window is what actually gets fed to the classifier
-- This keeps row counts sane (~hundreds of experiments, ~thousands of
-- windows) while still being the real, unmodified sensor data.

CREATE TABLE IF NOT EXISTS depots (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,      -- e.g. 'depot_0hp'
    load_hp      SMALLINT NOT NULL,         -- 0, 1, 2, or 3 -- the real motor load level
    nominal_rpm  SMALLINT                   -- approx. shaft speed at that load (1720-1797 RPM range)
);

CREATE TABLE IF NOT EXISTS experiments (
    id                 SERIAL PRIMARY KEY,
    depot_id           INTEGER NOT NULL REFERENCES depots(id),
    fault_type         TEXT NOT NULL CHECK (fault_type IN ('Normal', 'InnerRace', 'OuterRace', 'Ball')),
    fault_diameter_in  NUMERIC(5,3),        -- NULL for Normal; e.g. 0.007, 0.014, 0.021, 0.028 for faults
    sensor_position    TEXT NOT NULL CHECK (sensor_position IN ('DE', 'FE', 'BA')),  -- Drive End / Fan End / Base accelerometer
    sampling_rate_hz   INTEGER NOT NULL DEFAULT 12000,
    source_file        TEXT NOT NULL,       -- original .mat filename, for traceability back to the raw data
    source_variable    TEXT NOT NULL,       -- the exact MATLAB struct key read, e.g. 'X097_DE_time'
    n_samples          INTEGER NOT NULL,    -- length of the raw signal before windowing
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signal_windows (
    id             BIGSERIAL PRIMARY KEY,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    window_index   INTEGER NOT NULL,        -- 0, 1, 2, ... within this experiment
    start_sample   INTEGER NOT NULL,
    end_sample     INTEGER NOT NULL,
    raw_values     DOUBLE PRECISION[] NOT NULL,  -- the actual windowed signal, e.g. 2048 raw values
    label          TEXT NOT NULL,           -- denormalized copy of experiments.fault_type, for fast ML-loading queries
    UNIQUE (experiment_id, window_index)
);

-- Populated once you start training models (later phase) — created now so the
-- schema doesn't need a migration mid-project.
CREATE TABLE IF NOT EXISTS model_predictions (
    id               BIGSERIAL PRIMARY KEY,
    window_id        BIGINT NOT NULL REFERENCES signal_windows(id),
    model_version    TEXT NOT NULL,         -- e.g. 'local_cnn_v1', 'fedavg_round_20'
    predicted_label  TEXT NOT NULL,
    confidence       REAL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Populated once the LangGraph agent layer runs (later phase).
CREATE TABLE IF NOT EXISTS agent_decisions (
    id             BIGSERIAL PRIMARY KEY,
    prediction_id  BIGINT NOT NULL REFERENCES model_predictions(id),
    agent_name     TEXT NOT NULL,           -- 'monitor' | 'diagnostic' | 'risk' | 'maintenance'
    decision_text  TEXT NOT NULL,
    risk_score     REAL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiments_depot ON experiments(depot_id);
CREATE INDEX IF NOT EXISTS idx_windows_experiment ON signal_windows(experiment_id);
CREATE INDEX IF NOT EXISTS idx_windows_label ON signal_windows(label);
CREATE INDEX IF NOT EXISTS idx_predictions_window ON model_predictions(window_id);

-- Seed the 4 real depots (load levels). Safe to re-run.
INSERT INTO depots (name, load_hp, nominal_rpm) VALUES
    ('depot_0hp', 0, 1797),
    ('depot_1hp', 1, 1772),
    ('depot_2hp', 2, 1750),
    ('depot_3hp', 3, 1730)
ON CONFLICT (name) DO NOTHING;
