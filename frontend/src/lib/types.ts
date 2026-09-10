export interface Health {
  status: string;
  database: string;
  version: string;
}

export interface Depot {
  id: number;
  name: string;
  load_hp: number;
  nominal_rpm: number | null;
  n_experiments: number;
}

export interface Experiment {
  id: number;
  depot_id: number;
  fault_type: string;
  fault_diameter_in: number | null;
  sensor_position: string;
  sampling_rate_hz: number;
  source_file: string;
  source_variable: string;
  n_samples: number;
}

export interface WindowSummary {
  id: number;
  experiment_id: number;
  window_index: number;
  start_sample: number;
  end_sample: number;
  label: string;
}

export interface WindowDetail extends WindowSummary {
  raw_values: number[];
}

export interface FaultDistributionEntry {
  label: string;
  n_windows: number;
}
