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

export interface Customer {
  id: number;
  name: string;
  email: string | null;
  company: string | null;
}

export interface Technician {
  id: number;
  name: string;
  phone: string | null;
  status: "available" | "busy" | "offline";
}

export type EquipmentStatus = "active" | "maintenance" | "retired";

export interface Equipment {
  id: number;
  depot_id: number;
  customer_id: number;
  name: string;
  equipment_type: string;
  installed_at: string | null;
  status: EquipmentStatus;
  linked_experiment_id: number | null;
}

export type ServiceRequestStatus =
  | "reported"
  | "diagnosed"
  | "dispatched"
  | "in_progress"
  | "resolved"
  | "cancelled";
export type ServiceRequestPriority = "low" | "medium" | "high" | "critical";

export interface ServiceRequest {
  id: number;
  customer_id: number;
  depot_id: number;
  equipment_id: number | null;
  title: string;
  description: string | null;
  priority: ServiceRequestPriority;
  status: ServiceRequestStatus;
  agent_decision_id: number | null;
  prediction_id: number | null;
  technician_id: number | null;
  created_at: string;
  updated_at: string;
  /** Set only when agent_decision_id is not null (AI-filed request). */
  agent_decision_text: string | null;
  agent_risk_score: number | null;
  predicted_fault: string | null;
}

export interface ServiceRequestCreate {
  customer_id: number;
  depot_id: number;
  equipment_id?: number | null;
  title: string;
  description?: string | null;
  priority?: ServiceRequestPriority;
}

export type WorkOrderStatus =
  | "assigned"
  | "traveling"
  | "in_progress"
  | "completed"
  | "cancelled";

export interface WorkOrder {
  id: number;
  service_request_id: number;
  technician_id: number;
  status: WorkOrderStatus;
  location_lat: number | null;
  location_lng: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface DepotModelMetrics {
  depot_id: number;
  depot_name: string;
  best_val_acc: number;
  train_windows: number;
  val_windows: number;
}

export interface FederatedSummary {
  strategy: string;
  rounds: number;
  mean_baseline_accuracy: number;
  mean_final_accuracy: number;
  gain_vs_baseline: number;
  per_depot_final_accuracy: Record<string, number>;
}

export interface ModelPerformance {
  per_depot: DepotModelMetrics[];
  federated: FederatedSummary | null;
}

export interface MatchedFrequency {
  name: string;
  hz: number;
  band_low_hz: number;
  band_high_hz: number;
  harmonic: number;
}

export interface EvidencePackage {
  service_request_id: number;
  window_id: number;
  equipment_id: number | null;
  site_name: string | null;
  predicted_label: string;
  model_confidence: number | null;
  model_version: string | null;
  raw_values: number[];
  sampling_rate_hz: number;
  spectrum_freqs_hz: number[];
  spectrum_magnitudes: number[];
  matched_frequency: MatchedFrequency | null;
  all_characteristic_frequencies: Record<string, number>;
  signal_rms: number;
  signal_peak: number;
  diagnosis: string;
  diagnosis_confidence: number;
  risk_score: number | null;
  decision_text: string | null;
}

export interface ReplayTick {
  at: string;
  released: number;
  predictions_written: number;
  decisions_written: number;
  equipment_status_changes: number;
  service_requests_filed: number;
  error: string | null;
}

export interface ReplayWorkerState {
  running: boolean;
  interval_seconds: number;
  batch_size: number;
  ticks: number;
  last_tick: ReplayTick | null;
  last_error: string | null;
}

export interface ReplayStatus {
  total_windows: number;
  released: number;
  processed: number;
  pending: number;
  percent_released: number;
  complete: boolean;
  worker: ReplayWorkerState;
}

export interface ReplayStepResult {
  released: number;
  predictions_written: number;
  decisions_written: number;
  equipment_status_changes: number;
  service_requests_filed: number;
  error: string | null;
}

export interface ProductSummary {
  equipment_by_status: Record<string, number>;
  open_requests_by_priority: Record<string, number>;
  ai_flagged_open: number;
  active_work_orders: number;
}
