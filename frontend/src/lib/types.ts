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
