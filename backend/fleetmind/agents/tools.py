"""The deterministic evidence functions, exposed as LLM-callable tools.

Nothing here computes anything new about a fault. Every tool is a thin,
read-only wrapper around code that already exists and already ran:

    get_signal_statistics      -> agents.agents._signal_stats
    get_risk_assessment        -> agents.agents.risk_agent + maintenance_agent
    get_bearing_frequency_match-> ml.bearing_freq.matched_frequency_for_fault
    measure_spectral_band_energy -> ml.bearing_freq.spectrum (numpy FFT)

That separation is the whole point of the design. The numbers are produced by
auditable Python; the language model's only job is to *reason over* them and
write the explanation a technician reads. It is told, and the tool schemas
enforce, that it may not state a figure it did not receive from a tool.

`measure_spectral_band_energy` is the one tool that takes real arguments: it
lets the model actually check whether energy sits on the defect frequency
rather than take the match on faith. It is still pure measurement -- an FFT
over the stored window -- not a judgement.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fleetmind.agents.agents import (
    FAULT_SEVERITY,
    LOAD_RISK_FACTOR,
    _signal_stats,
    diagnostic_agent,
    maintenance_agent,
    risk_agent,
)
from fleetmind.ml.bearing_freq import (
    characteristic_frequencies,
    matched_frequency_for_fault,
    shaft_speed_hz,
    spectrum,
)


@dataclass(frozen=True)
class EvidenceContext:
    """Everything the tools need about one flagged signal window.

    Built from the same ServiceRequest -> AgentDecision -> ModelPrediction ->
    SignalWindow -> Experiment -> Depot chain the evidence endpoint walks, so
    the tools and the evidence plots are guaranteed to describe one window.
    """

    service_request_id: int
    window_id: int
    equipment_name: str | None
    site_name: str | None
    depot_id: int
    load_hp: int | None
    nominal_rpm: float
    sensor_position: str
    sampling_rate_hz: int
    predicted_label: str
    model_confidence: float | None
    model_version: str | None
    raw_values: list[float]
    stored_risk_score: float | None
    stored_decision_text: str | None


# ── tool implementations ───────────────────────────────────────────────


def get_signal_statistics(ctx: EvidenceContext) -> dict[str, Any]:
    """RMS and peak-to-peak amplitude of the window."""
    rms, peak = _signal_stats(ctx.raw_values)
    return {
        "window_id": ctx.window_id,
        "n_samples": len(ctx.raw_values),
        "sampling_rate_hz": ctx.sampling_rate_hz,
        "duration_seconds": round(len(ctx.raw_values) / ctx.sampling_rate_hz, 4),
        "rms_g": round(rms, 6),
        "peak_to_peak_g": round(peak, 6),
    }


def get_risk_assessment(ctx: EvidenceContext) -> dict[str, Any]:
    """Re-run the rule-based risk and maintenance agents over this window.

    Same functions, same inputs as the decision already stored in
    `agent_decisions`, so the numbers returned here are the stored ones.
    `stored_risk_score` is included so a divergence would be visible rather
    than silent.
    """
    state = diagnostic_agent(
        {
            "predicted_label": ctx.predicted_label,
            "confidence": ctx.model_confidence,
            "signal_values": ctx.raw_values,
        }
    )
    state = risk_agent({**state, "depot_id": ctx.depot_id, "load_hp": ctx.load_hp})
    state = maintenance_agent(state)
    load_factor = LOAD_RISK_FACTOR.get(ctx.load_hp, 1.0) if ctx.load_hp is not None else 1.0
    return {
        "predicted_fault": ctx.predicted_label,
        "model_confidence": ctx.model_confidence,
        "diagnosis": state["diagnosis"],
        "diagnosis_confidence": state["diagnosis_confidence"],
        "fault_severity": FAULT_SEVERITY.get(ctx.predicted_label, 0.5),
        "motor_load_hp": ctx.load_hp,
        "load_risk_factor": load_factor,
        "risk_score": state["risk_score"],
        "risk_factors": state["risk_factors"],
        "urgency": state["maintenance_urgency"],
        "rule_based_action": state["maintenance_action"],
        "stored_risk_score": ctx.stored_risk_score,
    }


def get_bearing_frequency_match(ctx: EvidenceContext) -> dict[str, Any]:
    """The characteristic defect frequency that would corroborate this fault.

    Explicitly NOT an explanation of the model: the CNN never sees these
    frequencies (see ml/bearing_freq.py). This is independent bearing physics
    a human can check the spectrum against.
    """
    matched = matched_frequency_for_fault(ctx.predicted_label, ctx.nominal_rpm, ctx.sensor_position)
    return {
        "shaft_speed_hz": round(shaft_speed_hz(ctx.nominal_rpm), 2),
        "nominal_rpm": ctx.nominal_rpm,
        "sensor_position": ctx.sensor_position,
        "all_characteristic_frequencies_hz": characteristic_frequencies(
            ctx.nominal_rpm, ctx.sensor_position
        ),
        "matched_frequency": (
            {
                "name": matched.name,
                "hz": matched.hz,
                "band_low_hz": matched.band_low_hz,
                "band_high_hz": matched.band_high_hz,
                "harmonic": matched.harmonic,
            }
            if matched
            else None
        ),
        "note": (
            "These frequencies are bearing geometry, not model internals. The "
            "classifier consumes a 512-bin normalized FFT magnitude and was "
            "never given these values."
        ),
    }


def measure_spectral_band_energy(
    ctx: EvidenceContext, low_hz: float, high_hz: float
) -> dict[str, Any]:
    """Share of spectral energy sitting in [low_hz, high_hz], plus the peak there."""
    if high_hz <= low_hz:
        raise ValueError("high_hz must be greater than low_hz")

    freqs, mags = spectrum(ctx.raw_values, ctx.sampling_rate_hz, max_hz=float("inf"))
    if not freqs:
        return {"error": "window has no samples"}

    total = sum(m * m for m in mags)
    in_band = [(f, m) for f, m in zip(freqs, mags, strict=True) if low_hz <= f <= high_hz]
    band_energy = sum(m * m for _, m in in_band)
    peak_f, peak_m = max(in_band, key=lambda p: p[1]) if in_band else (None, None)
    overall_f, overall_m = max(zip(freqs, mags, strict=True), key=lambda p: p[1])

    return {
        "band_low_hz": low_hz,
        "band_high_hz": high_hz,
        "fraction_of_total_energy": round(band_energy / total, 6) if total else 0.0,
        "peak_frequency_in_band_hz": peak_f,
        "peak_magnitude_in_band": peak_m,
        "overall_peak_frequency_hz": overall_f,
        "overall_peak_magnitude": overall_m,
        "note": (
            "A band holding a small share of total energy is weak corroboration; "
            "say so rather than overstating the match."
        ),
    }


# ── Anthropic tool schemas ─────────────────────────────────────────────
#
# `strict` + `additionalProperties: false` so the arguments the model sends
# back are guaranteed to validate before we run anything with them.

_NO_ARGS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_signal_statistics",
        "description": (
            "Time-domain statistics of the vibration window that triggered this "
            "service request: RMS amplitude, peak-to-peak amplitude, sample "
            "count, sampling rate and duration. Call this first."
        ),
        "strict": True,
        "input_schema": _NO_ARGS,
    },
    {
        "name": "get_risk_assessment",
        "description": (
            "The rule-based assessment already on file for this window: the "
            "model's predicted fault class and confidence, the fault severity "
            "weight, the site's motor load and load factor, the computed 0-1 "
            "risk score, the listed risk factors, and the urgency band with "
            "its canned action. These are the authoritative numbers."
        ),
        "strict": True,
        "input_schema": _NO_ARGS,
    },
    {
        "name": "get_bearing_frequency_match",
        "description": (
            "Bearing characteristic defect frequencies (BPFI/BPFO/BSF/FTF) for "
            "this machine's shaft speed and sensor position, and which one "
            "would corroborate the predicted fault, with a +/-5% band. These "
            "come from bearing geometry and are independent of the model."
        ),
        "strict": True,
        "input_schema": _NO_ARGS,
    },
    {
        "name": "measure_spectral_band_energy",
        "description": (
            "Measure how much of the window's spectral energy falls inside a "
            "frequency band, and where the peak in that band sits. Use it to "
            "check whether energy actually concentrates on the defect "
            "frequency returned by get_bearing_frequency_match."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "low_hz": {"type": "number", "description": "Lower edge of the band in Hz."},
                "high_hz": {"type": "number", "description": "Upper edge of the band in Hz."},
            },
            "required": ["low_hz", "high_hz"],
            "additionalProperties": False,
        },
    },
]


def openai_tool_definitions() -> list[dict[str, Any]]:
    """The same tools in OpenAI/Groq function-calling shape.

    Derived from TOOL_DEFINITIONS rather than written twice, so the schema a
    model is shown can never drift from the one `run_tool` validates against.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_DEFINITIONS
    ]


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "get_signal_statistics": get_signal_statistics,
    "get_risk_assessment": get_risk_assessment,
    "get_bearing_frequency_match": get_bearing_frequency_match,
    "measure_spectral_band_energy": measure_spectral_band_energy,
}


def run_tool(ctx: EvidenceContext, name: str, arguments: dict[str, Any]) -> str:
    """Execute one tool call and return its JSON result.

    Raises KeyError for an unknown tool so the caller can return an explicit
    `is_error` tool_result rather than silently feeding back nothing.
    """
    fn = _DISPATCH[name]
    return json.dumps(fn(ctx, **arguments), default=str)
