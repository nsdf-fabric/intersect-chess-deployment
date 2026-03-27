"""CHESS Autonomous Experiment Campaign Orchestrator with Dashboard.

Orchestrates the autonomous experiment feedback loop across INTERSECT services
and serves a live web dashboard with experiment visualization.

Services:
  1. chess-instrument-control-service — motor position writes
  2. chess-data-service — HDF5 monitoring, emits new_measurement events
  3. dial-service — Bayesian optimization (active learning)

Dashboard features:
  - Broker/client connection status
  - Start/stop orchestrator control
  - Live 2D plot: measured points, next recommendation, surrogate model heatmap

Usage:
    python campaign_orchestrator.py --config campaign-conf.json [--host 0.0.0.0] [--port 8000]

Requires:
    pip install intersect_sdk[amqp]>=0.8.0,<0.9.0 fastapi uvicorn
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("chess-campaign")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(
    os.getenv("CAMPAIGN_CONFIG_PATH", str(Path(__file__).with_name("campaign-conf.json")))
)
DEFAULT_NUM_ITERATIONS = 30
SURROGATE_GRID_SIZE = 30
MAX_LISTENER_RETRIES = 10
LISTENER_RETRY_DELAY = 10  # seconds
SERVICE_STARTUP_WAIT = 10  # seconds — wait for services to bind AMQP queues

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="CHESS Campaign Orchestrator", version="0.1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_grid(
    bounds: list[list[float]], grid_size: int = SURROGATE_GRID_SIZE,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Generate a 2D grid of prediction points within *bounds*."""
    x_min, x_max = bounds[0]
    z_min, z_max = bounds[1]
    xs = [x_min + i * (x_max - x_min) / (grid_size - 1) for i in range(grid_size)]
    zs = [z_min + i * (z_max - z_min) / (grid_size - 1) for i in range(grid_size)]
    # Row-major: z varies in outer loop, x in inner loop
    points = [[x, z] for z in zs for x in xs]
    return xs, zs, points


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class ChessAutonomousExperimentOrchestrator:
    """Campaign orchestrator with dashboard state tracking.

    Coordinates:
      - chess-data-egress-service: monitors HDF5 for new data
      - dial-service: performs active learning (Bayesian optimization)
      - chess-instrument-control-service: writes motor positions for SPEC
    """

    def __init__(
        self,
        data_service_destination: str,
        dial_service_destination: str,
        instrument_control_destination: str,
        experiment_name: str,
        base_dir: str,
        hdf5_filename: str,
        dataset_path: str,
        initial_dataset_x: list[list[float]],
        initial_dataset_y: list[float],
        bounds: list[list[float]],
    ):
        # Service destinations
        self.data_service = data_service_destination
        self.dial_service = dial_service_destination
        self.instrument_control = instrument_control_destination

        # Experiment parameters
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.hdf5_filename = hdf5_filename
        self.dataset_path = dataset_path
        self.initial_dataset_x = initial_dataset_x
        self.initial_dataset_y = initial_dataset_y
        self.bounds = bounds

        # Thread safety
        self._lock = threading.Lock()

        # INTERSECT client reference
        self._client: IntersectClient | None = None

        # Broker / lifecycle state
        self.broker_connected = False
        self.broker_status = "not_started"
        self.broker_error = ""
        self.enabled = True

        # Campaign settings
        self.num_iterations = DEFAULT_NUM_ITERATIONS
        self._send_init_messages = False
        self._skip_startup_wait = False

        # Campaign state
        self.workflow_id = ""
        self.iteration = 0
        self.campaign_complete = False
        self.last_error = ""

        # Plot data — measured points (seeded with initial data)
        self.measured_x: list[list[float]] = [list(pt) for pt in initial_dataset_x]
        self.measured_y: list[float] = list(initial_dataset_y)
        self.next_point: list[float] | None = None

        # Surrogate prediction grid
        self.grid_x, self.grid_z, self._grid_points = _make_grid(bounds)
        self.surrogate_means: list[list[float]] | None = None
        self.surrogate_stds: list[list[float]] | None = None

        logger.info(
            "Orchestrator initialized: data=%s dial=%s instrument=%s",
            data_service_destination,
            dial_service_destination,
            instrument_control_destination,
        )

    # ---- Client / state helpers -------------------------------------------
    def set_client(self, client: IntersectClient) -> None:
        self._client = client

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self.enabled = value

    def set_broker_health(
        self, *, connected: bool, status: str, error: str | None = None,
    ) -> None:
        with self._lock:
            self.broker_connected = connected
            self.broker_status = status
            self.broker_error = error or ""

    def _record_error(self, message: str) -> None:
        with self._lock:
            self.last_error = message
        logger.error(message)

    def reset(self) -> None:
        """Reset campaign state so a new run can begin."""
        with self._lock:
            self.workflow_id = ""
            self.iteration = 0
            self.campaign_complete = False
            self.last_error = ""
            self.measured_x = [list(pt) for pt in self.initial_dataset_x]
            self.measured_y = list(self.initial_dataset_y)
            self.next_point = None
            self.surrogate_means = None
            self.surrogate_stds = None
            self._send_init_messages = False
        logger.info("Campaign state reset")

    # ---- Snapshots for API ------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "broker_connected": self.broker_connected,
                "broker_status": self.broker_status,
                "broker_error": self.broker_error,
                "enabled": self.enabled,
                "workflow_id": self.workflow_id,
                "iteration": self.iteration,
                "num_iterations": self.num_iterations,
                "campaign_complete": self.campaign_complete,
                "num_measurements": len(self.measured_x),
                "last_error": self.last_error,
                "next_point": self.next_point,
            }

    def plot_snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "measured_x": list(self.measured_x),
                "measured_y": list(self.measured_y),
                "next_point": self.next_point,
                "bounds": self.bounds,
                "iteration": self.iteration,
                "workflow_id": self.workflow_id,
                "campaign_complete": self.campaign_complete,
            }
            if self.surrogate_means is not None:
                result["surrogate"] = {
                    "grid_x": self.grid_x,
                    "grid_z": self.grid_z,
                    "means": self.surrogate_means,
                    "stds": self.surrogate_stds,
                }
            else:
                result["surrogate"] = None
            return result

    # ---- INTERSECT initial messages ---------------------------------------
    def initial_messages(self) -> IntersectClientCallback:
        """Return initial messages for the INTERSECT client.

        When ``_send_init_messages`` is True the experiment-init message is
        included so the workflow chain starts immediately on connect.
        Otherwise only an event subscription is registered (broker connects
        but stays idle).  SDK v0.8 requires at least one message or event
        subscription in the initial config.
        """
        if not self._send_init_messages:
            # Connect to broker idle — subscribe to events only so the SDK
            # accepts the config, but don't kick off any workflow.
            return IntersectClientCallback(
                services_to_start_listening_for_events=[
                    self.data_service,
                ],
            )
        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.instrument_control,
                    operation="chess_instrument_control.initialize_experiment",
                    payload={
                        "experiment_name": self.experiment_name,
                        "base_dir": self.base_dir,
                    },
                ),
            ],
        )

    # ---- INTERSECT callback (default handler) -----------------------------
    def __call__(
        self,
        _source: str,
        operation: str,
        has_error: bool,
        payload: INTERSECT_JSON_VALUE,
    ) -> IntersectClientCallback:
        """Handle responses from INTERSECT services."""
        if has_error:
            self._record_error(f"Error from {operation}: {payload}")
            # Surrogate errors are non-fatal
            if operation == "dial.get_surrogate_values":
                return IntersectClientCallback()
            logger.error("Fatal service error from %s — stopping campaign", operation)
            with self._lock:
                self.campaign_complete = True
            return IntersectClientCallback()

        # Step 1: Experiment initialized -> start monitoring + initialize Dial
        if operation == "chess_instrument_control.initialize_experiment":
            logger.info("Experiment initialized at: %s", payload)
            return IntersectClientCallback(
                messages_to_send=[
                    IntersectDirectMessageParams(
                        destination=self.data_service,
                        operation="chess_data_egress.start_monitoring",
                        payload={
                            "filename": self.hdf5_filename,
                            "dataset_path": self.dataset_path,
                            "poll_interval": 0.5,
                        },
                    ),
                    IntersectDirectMessageParams(
                        destination=self.dial_service,
                        operation="dial.initialize_workflow",
                        payload={
                            "dataset_x": self.initial_dataset_x,
                            "dataset_y": self.initial_dataset_y,
                            "bounds": self.bounds,
                            "kernel": "rbf",
                            "y_is_good": True,
                            "backend": "sklearn",
                        },
                    ),
                ],
            )

        # Step 2: Dial workflow initialized -> subscribe to events + get initial surrogate
        if operation == "dial.initialize_workflow":
            with self._lock:
                self.workflow_id = payload
            logger.info("Dial workflow initialized: %s", self.workflow_id)
            return IntersectClientCallback(
                services_to_start_listening_for_events=[
                    self.data_service,
                ],
                messages_to_send=[
                    # Request initial surrogate model for dashboard visualization
                    IntersectDirectMessageParams(
                        destination=self.dial_service,
                        operation="dial.get_surrogate_values",
                        payload={
                            "workflow_id": payload,
                            "points_to_predict": self._grid_points,
                        },
                    ),
                    # Request the first next-point to kick off the loop
                    IntersectDirectMessageParams(
                        destination=self.dial_service,
                        operation="dial.get_next_point",
                        payload={
                            "workflow_id": payload,
                            "strategy": "expected_improvement",
                            "bounds": self.bounds,
                        },
                    ),
                ],
            )

        # Step 3: New measurement events are handled by on_event()

        # Responses we acknowledge but don't need to act on
        if operation == "chess_data_egress.start_monitoring":
            logger.info("Data monitoring started: %s", payload)
            return IntersectClientCallback()

        # Step 4: Dial updated -> ask for next point + refresh surrogate
        if operation == "dial.update_workflow_with_data":
            return IntersectClientCallback(
                messages_to_send=[
                    IntersectDirectMessageParams(
                        destination=self.dial_service,
                        operation="dial.get_next_point",
                        payload={
                            "workflow_id": self.workflow_id,
                            "strategy": "expected_improvement",
                            "bounds": self.bounds,
                        },
                    ),
                    IntersectDirectMessageParams(
                        destination=self.dial_service,
                        operation="dial.get_surrogate_values",
                        payload={
                            "workflow_id": self.workflow_id,
                            "points_to_predict": self._grid_points,
                        },
                    ),
                ],
            )

        # Step 5a: Surrogate values received -> store for dashboard
        if operation == "dial.get_surrogate_values":
            self._store_surrogate(payload)
            return IntersectClientCallback()

        # Step 5b: Dial recommends next point -> write motor position
        if operation == "dial.get_next_point":
            next_labx, next_labz = payload[0], payload[1]
            with self._lock:
                self.iteration += 1
                self.next_point = [next_labx, next_labz]
            logger.info(
                "Iteration %d: next point labx=%.4f, labz=%.4f",
                self.iteration, next_labx, next_labz,
            )

            if self.iteration >= self.num_iterations:
                logger.info("Reached %d iterations — campaign complete.", self.num_iterations)
                with self._lock:
                    self.campaign_complete = True
                return IntersectClientCallback()

            return IntersectClientCallback(
                messages_to_send=[
                    IntersectDirectMessageParams(
                        destination=self.instrument_control,
                        operation="chess_instrument_control.write_motor_position",
                        payload={
                            "labx": next_labx,
                            "labz": next_labz,
                        },
                    ),
                ],
            )

        # Step 6: Motor position written -> wait for next measurement event
        if operation == "chess_instrument_control.write_motor_position":
            logger.info("Motor position written: %s", payload)
            return IntersectClientCallback()

        logger.warning("Unhandled operation: %s", operation)
        return IntersectClientCallback()

    # ---- Event callback (handles service events) -------------------------
    def on_event(
        self,
        _source: str,
        operation: str,
        event_name: str,
        payload: INTERSECT_JSON_VALUE,
    ) -> IntersectClientCallback | None:
        """Handle events from INTERSECT services."""
        if event_name == "new_measurement":
            labx = payload["labx"]
            labz = payload["labz"]
            center_value = payload["center_value"]
            logger.info(
                "New measurement: labx=%.4f, labz=%.4f, value=%.6f",
                labx, labz, center_value,
            )
            with self._lock:
                self.measured_x.append([labx, labz])
                self.measured_y.append(center_value)
            return IntersectClientCallback(
                messages_to_send=[
                    IntersectDirectMessageParams(
                        destination=self.dial_service,
                        operation="dial.update_workflow_with_data",
                        payload={
                            "workflow_id": self.workflow_id,
                            "next_x": [labx, labz],
                            "next_y": center_value,
                        },
                    ),
                ],
            )

        logger.warning("Unhandled event: %s.%s", operation, event_name)
        return None

    # ---- Surrogate data storage -------------------------------------------
    def _store_surrogate(self, payload: INTERSECT_JSON_VALUE) -> None:
        """Reshape flat surrogate predictions into 2D arrays for the heatmap."""
        try:
            means_flat = payload[0]  # predicted values
            stds_flat = payload[2]   # raw uncertainties
            gs = len(self.grid_x)
            means_2d = [means_flat[i * gs:(i + 1) * gs] for i in range(gs)]
            stds_2d = [stds_flat[i * gs:(i + 1) * gs] for i in range(gs)]
            with self._lock:
                self.surrogate_means = means_2d
                self.surrogate_stds = stds_2d
            logger.info("Surrogate model updated (%d×%d grid)", gs, gs)
        except Exception:
            logger.exception("Failed to store surrogate data")


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
orchestrator: ChessAutonomousExperimentOrchestrator | None = None
listener_thread: threading.Thread | None = None
listener_stop_event = threading.Event()
_conf: dict[str, Any] = {}


def listener_running() -> bool:
    return listener_thread is not None and listener_thread.is_alive()


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
def state() -> dict[str, Any]:
    if orchestrator is None:
        return {"error": "orchestrator not initialized"}
    snapshot = orchestrator.snapshot()
    snapshot["listener_running"] = listener_running()
    return snapshot


@app.get("/pipeline")
def pipeline_state() -> dict[str, Any]:
    """Return status data for the dashboard."""
    if orchestrator is None:
        return {"error": "orchestrator not initialized"}
    s = orchestrator.snapshot()
    return {
        "listener_running": listener_running(),
        "enabled": s["enabled"],
        "broker_connected": s["broker_connected"],
        "broker_status": s["broker_status"],
        "broker_error": s["broker_error"],
        "workflow_id": s["workflow_id"],
        "iteration": s["iteration"],
        "num_iterations": s["num_iterations"],
        "campaign_complete": s["campaign_complete"],
        "num_measurements": s["num_measurements"],
        "next_point": s["next_point"],
        "last_error": s["last_error"],
    }


@app.get("/plot-data")
def plot_data() -> dict[str, Any]:
    """Return data for the experiment plot."""
    if orchestrator is None:
        return {"error": "orchestrator not initialized"}
    return orchestrator.plot_snapshot()


@app.post("/start-campaign")
def start_campaign(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start (or restart) the campaign with a given number of iterations."""
    if orchestrator is None:
        return {"error": "orchestrator not initialized"}
    if orchestrator.campaign_complete is False and orchestrator.iteration > 0:
        return {"error": "campaign already running — reset first"}
    num = DEFAULT_NUM_ITERATIONS
    if body and "num_iterations" in body:
        num = int(body["num_iterations"])
    # Stop current idle listener, then restart with init messages enabled
    if listener_running():
        stop_listener_thread()
    orchestrator.reset()
    orchestrator.num_iterations = max(1, num)
    orchestrator._send_init_messages = True
    orchestrator._skip_startup_wait = True
    orchestrator.set_enabled(True)
    start_listener_thread()
    logger.info("Campaign started for %d iterations", orchestrator.num_iterations)
    return pipeline_state()


@app.post("/reset-campaign")
def reset_campaign() -> dict[str, Any]:
    """Stop the running campaign and reset state."""
    if orchestrator is None:
        return {"error": "orchestrator not initialized"}
    if listener_running():
        orchestrator.set_enabled(False)
        stop_listener_thread()
    orchestrator.reset()
    # Reconnect idle (no init messages)
    orchestrator._skip_startup_wait = True
    orchestrator.set_enabled(True)
    start_listener_thread()
    logger.info("Campaign reset — reconnecting idle")
    return pipeline_state()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CHESS Campaign Orchestrator</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 24px; background: #f5f6f8; color: #222;
    }
    h1 { margin: 0 0 4px; font-size: 1.4em; }
    .subtitle { color: #666; font-size: 0.9em; margin-bottom: 20px; }

    /* Pipeline diagram */
    .pipeline {
      display: flex; align-items: center; justify-content: center;
      gap: 0; margin: 20px 0 24px; flex-wrap: nowrap;
    }
    .stage-box {
      width: 160px; height: 90px;
      border: 2.5px solid #3b5a6e; border-radius: 8px;
      background: #e9ecef; color: #3b5a6e;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-weight: 700; font-size: 0.8em; text-align: center;
      position: relative; transition: background 0.5s, color 0.4s, border-color 0.4s;
      line-height: 1.3; user-select: none;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .stage-box.active {
      background: #1f7a1f; color: #fff; border-color: #145214;
      box-shadow: 0 0 12px rgba(31,122,31,0.4);
    }
    .arrow {
      width: 40px; display: flex; align-items: center; justify-content: center;
      font-size: 1.5em; color: #3b5a6e; flex-shrink: 0; user-select: none;
    }

    /* Status bar */
    .status-bar {
      display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 18px;
    }
    .badge {
      display: inline-block; padding: 5px 14px; border-radius: 4px;
      font-size: 0.82em; font-weight: 600; color: #fff;
    }
    .badge.green  { background: #1f7a1f; }
    .badge.red    { background: #b42318; }
    .badge.yellow { background: #b08a00; }
    .badge.gray   { background: #6c757d; }
    .toggle-btn {
      padding: 6px 16px; border: 1px solid #3b5a6e; border-radius: 4px;
      background: #fff; color: #3b5a6e; font-weight: 600; cursor: pointer;
      font-size: 0.82em; transition: background 0.2s;
    }
    .toggle-btn:hover { background: #e9ecef; }

    /* Cards */
    .card {
      background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 16px;
    }
    .card h3 { margin: 0 0 8px; font-size: 0.95em; color: #3b5a6e; }
    .card table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
    .card td { padding: 3px 0; }
    .card td:first-child { font-weight: 600; color: #555; width: 50%; }
    .card td:last-child { text-align: right; }
    .error-text { color: #b42318; font-weight: 600; }

    .details { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 700px) { .details { grid-template-columns: 1fr; } }

    .legend {
      margin-top: 12px; font-size: 0.75em; color: #888; text-align: center;
    }
    .legend span { margin: 0 10px; }
    .legend .dot {
      display: inline-block; width: 10px; height: 10px; border-radius: 50%;
      vertical-align: middle; margin-right: 4px;
    }
    .legend .dot.g { background: #1f7a1f; }
    .legend .dot.d { background: #e9ecef; border: 1.5px solid #3b5a6e; }
  </style>
</head>
<body>
  <h1>CHESS Campaign Orchestrator</h1>
  <div class="subtitle">Autonomous Experiment: Data Service &rarr; DIAL &rarr; Instrument Control</div>

  <!-- Pipeline diagram -->
  <div class="pipeline">
    <div class="stage-box" id="box-data">
      <span>Data<br>Service</span>
    </div>
    <div class="arrow">&#x27A1;</div>
    <div class="stage-box" id="box-orchestrator">
      <span>Campaign<br>Orchestrator</span>
    </div>
    <div class="arrow">&#x27A1;</div>
    <div class="stage-box" id="box-dial">
      <span>DIAL<br>Service</span>
    </div>
    <div class="arrow">&#x27A1;</div>
    <div class="stage-box" id="box-instrument">
      <span>Instrument<br>Control</span>
    </div>
  </div>

  <!-- Status bar -->
  <div class="status-bar">
    <span class="badge gray" id="badge-broker">Broker: --</span>
    <span class="badge gray" id="badge-client">Client: --</span>
    <span class="badge gray" id="badge-iteration">Iteration: 0</span>
  </div>

  <!-- Campaign controls -->
  <div class="status-bar" style="margin-top:0">
    <label style="font-weight:600;font-size:0.85em;color:#3b5a6e">Iterations:
      <input id="iter-input" type="number" min="1" value="30"
             style="width:70px;padding:4px 6px;border:1px solid #aaa;border-radius:4px;font-size:0.95em;margin-left:4px">
    </label>
    <button class="toggle-btn" id="start-btn" onclick="startCampaign()"
            style="background:#1f7a1f;color:#fff;border-color:#145214">&#9654; Start Campaign</button>
    <button class="toggle-btn" id="reset-btn" onclick="resetCampaign()"
            style="background:#b42318;color:#fff;border-color:#8a1a12">&#8635; Reset</button>
  </div>

  <!-- Plot -->
  <div class="card">
    <h3>Experiment Progress — Surrogate Model &amp; Measurements</h3>
    <div id="plot" style="width:100%;height:520px;"></div>
  </div>

  <!-- Details -->
  <div class="details">
    <div class="card">
      <h3>Campaign Info</h3>
      <table>
        <tr><td>Workflow ID</td><td id="d-workflow-id">--</td></tr>
        <tr><td>Iteration</td><td id="d-iteration">0</td></tr>
        <tr><td>Total measurements</td><td id="d-measurements">0</td></tr>
        <tr><td>Next point</td><td id="d-next-point">--</td></tr>
        <tr><td>Campaign status</td><td id="d-campaign-status">--</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>Connection</h3>
      <table>
        <tr><td>Broker status</td><td id="d-broker-status">--</td></tr>
        <tr><td>Broker error</td><td id="d-broker-error" class="error-text">none</td></tr>
        <tr><td>Last error</td><td id="d-last-error" class="error-text">none</td></tr>
      </table>
    </div>
  </div>

  <div class="legend">
    <span><span class="dot g"></span> Connected / Active</span>
    <span><span class="dot d"></span> Disconnected / Idle</span>
  </div>

<script>
const STATUS_POLL_MS = 2000;
const PLOT_POLL_MS = 5000;
let plotInitialized = false;

// --- Helpers ---
function setBadge(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = "badge " + cls;
}
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text || "--";
}
function setBoxActive(id, active) {
  const el = document.getElementById(id);
  if (!el) return;
  if (active) el.classList.add("active");
  else el.classList.remove("active");
}

// --- Status polling ---
async function pollStatus() {
  try {
    const resp = await fetch("pipeline");
    if (!resp.ok) return;
    const d = await resp.json();
    if (d.error) return;

    const running = d.listener_running;
    setBadge("badge-client", "Client: " + (running ? "ON" : "OFF"), running ? "green" : "gray");
    setBadge("badge-broker", "Broker: " + (d.broker_status || "unknown"),
             d.broker_connected ? "green" : (d.broker_status === "failed" ? "red" : "yellow"));
    const iterText = "Iteration: " + d.iteration + " / " + d.num_iterations;
    setBadge("badge-iteration", iterText,
             d.campaign_complete ? "yellow" : (running ? "green" : "gray"));

    // Enable/disable buttons based on campaign state (not just listener)
    const campaignActive = !!d.workflow_id && !d.campaign_complete;
    const startBtn = document.getElementById("start-btn");
    const resetBtn = document.getElementById("reset-btn");
    const iterInput = document.getElementById("iter-input");
    if (campaignActive) {
      startBtn.disabled = true; startBtn.style.opacity = "0.5";
      iterInput.disabled = true; iterInput.style.opacity = "0.6";
    } else {
      startBtn.disabled = false; startBtn.style.opacity = "1";
      iterInput.disabled = false; iterInput.style.opacity = "1";
    }

    // Pipeline boxes
    setBoxActive("box-data", running && d.broker_connected);
    setBoxActive("box-orchestrator", running && d.broker_connected);
    setBoxActive("box-dial", running && d.broker_connected && !!d.workflow_id);
    setBoxActive("box-instrument", running && d.broker_connected && d.iteration > 0);

    // Info cards
    setText("d-workflow-id", d.workflow_id || "--");
    setText("d-iteration", d.iteration + " / " + d.num_iterations);
    setText("d-measurements", String(d.num_measurements));
    if (d.next_point) {
      setText("d-next-point", "labx=" + d.next_point[0].toFixed(3) + ", labz=" + d.next_point[1].toFixed(3));
    } else { setText("d-next-point", "--"); }
    setText("d-campaign-status", d.campaign_complete ? "Complete" : (campaignActive ? "Running" : "Ready"));
    setText("d-broker-status", d.broker_status);
    setText("d-broker-error", d.broker_error || "none");
    setText("d-last-error", d.last_error || "none");
  } catch (e) { console.error("Status poll error:", e); }
}

// --- Plot polling ---
async function pollPlot() {
  try {
    const resp = await fetch("plot-data");
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.error) return;

    const traces = [];

    // Heatmap: surrogate model predictions
    if (data.surrogate) {
      traces.push({
        type: "heatmap",
        x: data.surrogate.grid_x,
        y: data.surrogate.grid_z,
        z: data.surrogate.means,
        colorscale: "Viridis",
        zsmooth: "best",
        name: "Surrogate Model",
        colorbar: { title: "Predicted", titleside: "right" },
        hovertemplate: "labx=%{x:.2f}<br>labz=%{y:.2f}<br>value=%{z:.4f}<extra>Surrogate</extra>",
      });
    }

    // Scatter: measured points
    if (data.measured_x && data.measured_x.length > 0) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: data.measured_x.map(function(p) { return p[0]; }),
        y: data.measured_x.map(function(p) { return p[1]; }),
        marker: {
          size: 9,
          color: "white",
          line: { color: "black", width: 2 },
          symbol: "circle",
        },
        text: data.measured_y.map(function(v) { return "Measured: " + v.toFixed(4); }),
        hovertemplate: "labx=%{x:.2f}<br>labz=%{y:.2f}<br>%{text}<extra>Measured</extra>",
        name: "Measured Points",
      });
    }

    // Star: next recommended point
    if (data.next_point) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: [data.next_point[0]],
        y: [data.next_point[1]],
        marker: {
          size: 18,
          color: "red",
          symbol: "star",
          line: { color: "white", width: 2 },
        },
        hovertemplate: "labx=%{x:.2f}<br>labz=%{y:.2f}<extra>Next Point</extra>",
        name: "Next Measurement",
      });
    }

    const layout = {
      xaxis: { title: "labx" },
      yaxis: { title: "labz" },
      margin: { t: 30, b: 50, l: 60, r: 20 },
      legend: { x: 0, y: -0.15, orientation: "h" },
      plot_bgcolor: "#f5f6f8",
    };

    if (data.bounds && data.bounds.length >= 2) {
      layout.xaxis.range = data.bounds[0];
      layout.yaxis.range = data.bounds[1];
    }

    Plotly.react("plot", traces, layout, { responsive: true });
    plotInitialized = true;
  } catch (e) { console.error("Plot poll error:", e); }
}

// --- Campaign controls ---
async function startCampaign() {
  const n = parseInt(document.getElementById("iter-input").value) || 30;
  try {
    await fetch("start-campaign", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({num_iterations: n})
    });
    await pollStatus();
  } catch (e) { console.error("Start error:", e); }
}
async function resetCampaign() {
  if (!confirm("Reset the campaign? This will stop any running loop and clear all data.")) return;
  try {
    await fetch("reset-campaign", { method: "POST" });
    await pollStatus();
    await pollPlot();
  } catch (e) { console.error("Reset error:", e); }
}

// --- Init ---
pollStatus();
pollPlot();
setInterval(pollStatus, STATUS_POLL_MS);
setInterval(pollPlot, PLOT_POLL_MS);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# INTERSECT client lifecycle (background thread)
# ---------------------------------------------------------------------------
def run_intersect_listener() -> None:
    """Run the INTERSECT listener with automatic retry on crash."""
    attempt = 0
    while not listener_stop_event.is_set() and attempt < MAX_LISTENER_RETRIES:
        attempt += 1
        try:
            _run_intersect_listener_once()
        except Exception:
            logger.exception(
                "INTERSECT listener crashed (attempt %d/%d)",
                attempt, MAX_LISTENER_RETRIES,
            )
            if orchestrator is not None:
                orchestrator._record_error(
                    f"Listener crashed (attempt {attempt}/{MAX_LISTENER_RETRIES}). "
                    f"Retrying in {LISTENER_RETRY_DELAY}s…"
                )
            if not listener_stop_event.is_set() and attempt < MAX_LISTENER_RETRIES:
                listener_stop_event.wait(LISTENER_RETRY_DELAY)
                continue
        else:
            break

    if attempt >= MAX_LISTENER_RETRIES and orchestrator is not None:
        orchestrator.set_broker_health(
            connected=False, status="failed",
            error=f"Listener failed after {MAX_LISTENER_RETRIES} attempts. Use dashboard to restart.",
        )


def _run_intersect_listener_once() -> None:
    """Single attempt at running the INTERSECT client."""
    if orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")

    orchestrator.set_broker_health(connected=False, status="starting")

    # Wait for other INTERSECT services to complete broker registration.
    # Without this delay, the orchestrator's initial callback chain can send
    # messages before the target services have bound their AMQP queues,
    # causing RabbitMQ to silently drop them.
    # Skip the wait on reconnects (services already registered).
    if orchestrator._skip_startup_wait:
        orchestrator._skip_startup_wait = False
    else:
        logger.info("Waiting %ds for INTERSECT services to register with broker...", SERVICE_STARTUP_WAIT)
        time.sleep(SERVICE_STARTUP_WAIT)

    client_config = IntersectClientConfig(
        initial_message_event_config=orchestrator.initial_messages(),
        **_conf["intersect"],
    )

    client = IntersectClient(
        config=client_config,
        user_callback=orchestrator,
        event_callback=orchestrator.on_event,
    )
    orchestrator.set_client(client)

    try:
        client.startup()
        is_connected = getattr(client, "is_connected", None)
        connected = bool(is_connected()) if callable(is_connected) else False
        if connected:
            orchestrator.set_broker_health(connected=True, status="connected")
        elif client.considered_unrecoverable():
            orchestrator.set_broker_health(
                connected=False, status="failed",
                error="Broker connection is unrecoverable.",
            )
        else:
            orchestrator.set_broker_health(connected=False, status="starting")

        logger.info(
            "INTERSECT listener started — data=%s dial=%s instrument=%s",
            orchestrator.data_service,
            orchestrator.dial_service,
            orchestrator.instrument_control,
        )

        while not listener_stop_event.is_set() and not client.considered_unrecoverable():
            connected = bool(is_connected()) if callable(is_connected) else False
            if connected:
                orchestrator.set_broker_health(connected=True, status="connected")
            else:
                orchestrator.set_broker_health(connected=False, status="starting")
            time.sleep(0.5)

        if client.considered_unrecoverable():
            orchestrator.set_broker_health(
                connected=False, status="failed",
                error="Broker connection is unrecoverable.",
            )
            raise RuntimeError("INTERSECT SDK unrecoverable state — will retry.")
    except Exception:
        orchestrator.set_broker_health(
            connected=False, status="failed",
            error="INTERSECT listener crashed. Check logs for details.",
        )
        raise
    finally:
        client.shutdown(reason="Listener thread stopping")
        if listener_stop_event.is_set():
            orchestrator.set_broker_health(connected=False, status="stopped")
        logger.info("INTERSECT listener stopped")


def start_listener_thread() -> None:
    global listener_thread
    if listener_thread is not None and listener_thread.is_alive():
        return
    if orchestrator is not None:
        orchestrator.set_broker_health(connected=False, status="starting")
    listener_stop_event.clear()
    listener_thread = threading.Thread(
        target=run_intersect_listener, name="intersect-listener", daemon=True,
    )
    listener_thread.start()


def stop_listener_thread() -> None:
    if orchestrator is not None:
        orchestrator.set_broker_health(connected=False, status="stopping")
    listener_stop_event.set()
    if listener_thread is not None and listener_thread.is_alive():
        listener_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
def _load_config() -> dict[str, Any]:
    """Load and return the campaign configuration JSON."""
    with CONFIG_PATH.open("rb") as f:
        return json.load(f)


@app.on_event("startup")
def on_startup() -> None:
    global orchestrator, _conf
    try:
        _conf = _load_config()
    except (json.JSONDecodeError, OSError) as e:
        logger.critical("Unable to load config from %s: %s", CONFIG_PATH, e)
        return

    campaign = _conf["campaign"]
    orchestrator = ChessAutonomousExperimentOrchestrator(
        data_service_destination=campaign["data_service_destination"],
        dial_service_destination=campaign["dial_service_destination"],
        instrument_control_destination=campaign["instrument_control_destination"],
        experiment_name=campaign["experiment_name"],
        base_dir=campaign["base_dir"],
        hdf5_filename=campaign["hdf5_filename"],
        dataset_path=campaign["dataset_path"],
        initial_dataset_x=campaign["initial_dataset_x"],
        initial_dataset_y=campaign["initial_dataset_y"],
        bounds=campaign["bounds"],
    )
    start_listener_thread()
    logger.info("Orchestrator connecting to broker (idle — waiting for Start from dashboard)")


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_listener_thread()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="CHESS Campaign Orchestrator Dashboard")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "campaign-conf.json",
        help="Path to campaign configuration JSON",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind address")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port")
    args = parser.parse_args()

    os.environ["CAMPAIGN_CONFIG_PATH"] = str(args.config.resolve())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
