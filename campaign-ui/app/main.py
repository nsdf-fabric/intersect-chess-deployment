import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Campaign UI")
templates = Jinja2Templates(directory="app/templates")


@dataclass(frozen=True)
class AppConfig:
    orchestrator_base_url: str
    orchestrator_events_url: str
    orchestrator_api_key: str
    preload_root: Path


def load_config() -> AppConfig:
    conf_path = Path(__file__).with_name("conf.yaml")
    with conf_path.open("r", encoding="utf-8") as f:
        conf_data = yaml.safe_load(f) or {}

    base_url = str(conf_data.get("orchestrator_base_url", "http://orchestrator:8000")).rstrip("/")
    events_url = str(
        conf_data.get("orchestrator_events_url", "ws://localhost:8000/v1/orchestrator/events")
    ).strip()
    api_key = str(conf_data.get("orchestrator_api_key", "1234567890abcdef1234567890abcdef"))
    preload_root = str(conf_data.get("preload_root", "/campaigns"))

    # Environment variables override YAML when provided.
    base_url = os.getenv("CAMPAIGN_UI_ORCH_BASE_URL", base_url).rstrip("/")
    events_url = os.getenv("CAMPAIGN_UI_ORCH_EVENTS_URL", events_url).strip()
    api_key = os.getenv("CAMPAIGN_UI_API_KEY", api_key)
    preload_root = os.getenv("CAMPAIGN_PRELOAD_DIR", preload_root)

    return AppConfig(
        orchestrator_base_url=base_url,
        orchestrator_events_url=events_url,
        orchestrator_api_key=api_key,
        preload_root=Path(preload_root),
    )


config = load_config()


@dataclass
class UIState:
    selected_name: str | None = None
    campaign_json: str = ""
    graph_mermaid: str = "graph LR\n  A[Load campaign JSON] --> B[Preview workflow graph]"
    status_message: str = ""
    status_kind: str = "info"
    active_campaign_id: str = ""
    task_node_map_json: str = "{}"


state = UIState()
preloaded_campaigns: dict[str, str] = {}
DEFAULT_PRELOADED_CAMPAIGN = "conf/chess-autonomous-legacy-callback-flow.campaign.json"


def _escape_mermaid_label(value: str) -> str:
    return (
        value.replace('"', "'")
        .replace("\\n", " ")
        .replace("<", "")
        .replace(">", "")
        .strip()
    )


def _task_label(task: dict[str, Any]) -> str:
    capability = str(task.get("capability") or "unknown-capability")
    operation = task.get("operation_id")
    event_name = task.get("event_name")
    activity = str(operation or event_name or "unknown-activity")
    kind = "op" if operation else "event"
    return f"{capability}\\n{kind}: {activity}"


def _build_mermaid(campaign: dict[str, Any]) -> tuple[str, dict[str, str]]:
    lines = ["graph TD"]
    root_id = "campaign_root"
    campaign_name = _escape_mermaid_label(str(campaign.get("name") or campaign.get("id") or "campaign"))
    lines.append(f'  {root_id}["Campaign: {campaign_name}"]')

    group_anchor_map: dict[str, str] = {}
    task_node_map: dict[str, str] = {}
    task_group_map: dict[str, str] = {}
    pending_group_deps: list[tuple[str, str]] = []
    group_anchor_ids: list[str] = []
    task_node_ids: list[str] = []

    edge_index = 0
    root_edge_indexes: list[int] = []
    group_edge_indexes: list[int] = []
    task_dep_edge_indexes: list[int] = []

    def _add_edge(edge_stmt: str, style_kind: str) -> None:
        nonlocal edge_index
        lines.append(f"  {edge_stmt}")
        if style_kind == "root":
            root_edge_indexes.append(edge_index)
        elif style_kind == "group":
            group_edge_indexes.append(edge_index)
        elif style_kind == "task_dep":
            task_dep_edge_indexes.append(edge_index)
        edge_index += 1

    for group_index, task_group in enumerate(campaign.get("task_groups") or []):
        group_id_raw = str(task_group.get("id") or f"group-{group_index + 1}")
        subgraph_id = f"sg_{group_index + 1}"
        group_anchor_id = f"g_{group_index + 1}_anchor"
        group_anchor_map[group_id_raw] = group_anchor_id
        group_anchor_ids.append(group_anchor_id)

        lines.append(f'  subgraph {subgraph_id}["Task Group {group_index + 1}"]')
        lines.append("    direction TB")
        lines.append(f'    {group_anchor_id}["Task Group {group_index + 1}"]')

        for task_index, task in enumerate(task_group.get("tasks") or []):
            task_id_raw = str(task.get("id") or f"task-{group_index + 1}-{task_index + 1}")
            task_node_id = f"t_{group_index + 1}_{task_index + 1}"
            task_node_map[task_id_raw] = task_node_id
            task_group_map[task_id_raw] = group_id_raw
            task_node_ids.append(task_node_id)
            task_label = _escape_mermaid_label(_task_label(task))
            lines.append(f'    {task_node_id}["{task_label}"]')
        lines.append("  end")

        _add_edge(f"{root_id} --> {group_anchor_id}", "root")

        for dependency in task_group.get("group_dependencies") or []:
            pending_group_deps.append((str(dependency), group_id_raw))

    for source_group_raw, target_group_raw in pending_group_deps:
        source_anchor_id = group_anchor_map.get(source_group_raw)
        target_anchor_id = group_anchor_map.get(target_group_raw)
        if source_anchor_id and target_anchor_id:
            _add_edge(f"{source_anchor_id} ==> {target_anchor_id}", "group")

    for group_index, task_group in enumerate(campaign.get("task_groups") or []):
        current_group_raw = str(task_group.get("id") or f"group-{group_index + 1}")
        for task_index, task in enumerate(task_group.get("tasks") or []):
            task_id_raw = str(task.get("id") or f"task-{group_index + 1}-{task_index + 1}")
            task_node_id = task_node_map.get(task_id_raw)
            if not task_node_id:
                continue
            for task_dependency in task.get("task_dependencies") or []:
                source_task_id = str(task_dependency)
                source_node_id = task_node_map.get(source_task_id)
                source_group_raw = task_group_map.get(source_task_id)
                # Keep task dependency arrows inside their task-group box only.
                if source_node_id and source_group_raw == current_group_raw:
                    _add_edge(f"{source_node_id} -.-> {task_node_id}", "task_dep")

    lines.append("  classDef groupAnchor fill:#fef3c7,stroke:#b45309,stroke-width:1.5px,color:#7c2d12")
    lines.append("  classDef taskNode fill:#ffffff,stroke:#9ca3af,stroke-width:1px,color:#111827")

    for group_anchor_id in group_anchor_ids:
        lines.append(f"  class {group_anchor_id} groupAnchor")
    for task_node_id in task_node_ids:
        lines.append(f"  class {task_node_id} taskNode")

    if root_edge_indexes:
        root_style_targets = ",".join(str(i) for i in root_edge_indexes)
        lines.append(f"  linkStyle {root_style_targets} stroke:#9ca3af,stroke-width:1.2px")
    if group_edge_indexes:
        group_style_targets = ",".join(str(i) for i in group_edge_indexes)
        lines.append(f"  linkStyle {group_style_targets} stroke:#b45309,stroke-width:2.4px,color:#b45309")
    if task_dep_edge_indexes:
        task_dep_style_targets = ",".join(str(i) for i in task_dep_edge_indexes)
        lines.append(
            f"  linkStyle {task_dep_style_targets} stroke:#1d4ed8,stroke-width:2px,stroke-dasharray: 5 3,color:#1d4ed8"
        )

    return "\n".join(lines), task_node_map


def _orchestrator_events_url() -> str:
    return config.orchestrator_events_url


def _normalize_and_set_campaign(name: str, campaign_text: str) -> None:
    campaign = json.loads(campaign_text)
    state.selected_name = name
    state.campaign_json = json.dumps(campaign, indent=2)
    state.graph_mermaid, task_node_map = _build_mermaid(campaign)
    state.task_node_map_json = json.dumps(task_node_map)


def _load_preloaded_campaigns() -> dict[str, str]:
    campaigns: dict[str, str] = {}
    if not config.preload_root.exists():
        return campaigns

    for campaign_file in sorted(config.preload_root.rglob("*.campaign.json")):
        relative_name = str(campaign_file.relative_to(config.preload_root))
        try:
            content = campaign_file.read_text(encoding="utf-8")
            json.loads(content)
            campaigns[relative_name] = content
        except (OSError, json.JSONDecodeError):
            continue
    return campaigns


async def _render_workspace(request: Request, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="partials/workspace.html",
        context={
            "state": state,
            "preloaded_campaigns": preloaded_campaigns,
            "orchestrator_events_url": _orchestrator_events_url(),
        },
        status_code=status_code,
    )


@app.on_event("startup")
def startup() -> None:
    global preloaded_campaigns
    preloaded_campaigns = _load_preloaded_campaigns()
    if preloaded_campaigns:
        default_name = DEFAULT_PRELOADED_CAMPAIGN
        if default_name not in preloaded_campaigns:
            default_name = next(iter(preloaded_campaigns))
        _normalize_and_set_campaign(default_name, preloaded_campaigns[default_name])
        state.status_message = f"Loaded default campaign: {default_name}"
        state.status_kind = "success"
    else:
        state.status_message = "No preloaded campaign files found. Upload one to begin."
        state.status_kind = "warning"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "state": state,
            "preloaded_campaigns": preloaded_campaigns,
            "orchestrator_events_url": _orchestrator_events_url(),
        },
    )


@app.post("/ui/select-preloaded", response_class=HTMLResponse)
async def select_preloaded(request: Request, campaign_name: str = Form(...)) -> HTMLResponse:
    campaign_content = preloaded_campaigns.get(campaign_name)
    if not campaign_content:
        state.status_message = f"Campaign not found: {campaign_name}"
        state.status_kind = "error"
        return await _render_workspace(request, status_code=404)

    try:
        _normalize_and_set_campaign(campaign_name, campaign_content)
        state.status_message = f"Loaded preloaded campaign: {campaign_name}"
        state.status_kind = "success"
    except json.JSONDecodeError as exc:
        state.status_message = f"Invalid JSON in preloaded campaign: {exc}"
        state.status_kind = "error"
        return await _render_workspace(request, status_code=400)

    return await _render_workspace(request)


@app.post("/ui/upload", response_class=HTMLResponse)
async def upload_campaign(request: Request, campaign_file: UploadFile) -> HTMLResponse:
    if not campaign_file.filename:
        state.status_message = "No file selected."
        state.status_kind = "warning"
        return await _render_workspace(request, status_code=400)

    try:
        campaign_text = (await campaign_file.read()).decode("utf-8")
        _normalize_and_set_campaign(campaign_file.filename, campaign_text)
        state.status_message = f"Uploaded campaign: {campaign_file.filename}"
        state.status_kind = "success"
    except UnicodeDecodeError:
        state.status_message = "Campaign file must be UTF-8 text."
        state.status_kind = "error"
        return await _render_workspace(request, status_code=400)
    except json.JSONDecodeError as exc:
        state.status_message = f"Invalid campaign JSON: {exc}"
        state.status_kind = "error"
        return await _render_workspace(request, status_code=400)

    return await _render_workspace(request)


async def _post_to_orchestrator(
    path: str,
    payload: dict[str, Any] | str | None = None,
    json_content_type: str = "application/json",
) -> tuple[int, str]:
    url = f"{config.orchestrator_base_url}{path}"
    headers = {
        "Authorization": config.orchestrator_api_key,
        "Content-Type": json_content_type,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        if payload is None:
            response = await client.post(url, headers=headers)
        else:
            response = await client.post(url, headers=headers, content=json.dumps(payload))
        return response.status_code, response.text


@app.post("/ui/submit", response_class=HTMLResponse)
async def submit_campaign(
    request: Request,
    campaign_json: str = Form(...),
    regenerate_id: str | None = Form(default=None),
) -> HTMLResponse:
    try:
        campaign = json.loads(campaign_json)
    except json.JSONDecodeError as exc:
        state.status_message = f"Invalid campaign JSON: {exc}"
        state.status_kind = "error"
        return await _render_workspace(request, status_code=400)

    if regenerate_id == "on":
        campaign["id"] = str(uuid.uuid4())

    # Newer orchestrator builds require run_id in the campaign payload.
    if not campaign.get("run_id"):
        campaign["run_id"] = str(uuid.uuid4())

    state.campaign_json = json.dumps(campaign, indent=2)
    state.graph_mermaid, task_node_map = _build_mermaid(campaign)
    state.task_node_map_json = json.dumps(task_node_map)
    state.active_campaign_id = str(campaign.get("id") or "")

    try:
        status_code, response_text = await _post_to_orchestrator("/v1/orchestrator/start_campaign", campaign)
    except httpx.HTTPError as exc:
        state.status_message = f"Submit failed (network): {exc}"
        state.status_kind = "error"
        return await _render_workspace(request, status_code=502)

    if status_code >= 400:
        state.status_message = f"Submit failed ({status_code}): {response_text[:600]}"
        state.status_kind = "error"
        return await _render_workspace(request, status_code=status_code)

    response_campaign_id = response_text.strip().strip('"')
    if response_campaign_id:
        # Some orchestrator versions return run_id instead of campaign id.
        # Keep active campaign id from payload for stop requests.
        returned_token = response_campaign_id
    else:
        returned_token = ""
    state.status_message = f"Campaign submitted ({status_code}). Campaign ID: {state.active_campaign_id}"
    if returned_token and returned_token != state.active_campaign_id:
        state.status_message += f" Run ID: {returned_token}"
    state.status_kind = "success"
    return await _render_workspace(request)


@app.post("/ui/stop", response_class=HTMLResponse)
async def stop_campaign(
    request: Request,
    campaign_id: str = Form(default=""),
) -> HTMLResponse:
    candidate_campaign_id = campaign_id.strip() or state.active_campaign_id.strip()
    attempts: list[tuple[str, dict[str, Any] | str | None, str]] = []

    if candidate_campaign_id:
        attempts = [
            # Current orchestrator OpenAPI expects a raw UUID JSON string body.
            ("/v1/orchestrator/stop_campaign", candidate_campaign_id, "application/json"),
            ("/v1/orchestrator/stop_campaign", {"id": candidate_campaign_id}, "application/json"),
            ("/v1/orchestrator/stop_campaign", {"campaign_id": candidate_campaign_id}, "application/json"),
            (f"/v1/orchestrator/stop_campaign/{candidate_campaign_id}", None, "application/json"),
            (f"/v1/orchestrator/stop_campaign?id={candidate_campaign_id}", None, "application/json"),
        ]
    else:
        attempts = [("/v1/orchestrator/stop_campaign", None, "application/json")]

    errors: list[str] = []
    for path, payload, content_type in attempts:
        try:
            status_code, response_text = await _post_to_orchestrator(path, payload, json_content_type=content_type)
        except httpx.HTTPError as exc:
            errors.append(f"{path}: network error {exc}")
            continue

        if status_code < 400:
            state.status_message = f"Stop campaign request accepted ({status_code})."
            state.status_kind = "success"
            return await _render_workspace(request)

        errors.append(f"{path}: {status_code} {response_text[:240]}")

    state.status_message = "Stop failed. Attempts: " + " | ".join(errors[:4])
    state.status_kind = "error"
    return await _render_workspace(request, status_code=502)
