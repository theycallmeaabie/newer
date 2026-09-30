"""FastAPI application for burn-in screening.

Phase 0: health endpoint, CORS for the Vite dev server, and static serving of the built
dashboard (web/dist) when it exists. Screening endpoints arrive in Phase 5.

Runs fully offline: no outbound requests, no telemetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from burnin import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = REPO_ROOT / "web" / "dist"


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    offline: bool
    # Surfaces in the dashboard banner and in reports: synthetic results must be labelled as such.
    data_source: Literal["none", "synthetic", "uploaded"]


app = FastAPI(
    title="Burn-in Screening API",
    version=__version__,
    description="Dynamic outlier detection and drift prediction for component burn-in (PS 26170).",
    # Swagger UI / ReDoc pull JS, CSS and fonts from CDNs, which breaks the offline rule.
    # The schema stays available at /openapi.json.
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", version=__version__, offline=True, data_source="none")


# Serve the built dashboard from the same process for the one-command demo.
# Mounted last so it never shadows /api routes.
if WEB_DIST.is_dir():  # pragma: no cover - depends on a prior `npm run build`
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
