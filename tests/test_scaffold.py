"""Phase 0 scaffold tests: layout, imports, configs, API and the labels-isolation rule."""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import burnin

REQUIRED_PATHS = [
    "README.md",
    "CONVENTIONS.md",
    "pyproject.toml",
    "Makefile",
    "configs/generator.yaml",
    "configs/screening.yaml",
    "data/README.md",
    "src/burnin/schema.py",
    "src/burnin/ingest.py",
    "src/burnin/synth/generator.py",
    "src/burnin/synth/defects.py",
    "src/burnin/module_a/static_limits.py",
    "src/burnin/module_a/dynamic_pat.py",
    "src/burnin/module_a/delta_limits.py",
    "src/burnin/module_a/multivariate.py",
    "src/burnin/module_a/lot_checks.py",
    "src/burnin/module_b/baselines.py",
    "src/burnin/module_b/quantile_model.py",
    "src/burnin/module_b/conformal.py",
    "src/burnin/module_b/safety_slope.py",
    "src/burnin/fusion.py",
    "src/burnin/explain.py",
    "src/burnin/evaluate.py",
    "src/burnin/reports.py",
    "api/main.py",
    "web/package.json",
    "scripts/demo.sh",
]


@pytest.mark.parametrize("rel", REQUIRED_PATHS)
def test_repo_layout(repo_root: Path, rel: str) -> None:
    assert (repo_root / rel).exists(), f"missing {rel}"


def test_all_burnin_modules_import() -> None:
    """Every module in the package imports cleanly (stubs included)."""
    names = [m.name for m in pkgutil.walk_packages(burnin.__path__, prefix="burnin.")]
    assert "burnin.module_a.dynamic_pat" in names
    for name in names:
        importlib.import_module(name)


def test_configs_parse(repo_root: Path) -> None:
    gen = yaml.safe_load((repo_root / "configs/generator.yaml").read_text())
    scr = yaml.safe_load((repo_root / "configs/screening.yaml").read_text())
    assert isinstance(gen["seed"], int)
    assert gen["lots"]["size"]["min"] <= 20 <= 500 <= gen["lots"]["size"]["max"]
    assert abs(sum(gen["defects"]["mix"].values()) - 1.0) < 1e-9
    assert scr["robust_stats"]["mad_scale"] == pytest.approx(1.4826)
    assert scr["evaluation"]["recall_target"] >= 0.99
    assert (
        scr["evaluation"]["costs"]["false_negative"] > scr["evaluation"]["costs"]["false_positive"]
    )
    assert scr["module_b"]["safety_slope"]["flag_on"] == "upper_bound"
    bins = scr["module_b"]["conformal"]["mondrian_bins_z24"]
    assert all(isinstance(b, float) for b in bins), bins  # YAML spells infinity `.inf`
    assert bins == sorted(bins)


def test_api_health() -> None:
    from api.main import app

    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == burnin.__version__
    assert body["offline"] is True
    assert body["data_source"] in {"none", "synthetic", "uploaded"}


def test_api_has_no_cdn_backed_pages() -> None:
    """Swagger UI / ReDoc load JS, CSS and fonts from CDNs, so they are off (CONVENTIONS rule 6)."""
    from api.main import app

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "cdn." not in r.text and "googleapis" not in r.text


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    from burnin.cli import main

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == burnin.__version__


# The synth package WRITES labels (it is not on the screening path); evaluate.py and the training
# entry points in cli.py READ them. Nothing else under src/ or api/ may mention them.
LABEL_READERS_ALLOWED = ("burnin/evaluate.py", "burnin/cli.py", "burnin/synth/")
_LABELS_PATTERN = re.compile(r"labels\.csv|is_defective|defect_type", re.IGNORECASE)


def test_production_code_never_reads_labels(repo_root: Path) -> None:
    """Ground-truth labels are for evaluation and training only (CONVENTIONS.md, rule 3).

    Any production module that mentions the labels file or its columns is a leak of ground truth
    into the screening path. Docstrings count too: keep the rule visible by keeping it out.
    """
    offenders = []
    for path in (repo_root / "src").rglob("*.py"):
        rel = path.relative_to(repo_root / "src").as_posix()
        if rel.startswith(LABEL_READERS_ALLOWED):
            continue
        if _LABELS_PATTERN.search(path.read_text()):
            offenders.append(rel)
    for path in (repo_root / "api").rglob("*.py"):
        if _LABELS_PATTERN.search(path.read_text()):
            offenders.append(path.relative_to(repo_root).as_posix())
    assert not offenders, f"production modules reference ground-truth labels: {offenders}"
