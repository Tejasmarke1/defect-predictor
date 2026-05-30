"""
scripts/day5_run.py
-------------------
Validates model files, starts the API server, waits for it to be healthy,
runs smoke tests against all endpoints, and prints a timing summary.

Steps:
  1. Validate all required model files exist
  2. Start uvicorn in a subprocess
  3. Poll /health until ready (timeout 30s)
  4. Smoke test: GET /health
  5. Smoke test: POST /analyze  (pallets/flask, top_k=5)
  6. Smoke test: GET  /analyze/{job_id}
  7. Smoke test: GET  /explain/{job_id}/{top_file}
  8. Smoke test: GET  /experiments
  9. Print timing table + Day 5 git commit command

Run:
    python scripts/day5_run.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_URL = "http://localhost:8000"
REQUIRED_MODELS = [
    ROOT / "models" / "xgboost_defect_predictor.json",
    ROOT / "models" / "gnn_model.pt",
    ROOT / "models" / "hybrid_model.pkl",
    ROOT / "models" / "model_meta.json",
    ROOT / "data"   / "processed" / "gnn_embeddings.pkl",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _check_models() -> None:
    """Step 1 — fail fast if any required model file is missing."""
    _print_section("Step 1: Validating model files")
    missing = [p for p in REQUIRED_MODELS if not p.exists()]
    for p in REQUIRED_MODELS:
        mark = "✓" if p.exists() else "✗ MISSING"
        print(f"  {mark}  {p.relative_to(ROOT)}")
    if missing:
        print(f"\n  ERROR: {len(missing)} model file(s) missing.")
        print("  Run scripts/day4_run.py first to train and save models.")
        sys.exit(1)
    print("\n  All model files present ✓")


def _start_server() -> subprocess.Popen:
    """Step 2 — launch uvicorn in a subprocess."""
    _print_section("Step 2: Starting API server")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1",
            "--log-level", "warning",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"  uvicorn started (PID {proc.pid})")
    return proc


def _wait_for_health(timeout: int = 30) -> None:
    """Step 3 — poll /health until 200 or timeout."""
    import requests

    _print_section("Step 3: Waiting for API to be healthy")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                print(f"  API ready — status={data['status']}")
                return
        except Exception:
            pass
        time.sleep(2)
        print("  ... waiting")

    print(f"  ERROR: API did not become healthy within {timeout}s")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _test_health() -> dict:
    """Step 4 — GET /health"""
    import requests
    _print_section("Smoke test 4: GET /health")
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    print(f"  status          : {data['status']}")
    print(f"  models_loaded   : {data['models_loaded']}")
    print(f"  uptime_seconds  : {data['uptime_seconds']:.1f}s")
    print(f"  mlflow_connected: {data['mlflow_connected']}")
    print(f"  response time   : {elapsed:.0f}ms")
    assert data["status"] in ("healthy", "degraded"), \
        f"Unexpected status: {data['status']}"
    return data


def _test_analyze() -> dict:
    """Step 5 — POST /analyze"""
    import requests
    _print_section("Smoke test 5: POST /analyze (pallets/flask, top_k=5)")
    payload = {
        "repo_url":   "https://github.com/pallets/flask",
        "since_days": 180,
        "top_k":      5,
        "use_hybrid": True,
    }
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=300)
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
    data = r.json()
    print(f"  job_id              : {data['job_id']}")
    print(f"  status              : {data['status']}")
    print(f"  model_used          : {data['model_used']}")
    print(f"  total_files_analyzed: {data['total_files_analyzed']}")
    print(f"  buggy_predicted     : {data['buggy_files_predicted']}")
    print(f"  mining_time_ms      : {data['mining_time_ms']:.0f}ms")
    print(f"  feature_time_ms     : {data['feature_time_ms']:.0f}ms")
    print(f"  prediction_time_ms  : {data['prediction_time_ms']:.0f}ms")
    print(f"  analysis_time_ms    : {data['analysis_time_ms']:.0f}ms")
    print(f"  total round-trip    : {elapsed:.0f}ms")
    if data.get("warnings"):
        print(f"  warnings            : {data['warnings']}")
    print(f"\n  Top {len(data['top_k_results'])} risky files:")
    for r_ in data["top_k_results"]:
        print(f"    [{r_['rank']:>2}] {r_['risk_label']:<6} {r_['risk_score']:.3f}  {r_['file_path']}")
    assert data["status"] == "completed", f"Job status: {data['status']}"
    return data


def _test_get_analysis(job_id: str) -> dict:
    """Step 6 — GET /analyze/{job_id}"""
    import requests
    _print_section(f"Smoke test 6: GET /analyze/{job_id}")
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/analyze/{job_id}", timeout=10)
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    print(f"  status   : {data['status']}")
    print(f"  response : {elapsed:.0f}ms")
    assert data["status"] == "completed"
    return data


def _test_explain(job_id: str, file_path: str) -> dict:
    """Step 7 — GET /explain/{job_id}/{file_path}"""
    import requests
    _print_section(f"Smoke test 7: GET /explain/{job_id}/{file_path}")
    t0 = time.time()
    r = requests.get(
        f"{BASE_URL}/explain/{job_id}/{file_path}",
        timeout=30,
    )
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
    data = r.json()
    print(f"  file_path             : {data['file_path']}")
    print(f"  risk_score            : {data['risk_score']}")
    print(f"  risk_label            : {data['risk_label']}")
    print(f"  plain_english_summary : {data['plain_english_summary']}")
    print(f"  similar_files         : {data['similar_files']}")
    print(f"  embedding_neighbors   : {data['embedding_neighbors']}")
    print(f"  shap features         : {len(data['shap_waterfall'])}")
    print(f"  response time         : {elapsed:.0f}ms")
    assert data["plain_english_summary"], "plain_english_summary is empty"
    return data


def _test_experiments() -> dict:
    """Step 8 — GET /experiments"""
    import requests
    _print_section("Smoke test 8: GET /experiments")
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/experiments", timeout=90)
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
    data = r.json()
    print(f"  total_runs  : {data['total_runs']}")
    print(f"  best_run    : {data.get('best_run', {}).get('run_name', 'N/A')} "
          f"AUC={data.get('best_run', {}).get('auc', 'N/A')}")
    print(f"  response    : {elapsed:.0f}ms")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Day 5 — API Smoke Test Runner")
    print("=" * 60)

    # Step 1: validate files
    _check_models()

    # Step 2: start server
    proc = _start_server()

    results: dict[str, str] = {}
    timings: dict[str, float] = {}

    try:
        # Step 3: wait for health
        _wait_for_health(timeout=30)

        # Step 4-8: smoke tests
        tests = [
            ("GET /health",           _test_health),
            ("POST /analyze",         _test_analyze),
        ]

        analyze_data = None
        for name, fn in tests:
            t0 = time.time()
            try:
                result = fn()
                results[name] = "PASS"
                timings[name] = (time.time() - t0) * 1000
                if name == "POST /analyze":
                    analyze_data = result
            except Exception as e:
                results[name] = f"FAIL: {e}"
                timings[name] = (time.time() - t0) * 1000
                print(f"  ✗ {name}: {e}")

        # Steps 6-8 require a completed analyze job
        if analyze_data and analyze_data.get("status") == "completed":
            job_id   = analyze_data["job_id"]
            top_file = (analyze_data["top_k_results"][0]["file_path"]
                        if analyze_data["top_k_results"] else "")

            follow_up = [
                (f"GET /analyze/{job_id}",
                 lambda: _test_get_analysis(job_id)),
                (f"GET /explain/{job_id}/{top_file}",
                 lambda: _test_explain(job_id, top_file)),
                ("GET /experiments",
                 _test_experiments),
            ]
            for name, fn in follow_up:
                t0 = time.time()
                try:
                    fn()
                    results[name] = "PASS"
                    timings[name] = (time.time() - t0) * 1000
                except Exception as e:
                    results[name] = f"FAIL: {e}"
                    timings[name] = (time.time() - t0) * 1000
                    print(f"  ✗ {name}: {e}")

    finally:
        proc.terminate()
        proc.wait()
        print("\n  Server stopped.")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMOKE TEST RESULTS")
    print("=" * 60)
    all_pass = True
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        ms   = timings.get(name, 0)
        print(f"  {icon} {name:<40} {status:<8} {ms:.0f}ms")
        if status != "PASS":
            all_pass = False

    print("=" * 60)
    print(f"  Result: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 60)

    print("\nDay 5 git commit command:")
    print(
        "git add src/api/ scripts/day5_run.py notebooks/day5_api_test.ipynb "
        "Dockerfile docker-compose.yml requirements.txt && "
        'git commit -m "Day 6: FastAPI — /analyze /explain /health /experiments | '
        'hybrid AUC 0.8590 | Docker ready"'
    )
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()