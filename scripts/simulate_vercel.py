"""
Simulate Vercel environment:
1. Sets VERCEL=1.
2. Starts uvicorn / TestClient with read-only data simulation.
3. Requests /docs, /api/logs, and /api/eval/report/business_deal.
4. Confirms HTTP 200 and that /api/logs returns the two files in results/.
"""
import os
import sys

# Set VERCEL environment variable before imports
os.environ["VERCEL"] = "1"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
print(f"VERCEL environment detected.")
print(f"settings.data_dir:   {settings.data_dir}")
print(f"settings.chroma_dir: {settings.chroma_dir}")
print(f"settings.sqlite_path:{settings.sqlite_path}")

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# 1. GET /docs
resp_docs = client.get("/docs")
print(f"\n[1] GET /docs -> HTTP {resp_docs.status_code}")
assert resp_docs.status_code == 200, f"Expected 200, got {resp_docs.status_code}"

# 2. GET /api/logs
resp_logs = client.get("/api/logs")
print(f"\n[2] GET /api/logs -> HTTP {resp_logs.status_code}")
assert resp_logs.status_code == 200, f"Expected 200, got {resp_logs.status_code}"
logs_data = resp_logs.json()
log_files = [f["filename"] for f in logs_data.get("logs", [])]
print(f"Logs returned: {log_files}")
assert "benchmark_business_deal.jsonl" in log_files, "benchmark_business_deal.jsonl missing from /api/logs"
assert "reservation_sweep.jsonl" in log_files, "reservation_sweep.jsonl missing from /api/logs"
print("SUCCESS: Found both benchmark_business_deal.jsonl and reservation_sweep.jsonl in results/")

# 3. GET /api/eval/report/business_deal
resp_report = client.get("/api/eval/report/business_deal")
print(f"\n[3] GET /api/eval/report/business_deal -> HTTP {resp_report.status_code}")
assert resp_report.status_code == 200, f"Expected 200, got {resp_report.status_code}"
report_data = resp_report.json()
methods = list(report_data.get("summary", {}).keys())
print(f"Methods in report: {methods}")
assert len(methods) > 0, "No methods found in report summary"
print("SUCCESS: /api/eval/report/business_deal computed report from results/ successfully!")

print("\nALL VERCEL SIMULATION CHECKS PASSED (All returned 200 OK).")
