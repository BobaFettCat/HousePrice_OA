# Programmatic test script confirming caching and latency logging benefits
import time
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def run_suite():
    print("\n--- Initiating Milestone 3 Telemetry & Optimisation Verification ---")
    
    payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
    }

    # Call 1: Cache Miss (Direct Inference)
    t0 = time.time()
    r1 = client.post("/predict", json=payload)
    t1 = time.time()
    lat1 = (t1 - t0) * 1000.0
    print(f"\n[Inference 1 - Miss] Status: {r1.status_code} | Latency: {lat1:.2f}ms | Body: {r1.json()}")

    # Call 2: Cache Hit (Instant response expected)
    t0 = time.time()
    r2 = client.post("/predict", json=payload)
    t2 = time.time()
    lat2 = (t2 - t0) * 1000.0
    print(f"[Inference 2 - Hit ] Status: {r2.status_code} | Latency: {lat2:.2f}ms | Body: {r2.json()}")

    # Verification check
    print(f"\nLatency reduction due to caching optimization: {lat1 - lat2:.2f}ms")
    assert r2.json()["cached_response"] is True, "Error: Cache did not return flagged response."

    # Call Health endpoint to verify dashboard telemetry aggregation
    r_health = client.get("/")
    print(f"\n[Health System Telemetry Check]:\n{r_health.json()}")

if __name__ == "__main__":
    run_suite()
