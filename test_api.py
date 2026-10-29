# Programmatic test script simulating production client queries
import json
import pandas as pd
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def run_suite():
    results = []
    print("\n--- Initiating Intermediate Model Service Test Suite ---\n")
    
    # 1. Health check test
    r = client.get("/")
    print(f"[Health Check] Response: {r.status_code} - {r.json()}")
    results.append({"test": "health_check", "status": "PASS" if r.status_code == 200 else "FAIL"})

    # 2. Perfect valid query
    valid_payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
    }
    r = client.post("/predict", json=valid_payload)
    print(f"[Valid Inference] Response: {r.status_code} - {r.json()}")
    results.append({"test": "valid_inference", "status": "PASS" if r.status_code == 200 else "FAIL"})

    # 3. Invalid geographic boundary (Expected validation failure)
    invalid_geo_payload = valid_payload.copy()
    invalid_geo_payload["latitude"] = 55.0  # Outside CA bounds
    r = client.post("/predict", json=invalid_geo_payload)
    print(f"[Boundary Validation Error Check] Response: {r.status_code} - {r.json().get('detail', 'Validation Error')[:100]}")
    results.append({"test": "boundary_check", "status": "PASS" if r.status_code == 422 else "FAIL"})

    # 4. Negative physical dimension check
    invalid_neg_payload = valid_payload.copy()
    invalid_neg_payload["total_rooms"] = -10.0
    r = client.post("/predict", json=invalid_neg_payload)
    print(f"[Negative Dimension Check] Response: {r.status_code} - {r.json().get('detail', 'Validation Error')[:100]}")
    results.append({"test": "negative_physical_check", "status": "PASS" if r.status_code == 422 else "FAIL"})

    print("\n--- Test Suite Summary ---")
    for res in results:
        print(f"Test: {res['test']} -> {res['status']}")

if __name__ == "__main__":
    run_suite()
