"""
Verification test suite for Crop Recommendation System.
Validates:
1. API endpoints (/health, /model-info, /v1/recommend)
2. ML predictions vary dynamically based on field inputs (no static/hardcoded values)
3. Sorted probabilities and confidence matching top probability
4. Strict validation against out-of-range and missing inputs
"""
import urllib.request
import json
import sys

BASE_URL = "http://localhost:8000"

def post_json(path, data):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))

def get_json(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def run_tests():
    print("=== 1. Health & Model Info Tests ===")
    status, health = get_json("/health")
    assert status == 200 and health.get("status") == "healthy", f"Health check failed: {health}"
    print("PASS: /health returned 200 OK")

    status, info = get_json("/model-info")
    assert status == 200 and "model" in info, f"Model info failed: {info}"
    print(f"PASS: /model-info returned 200 OK (model: {info.get('model')}, test acc: {info.get('test_accuracy')})")

    print("\n=== 2. Dynamic Prediction Tests with Different Field Conditions ===")
    test_cases = [
        ("Rice Profile", {"N": 90, "P": 42, "K": 43, "temperature": 21.0, "humidity": 82.0, "ph": 6.5, "rainfall": 203.0}, "rice"),
        ("Cotton Profile", {"N": 120, "P": 40, "K": 20, "temperature": 25.0, "humidity": 80.0, "ph": 6.8, "rainfall": 80.0}, "cotton"),
        ("Coffee Profile", {"N": 100, "P": 30, "K": 30, "temperature": 25.0, "humidity": 60.0, "ph": 6.8, "rainfall": 160.0}, "coffee"),
        ("Grapes Profile", {"N": 20, "P": 130, "K": 200, "temperature": 24.0, "humidity": 80.0, "ph": 6.0, "rainfall": 70.0}, "grapes"),
        ("Kidneybeans Profile", {"N": 20, "P": 68, "K": 20, "temperature": 20.0, "humidity": 22.0, "ph": 5.7, "rainfall": 105.0}, "kidneybeans"),
        ("Maize Profile", {"N": 78, "P": 48, "K": 20, "temperature": 22.0, "humidity": 65.0, "ph": 6.2, "rainfall": 85.0}, "maize"),
    ]

    seen_crops = set()
    for name, payload, expected_crop in test_cases:
        status, res = post_json("/v1/recommend", payload)
        assert status == 200, f"Request failed for {name}: {res}"
        best = res["recommended_crop"]
        conf = res["confidence"]
        top_preds = res["top_predictions"]
        alternatives = res["alternatives"]
        
        seen_crops.add(best)
        print(f"PASS: {name} -> Recommended: {best} ({conf*100:.1f}%) | Alternatives: {[a['crop'] for a in alternatives]}")

        # Check sorted order
        probs = [p["probability"] for p in top_preds]
        assert probs == sorted(probs, reverse=True), f"Probabilities not sorted descending: {probs}"
        assert abs(conf - probs[0]) < 1e-4, f"Confidence {conf} does not match top probability {probs[0]}"

    assert len(seen_crops) > 1, f"Model always returned the same crop! Observed: {seen_crops}"
    print(f"\nPASS: Verified dynamic predictions across {len(seen_crops)} distinct crop types!")

    print("\n=== 3. Validation & Boundary Tests ===")
    # Missing field
    status, res = post_json("/v1/recommend", {"N": 90, "P": 42})
    assert status == 422, f"Expected 422 for missing fields, got {status}: {res}"
    print("PASS: Missing fields correctly rejected with 422")

    # Out of range (temperature > 55)
    status, res = post_json("/v1/recommend", {"N": 90, "P": 42, "K": 43, "temperature": 99.0, "humidity": 82.0, "ph": 6.5, "rainfall": 203.0})
    assert status == 422, f"Expected 422 for out-of-range temperature, got {status}: {res}"
    print("PASS: Out-of-range temperature correctly rejected with 422")

    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
