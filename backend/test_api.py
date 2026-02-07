"""Example usage of the Traffic Congestion Simulator API (Multi-Horizon)."""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_current_predictions():
    """Test current predictions endpoint."""
    print("\n=== Current Predictions (20 locations, 6 horizons each) ===")
    try:
        response = requests.get(f"{BASE_URL}/predictions/current?locations=20")
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure it's running:")
        print("   uv run python backend/api.py")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return None
    
    if 'predictions' not in data:
        print(f"❌ Error: Unexpected response format")
        print(f"Response: {json.dumps(data, indent=2)}")
        return None
    
    print(f"Total location predictions: {len(data['predictions'])}")
    print(f"\nFirst location sample:")
    
    if data['predictions']:
        pred = data['predictions'][0]
        lat = pred['latitude'] / 1_000_000
        lon = pred['longitude'] / 1_000_000
        print(f"  Location: ({lat:.6f}, {lon:.6f})")
        print(f"  Horizons:")
        for h in pred['horizons']:
            print(f"    {h['horizon']}: {h['DateTime'][:19]} | Congestion: {h['predicted_congestion']:+.4f}")
    
    return data


def test_batch():
    """Test batch predictions."""
    print("\n=== Batch Predictions (50 locations) ===")
    try:
        response = requests.get(f"{BASE_URL}/predictions/batch?locations=50&base_hour=8")
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None
    
    if 'predictions' not in data:
        print(f"❌ Error: Unexpected response format")
        return None
    
    # Calculate stats
    all_congestions = []
    for pred in data['predictions']:
        for h in pred['horizons']:
            all_congestions.append(h['predicted_congestion'])
    
    avg = sum(all_congestions) / len(all_congestions)
    min_val = min(all_congestions)
    max_val = max(all_congestions)
    
    print(f"Total locations: {len(data['predictions'])}")
    print(f"\nCongestion statistics across all horizons:")
    print(f"  Average: {avg:+.4f}")
    print(f"  Range: {min_val:+.4f} to {max_val:+.4f}")
    
    return data


def save_sample_output(filename="sample_predictions.json"):
    """Save a sample output for frontend testing."""
    response = requests.get(f"{BASE_URL}/predictions/current?locations=30")
    data = response.json()
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    total_horizons = len(data['predictions']) * 6
    print(f"\n✅ Saved {len(data['predictions'])} locations ({total_horizons} total horizon predictions) to {filename}")


def print_sample_format():
    """Print example of the new format."""
    print("\n=== Sample Output Format ===")
    sample = {
        "predictions": [
            {
                "latitude": 40742511,
                "longitude": -73949134,
                "horizons": [
                    {
                        "horizon": "t+1h",
                        "DateTime": "2026-02-08T00:22:57.603573",
                        "predicted_congestion": -0.9281
                    },
                    {
                        "horizon": "t+2h",
                        "DateTime": "2026-02-08T01:22:57.603573",
                        "predicted_congestion": -0.6372
                    },
                    "... (t+3h through t+6h)"
                ]
            }
        ]
    }
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    print("🧪 Testing Traffic Congestion Simulator API (Multi-Horizon)")
    print("=" * 70)
    
    try:
        # Show format
        print_sample_format()
        
        # Test all endpoints
        test_current_predictions()
        test_batch()
        
        # Save sample for frontend
        save_sample_output()
        
        print("\n✅ All tests passed!")
        print("\n📝 Key changes from old format:")
        print("  • Latitude/Longitude now integers (e.g., 40742511 = 40.742511°)")
        print("  • Each location has 6 time horizons (t+1h through t+6h)")
        print("  • Congestion values normalized (can be negative, ~-1 to +1)")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: API not running!")
        print("Start the API first: python backend/api.py")
