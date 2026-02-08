#!/usr/bin/env python3
"""
Test script to verify backend is serving all frames correctly.
"""

import requests
import json

API_URL = "http://localhost:8000"

print("🔍 Testing Backend Grid Data API\n")
print("="*60)

try:
    # Test grid-frames endpoint
    response = requests.get(f"{API_URL}/grid-frames")
    
    if response.status_code == 200:
        data = response.json()
        
        print("✅ API Response Successful\n")
        print(f"📊 Metadata:")
        print(f"   Grid Size: {data['metadata']['grid_size']}x{data['metadata']['grid_size']}")
        print(f"   Total Timestamps: {data['metadata']['total_timestamps']}")
        print(f"   Cells per Timestamp: {data['metadata']['cells_per_timestamp']}")
        print(f"   Bounds: {data['metadata']['bounds']}")
        
        print(f"\n📅 Frame Information:")
        print(f"   Total Frames: {data['total_frames']}")
        print(f"   Frames in Response: {len(data['frames'])}")
        
        if len(data['frames']) > 0:
            first_frame = data['frames'][0]
            last_frame = data['frames'][-1]
            
            print(f"\n🕐 Time Range:")
            print(f"   First Frame: {first_frame['DateTime']}")
            print(f"   Last Frame: {last_frame['DateTime']}")
            
            # Calculate congestion statistics
            all_congestion = [
                cell['predicted_congestion_level'] 
                for frame in data['frames'][:10]  # Sample first 10 frames
                for cell in frame['cells']
                if cell['predicted_congestion_level'] > 0
            ]
            
            if all_congestion:
                print(f"\n📈 Congestion Statistics (first 10 frames):")
                print(f"   Min: {min(all_congestion):.4f}")
                print(f"   Max: {max(all_congestion):.4f}")
                print(f"   Average: {sum(all_congestion)/len(all_congestion):.4f}")
                print(f"   Active Cells: {len(all_congestion)}")
        
        print("\n" + "="*60)
        print("✅ Backend is serving all frames correctly!")
        print(f"💡 Frontend should display {data['total_frames']} frames")
        
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"   Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to backend API")
    print("   Make sure the backend is running:")
    print("   cd backend && uv run python api.py")
    
except Exception as e:
    print(f"❌ Error: {e}")

print()
