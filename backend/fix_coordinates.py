"""
Script to convert latitude/longitude from integer format to decimal format.
Example: 19085912 -> 19.085912
"""

import json
from pathlib import Path

def fix_coordinates():
    """Convert integer coordinates to decimal format with proper placement."""
    
    forecasts_file = Path(__file__).parent / "forecasts.json"
    
    print("📂 Loading forecasts.json...")
    with open(forecasts_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Found {len(data['outputs'])} records")
    
    # Convert coordinates
    converted_count = 0
    for record in data['outputs']:
        lat = record['latitude']
        lon = record['longitude']
        
        # Check if they're in integer format (> 180 means not decimal)
        if lat > 180 or lat < -180:
            # Convert: divide by 1,000,000 to get decimal
            record['latitude'] = lat / 1_000_000
            converted_count += 1
        
        if lon > 180 or lon < -180:
            # Convert: divide by 1,000,000 to get decimal
            record['longitude'] = lon / 1_000_000
    
    print(f"✅ Converted {converted_count} coordinate values")
    
    # Get sample before/after
    sample = data['outputs'][0]
    print(f"\n📍 Sample coordinate: lat={sample['latitude']}, lon={sample['longitude']}")
    
    # Save back to file
    print("\n💾 Saving updated forecasts.json...")
    with open(forecasts_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Done! Coordinates have been converted to decimal format.")
    print(f"\n📌 Unique locations:")
    locations = set((d['latitude'], d['longitude']) for d in data['outputs'])
    for lat, lon in sorted(locations):
        print(f"   {lat}, {lon}")

if __name__ == "__main__":
    fix_coordinates()
