#!/usr/bin/env python3
"""
Automated ML Pipeline Runner
Runs the complete ML pipeline: inference → forecasts → grid generation

Usage:
    python run_pipeline.py                  # Use latest CSV from backend/uploads
    python run_pipeline.py path/to/file.csv # Use specific CSV file
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, cwd=None, description=""):
    """Run a command and return output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")
    print()
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(result.stderr)
        sys.exit(1)
    
    print(f"✅ {description} completed successfully")
    return result


def main():
    print("\n" + "="*60)
    print("🚀 ML PIPELINE AUTOMATION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get workspace root
    script_dir = Path(__file__).parent
    ml_dir = script_dir / "ml"
    backend_dir = script_dir / "backend"
    outputs_dir = backend_dir / "outputs"
    uploads_dir = backend_dir / "uploads"
    
    # Check if CSV argument provided
    csv_arg = []
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            print(f"❌ Error: CSV file not found: {csv_path}")
            sys.exit(1)
        csv_arg = ["--csv", str(csv_path)]
        print(f"📄 Using specified CSV: {csv_path}")
    else:
        # Check if any CSV exists in uploads
        csv_files = list(uploads_dir.glob("*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            print(f"📄 Using latest CSV from uploads: {latest_csv.name}")
        else:
            print(f"📄 No CSV specified, pipeline.py will auto-detect")
    
    # Step 1: Run ML Inference
    run_command(
        [sys.executable, "pipeline.py"] + csv_arg,
        cwd=ml_dir,
        description="Step 1/2: ML Inference Pipeline"
    )
    
    # Step 2: Generate Grid Data
    run_command(
        [sys.executable, "process_forecasts.py"],
        cwd=outputs_dir,
        description="Step 2/2: Grid Data Generation"
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)
    
    try:
        # Count forecasts
        with open(outputs_dir / "forecasts.json") as f:
            forecast_data = json.load(f)
            forecast_count = len(forecast_data.get("outputs", []))
        
        # Count grid frames
        with open(outputs_dir / "forecasts_grid.json") as f:
            grid_data = json.load(f)
            frame_count = len(grid_data.get("frames", []))
            grid_size = grid_data.get("metadata", {}).get("grid_size", "?")
        
        print(f"✅ Forecast records: {forecast_count:,}")
        print(f"✅ Grid frames: {frame_count}")
        print(f"✅ Grid size: {grid_size}x{grid_size}")
        print(f"\n📁 Output files:")
        print(f"   • {outputs_dir / 'forecasts.json'}")
        print(f"   • {outputs_dir / 'forecasts_grid.json'}")
        
    except Exception as e:
        print(f"⚠️  Could not read output files: {e}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Frontend will automatically load the new data")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        sys.exit(1)
