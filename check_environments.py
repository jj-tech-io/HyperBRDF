# 
import os
import numpy as np
import torch
import sys

def verify_data_setup():
    """Verify the MERL dataset setup"""
    print("=== Data Directory Structure ===")
    
    # Check data directory
    data_dir = "./data"
    print(f"\nChecking data directory: {data_dir}")
    if os.path.exists(data_dir):
        print("✓ Data directory exists")
        files = os.listdir(data_dir)
        print(f"Files found: {files}")
        
        # Check MERL median file
        merl_median = os.path.join(data_dir, "merl_median.binary")
        if os.path.exists(merl_median):
            print(f"\n✓ Found MERL median file: {merl_median}")
            try:
                # Try to read the binary file
                with open(merl_median, 'rb') as f:
                    data = np.fromfile(f, dtype=np.float32)
                print(f"  - File size: {len(data)} float32 values")
                print(f"  - First few values: {data[:5]}")
            except Exception as e:
                print(f"✗ Error reading MERL median file: {e}")
        else:
            print(f"\n✗ Missing MERL median file: {merl_median}")
            
        # Check brdf.fullbin
        brdf_file = os.path.join(data_dir, "brdf.fullbin")
        if os.path.exists(brdf_file):
            print(f"\n✓ Found BRDF file: {brdf_file}")
            try:
                size = os.path.getsize(brdf_file)
                print(f"  - File size: {size:,} bytes")
            except Exception as e:
                print(f"✗ Error accessing BRDF file: {e}")
        else:
            print(f"\n✗ Missing BRDF file: {brdf_file}")
    else:
        print("✗ Data directory not found!")

    # Check output directories
    print("\n=== Output Directory Structure ===")
    output_dirs = [
        "results/merl",
        "../binary_output"
    ]
    for d in output_dirs:
        if os.path.exists(d):
            print(f"✓ Output directory exists: {d}")
        else:
            try:
                os.makedirs(d, exist_ok=True)
                print(f"Created output directory: {d}")
            except Exception as e:
                print(f"✗ Error creating directory {d}: {e}")

if __name__ == "__main__":
    verify_data_setup()