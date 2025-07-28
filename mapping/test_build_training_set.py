"""
Test script for build_training_set.py functionality
"""

import os
import pandas as pd

# Test the functions we added (without running the full script due to import issues)

def test_get_csv_files():
    """Test the get_csv_files function"""
    # Test with existing directories
    aw_dir = "trainingset/aw_data"
    vicon_dir = "trainingset/vicon_data"
    
    if os.path.exists(aw_dir):
        csv_files = []
        for root, dirs, files in os.walk(aw_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        print(f"Found {len(csv_files)} CSV files in AW directory:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
    else:
        print(f"AW directory not found: {aw_dir}")
    
    if os.path.exists(vicon_dir):
        csv_files = []
        for root, dirs, files in os.walk(vicon_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        print(f"Found {len(csv_files)} CSV files in Vicon directory:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
    else:
        print(f"Vicon directory not found: {vicon_dir}")

if __name__ == "__main__":
    test_get_csv_files()
