#!/usr/bin/env python3
"""Debug timing data paths"""

import glob
import json
import os

print("Current working directory:", os.getcwd())
print("\nLooking for student response files...")

# Try different path patterns
patterns = [
    "student_responses/student_resp_*.json",
    "seed_oil_evaluation/student_responses/student_resp_*.json",
    "**/student_responses/student_resp_*.json"
]

for pattern in patterns:
    files = glob.glob(pattern, recursive=True)
    print(f"Pattern '{pattern}': {len(files)} files found")
    if files:
        print(f"  First file: {files[0]}")
        # Try to read timing data from first file
        try:
            with open(files[0], 'r') as f:
                data = json.load(f)
            response_time = data.get('response_time_seconds')
            student_model = data.get('student_model')
            print(f"  Sample data: model={student_model}, time={response_time}s")
        except Exception as e:
            print(f"  Error reading file: {e}")

print("\nChecking absolute paths...")
abs_pattern = "/Users/t/dev/area51/codex3-qa-prompt/seed_oil_evaluation/student_responses/student_resp_*.json"
abs_files = glob.glob(abs_pattern)
print(f"Absolute path pattern: {len(abs_files)} files found")

if abs_files:
    try:
        with open(abs_files[0], 'r') as f:
            data = json.load(f)
        response_time = data.get('response_time_seconds')
        student_model = data.get('student_model')
        print(f"  Sample data: model={student_model}, time={response_time}s")
    except Exception as e:
        print(f"  Error reading file: {e}")