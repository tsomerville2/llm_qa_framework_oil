#!/usr/bin/env python3
"""Test timing visualization functionality"""

import sys
import os
sys.path.append('seed_oil_evaluation')

from scoring_dashboard import ScoringDashboard

def test_timing_data():
    print("Testing timing data collection...")
    dashboard = ScoringDashboard()
    timing_data = dashboard.get_timing_data()
    
    print(f"Found timing data for {len(timing_data)} models:")
    for model, times in timing_data.items():
        print(f"  {model}: {len(times)} measurements, avg: {sum(times)/len(times):.3f}s")
    
    return timing_data

def test_timing_visualization():
    print("\nTesting timing visualization methods...")
    dashboard = ScoringDashboard()
    
    # Test the console initialization
    print(f"Console available: {dashboard.console is not None}")
    
    # Test plotext availability
    try:
        import plotext as plt
        print("Plotext available: True")
    except ImportError:
        print("Plotext available: False")
    
    timing_data = dashboard.get_timing_data()
    if timing_data:
        print("✅ Timing data collection working")
        
        # Test if the timing analysis menu methods exist
        if hasattr(dashboard, 'show_timing_analysis_menu'):
            print("✅ Timing analysis menu method exists")
        if hasattr(dashboard, 'show_multi_model_timing_comparison'):
            print("✅ Multi-model comparison method exists")
        if hasattr(dashboard, 'show_single_model_timing_analysis'):
            print("✅ Single-model analysis method exists")
    else:
        print("❌ No timing data found")

if __name__ == "__main__":
    test_timing_data()
    test_timing_visualization()