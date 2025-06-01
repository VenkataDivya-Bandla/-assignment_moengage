#!/usr/bin/env python3
"""
Test script to verify the analyzer works independently
"""

import subprocess
import sys
import json

def test_analyzer():
    """Test the analyzer with a sample URL"""
    # Use a sample MoEngage URL for testing
    test_url = "https://www.scientificamerican.com/article/surprising-ways-that-sunlight-might-heal-autoimmune-diseases"
    
    print(f"Testing analyzer with URL: {test_url}")
    
    try:
        result = subprocess.run([
    sys.executable, 'analyzer.py', test_url
], capture_output=True, text=True, timeout=600)  # Increased timeout
        
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Analyzer ran successfully!")
            try:
                data = json.loads(result.stdout)
                print(f"✅ JSON output is valid")
                print(f"Title: {data.get('title', 'N/A')}")
                print(f"Readability Score: {data.get('readability', {}).get('readability_score', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Raw output: {result.stdout}")
        else:
            print(f"❌ Analyzer failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("❌ Analyzer timed out")
    except FileNotFoundError:
        print("❌ analyzer.py not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_analyzer()