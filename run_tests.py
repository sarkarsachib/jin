#!/usr/bin/env python3
"""
Jin Core - Test Runner
Runs the test suite and reports results.
"""

import sys
import subprocess
from pathlib import Path

TEST_EXECUTABLE = Path('build/bin/test_jin_core')

def run_tests(filter_pattern=''):
    """Run the test executable."""
    if not TEST_EXECUTABLE.exists():
        print(f"Error: Test executable not found at {TEST_EXECUTABLE}")
        print("Please run 'python build.py' first.")
        return 1
    
    cmd = [str(TEST_EXECUTABLE)]
    if filter_pattern:
        cmd.append(filter_pattern)
    
    result = subprocess.run(cmd)
    return result.returncode

def main():
    """Main test runner entry point."""
    filter_pattern = ''
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--filter':
            if len(sys.argv) > 2:
                filter_pattern = sys.argv[2]
            else:
                print("Error: --filter requires a pattern argument")
                return 1
        else:
            filter_pattern = sys.argv[1]
    
    if filter_pattern:
        print(f"Running tests matching: {filter_pattern}\n")
    
    return run_tests(filter_pattern)

if __name__ == '__main__':
    sys.exit(main())
