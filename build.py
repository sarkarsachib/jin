#!/usr/bin/env python3
"""
Jin Core - Custom Build Script
Compiles the tensor engine and autodiff system from scratch.
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

# Configuration
CXX = os.environ.get('CXX', 'g++')
CXXFLAGS = [
    '-std=c++17',
    '-Wall',
    '-Wextra',
    '-O2',
    '-g',
    '-fPIC',
]

DEBUG_FLAGS = ['-O0', '-g', '-DDEBUG']
RELEASE_FLAGS = ['-O3', '-DNDEBUG']

SRC_DIR = Path('src')
TEST_DIR = Path('test')
BUILD_DIR = Path('build')
OBJ_DIR = BUILD_DIR / 'obj'
LIB_DIR = BUILD_DIR / 'lib'
BIN_DIR = BUILD_DIR / 'bin'

def ensure_dirs():
    """Create build directories if they don't exist."""
    for d in [BUILD_DIR, OBJ_DIR, LIB_DIR, BIN_DIR]:
        d.mkdir(exist_ok=True)

def find_sources(directory, pattern='**/*.cpp'):
    """Find all C++ source files in directory."""
    return list(Path(directory).glob(pattern))

def compile_file(src_file, obj_file, flags):
    """Compile a single C++ file to object file."""
    cmd = [CXX] + flags + ['-c', str(src_file), '-o', str(obj_file)]
    print(f"Compiling {src_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error compiling {src_file}:")
        print(result.stderr)
        return False
    
    return True

def link_library(obj_files, lib_file):
    """Link object files into a static library."""
    print(f"Creating library {lib_file}...")
    cmd = ['ar', 'rcs', str(lib_file)] + [str(f) for f in obj_files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating library:")
        print(result.stderr)
        return False
    
    return True

def link_executable(obj_files, exe_file, lib_files, flags):
    """Link object files and libraries into an executable."""
    print(f"Linking executable {exe_file}...")
    cmd = [CXX] + flags + [str(f) for f in obj_files] + ['-o', str(exe_file)] + lib_files + ['-lm', '-lpthread']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error linking executable:")
        print(result.stderr)
        return False
    
    return True

def build_library(mode='release'):
    """Build the Jin core library."""
    print(f"\n=== Building Jin Core Library ({mode}) ===\n")
    
    flags = CXXFLAGS + (DEBUG_FLAGS if mode == 'debug' else RELEASE_FLAGS)
    
    # Find all source files
    src_files = []
    for subdir in ['core', 'autodiff', 'utils', 'nn', 'optim']:
        src_files.extend(find_sources(SRC_DIR / subdir, '*.cpp'))
    
    if not src_files:
        print("No source files found!")
        return False
    
    # Compile all source files
    obj_files = []
    for src_file in src_files:
        rel_path = src_file.relative_to(SRC_DIR)
        obj_file = OBJ_DIR / rel_path.with_suffix('.o')
        obj_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not compile_file(src_file, obj_file, flags):
            return False
        
        obj_files.append(obj_file)
    
    # Create static library
    lib_file = LIB_DIR / 'libjin-core.a'
    if not link_library(obj_files, lib_file):
        return False
    
    print(f"\n✓ Library built successfully: {lib_file}\n")
    return True

def build_tests(mode='release'):
    """Build the test executable."""
    print(f"\n=== Building Tests ({mode}) ===\n")
    
    flags = CXXFLAGS + (DEBUG_FLAGS if mode == 'debug' else RELEASE_FLAGS)
    
    # Compile test main
    test_main = TEST_DIR / 'main.cpp'
    test_obj = OBJ_DIR / 'test_main.o'
    
    if not compile_file(test_main, test_obj, flags):
        return False
    
    # Link with library
    lib_file = LIB_DIR / 'libjin-core.a'
    exe_file = BIN_DIR / 'test_jin_core'
    
    if not link_executable([test_obj], exe_file, [str(lib_file)], flags):
        return False
    
    print(f"\n✓ Tests built successfully: {exe_file}\n")
    return True

def clean():
    """Clean build artifacts."""
    print("Cleaning build directory...")
    import shutil
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    print("✓ Clean complete\n")

def main():
    """Main build entry point."""
    mode = 'release'
    target = 'all'
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg == 'debug':
            mode = 'debug'
        elif arg == 'release':
            mode = 'release'
        elif arg == 'clean':
            clean()
            return 0
        elif arg in ['lib', 'tests', 'all']:
            target = arg
    
    ensure_dirs()
    
    success = True
    
    if target in ['lib', 'all']:
        success = build_library(mode) and success
    
    if target in ['tests', 'all']:
        success = build_tests(mode) and success
    
    if success:
        print("\n" + "="*50)
        print("✓ Build completed successfully!")
        print("="*50)
        print(f"\nLibrary: build/lib/libjin-core.a")
        print(f"Tests:   build/bin/test_jin_core")
        print(f"\nRun tests with: ./build/bin/test_jin_core")
        print(f"Or use:         python run_tests.py")
        print()
        return 0
    else:
        print("\n" + "="*50)
        print("✗ Build failed!")
        print("="*50 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
