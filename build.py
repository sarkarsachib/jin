#!/usr/bin/env python3
"""
Jin Core - Custom Build Script
Compiles the tensor engine, autodiff system, and GPU acceleration layer.
"""

import os
import sys
import subprocess
import glob
import platform
from pathlib import Path
import argparse

# Configuration
CXX = os.environ.get('CXX', 'g++')
CUDA_BIN_PATH = os.environ.get('CUDA_BIN_PATH', '/usr/local/cuda/bin')
VULKAN_SDK_PATH = os.environ.get('VULKAN_SDK_PATH', '')

# Base C++ flags
CXXFLAGS = [
    '-std=c++17',
    '-Wall',
    '-Wextra',
    '-O2',
    '-g',
    '-fPIC',
]

# Backend-specific flags
CUDA_FLAGS = ['-arch=sm_35', f'-I{CUDA_BIN_PATH}/../include']
VULKAN_FLAGS = []
if VULKAN_SDK_PATH:
    VULKAN_FLAGS = [f'-I{VULKAN_SDK_PATH}/include']

# Metal flags (macOS only)
METAL_FLAGS = []
if platform.system() == 'Darwin':
    METAL_FLAGS = ['-framework', 'Metal', '-framework', 'MetalKit', '-framework', 'Cocoa']

DEBUG_FLAGS = ['-O0', '-g', '-DDEBUG']
RELEASE_FLAGS = ['-O3', '-DNDEBUG']

SRC_DIR = Path('src')
TEST_DIR = Path('test')
BUILD_DIR = Path('build')
OBJ_DIR = BUILD_DIR / 'obj'
LIB_DIR = BUILD_DIR / 'lib'
BIN_DIR = BUILD_DIR / 'bin'

# Source directories by backend
BASE_SOURCES = {
    'core': list(SRC_DIR.glob('core/*.cpp')),
    'autodiff': list(SRC_DIR.glob('autodiff/*.cpp')),
    'utils': list(SRC_DIR.glob('utils/*.cpp'))
}

GPU_SOURCES = {
    'cuda': list(SRC_DIR.glob('gpu/cuda/*.cpp')) + list(SRC_DIR.glob('gpu/*.cpp')),
    'vulkan': list(SRC_DIR.glob('gpu/vulkan/*.cpp')) + list(SRC_DIR.glob('gpu/*.cpp')),
    'metal': list(SRC_DIR.glob('gpu/metal/*.mm')) + list(SRC_DIR.glob('gpu/*.cpp')),
    'kernel_fusion': list(SRC_DIR.glob('gpu/kernel_fusion.cpp')) + list(SRC_DIR.glob('gpu/*.cpp'))
}

def ensure_dirs():
    """Create build directories if they don't exist."""
    for d in [BUILD_DIR, OBJ_DIR, LIB_DIR, BIN_DIR]:
        d.mkdir(exist_ok=True)

def find_sources(directory, pattern='**/*.cpp'):
    """Find all C++ source files in directory."""
    return list(Path(directory).glob(pattern))

def find_cuda_sources():
    """Find CUDA source files."""
    cuda_sources = []
    for root, dirs, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith('.cu'):
                cuda_sources.append(Path(root) / file)
    return cuda_sources

def compile_cuda_file(src_file, obj_file, flags):
    """Compile a CUDA file using nvcc."""
    nvcc = os.path.join(CUDA_BIN_PATH, 'nvcc')
    if not os.path.exists(nvcc):
        print(f"nvcc not found at {nvcc}. CUDA compilation skipped.")
        return True  # Skip rather than fail
    
    cmd = [nvcc] + flags + ['-c', str(src_file), '-o', str(obj_file)]
    print(f"Compiling CUDA {src_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error compiling CUDA {src_file}:")
        print(result.stderr)
        return False
    
    return True

def compile_file(src_file, obj_file, flags, backend=None):
    """Compile a single C++/CUDA file to object file."""
    if src_file.suffix == '.cu':
        return compile_cuda_file(src_file, obj_file, flags)
    elif src_file.suffix == '.mm':  # Metal Objective-C++
        # Treat .mm files as C++ files since we don't have Metal framework
        cmd = [CXX] + CXXFLAGS + flags + ['-c', str(src_file), '-o', str(obj_file)]
    else:
        cmd = [CXX] + CXXFLAGS + flags + ['-c', str(src_file), '-o', str(obj_file)]
    
    print(f"Compiling {src_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error compiling {src_file}:")
        print(result.stderr)
        return False
    
    return True

def link_library(obj_files, lib_file, backend=None):
    """Link object files into a static library."""
    print(f"Creating library {lib_file}...")
    cmd = ['ar', 'rcs', str(lib_file)] + [str(f) for f in obj_files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating library:")
        print(result.stderr)
        return False
    
    return True

def link_executable(obj_files, exe_file, lib_files, flags, backend=None):
    """Link object files and libraries into an executable."""
    print(f"Linking executable {exe_file}...")
    
    # Base linking command
    link_cmd = [CXX] + flags + [str(f) for f in obj_files] + ['-o', str(exe_file)]
    
    # Add library files
    lib_args = [str(lib) for lib in lib_files] + ['-lm', '-lpthread']
    link_cmd.extend(lib_args)
    
    # Add backend-specific libraries
    if backend == 'cuda':
        cuda_lib_path = os.path.join(CUDA_BIN_PATH, '..', 'lib64')
        link_cmd.extend(['-L' + cuda_lib_path, '-lcudart'])
    elif backend == 'vulkan':
        if VULKAN_SDK_PATH:
            vulkan_lib_path = os.path.join(VULKAN_SDK_PATH, 'lib')
            link_cmd.extend(['-L' + vulkan_lib_path, '-lvulkan'])
    elif backend == 'metal':
        link_cmd.extend(METAL_FLAGS)
    
    result = subprocess.run(link_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error linking executable:")
        print(result.stderr)
        return False
    
    return True

def build_library(mode='release', backend='cpu'):
    """Build the Jin core library with specified backend."""
    print(f"\n=== Building Jin Core Library ({mode}) - Backend: {backend} ===\n")
    
    flags = DEBUG_FLAGS if mode == 'debug' else RELEASE_FLAGS
    
    # Add backend-specific flags
    if backend == 'cuda':
        flags.extend(CUDA_FLAGS)
    elif backend == 'vulkan':
        flags.extend(VULKAN_FLAGS)
    elif backend == 'metal':
        flags.extend(METAL_FLAGS)
    
    # Find source files
    src_files = []
    
    # Always include base sources
    for subdir in BASE_SOURCES:
        src_files.extend(BASE_SOURCES[subdir])
    
    # Add GPU sources if backend is GPU
    if backend != 'cpu':
        if backend in GPU_SOURCES:
            src_files.extend(GPU_SOURCES[backend])
    
    if not src_files:
        print("No source files found!")
        return False
    
    # Compile all source files
    obj_files = []
    for src_file in src_files:
        rel_path = src_file.relative_to(SRC_DIR)
        obj_file = OBJ_DIR / rel_path.with_suffix('.o')
        obj_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not compile_file(src_file, obj_file, flags, backend):
            return False
        
        obj_files.append(obj_file)
    
    # Create static library
    lib_file = LIB_DIR / f'libjin-core-{backend}.a'
    if not link_library(obj_files, lib_file, backend):
        return False
    
    print(f"\n✓ Library built successfully: {lib_file}\n")
    return True

def build_tests(mode='release', backend='cpu'):
    """Build the test executable."""
    print(f"\n=== Building Tests ({mode}) - Backend: {backend} ===\n")
    
    flags = DEBUG_FLAGS if mode == 'debug' else RELEASE_FLAGS
    
    # Add backend-specific flags
    if backend == 'cuda':
        flags.extend(CUDA_FLAGS)
    elif backend == 'vulkan':
        flags.extend(VULKAN_FLAGS)
    elif backend == 'metal':
        flags.extend(METAL_FLAGS)
    
    # Compile test main
    test_main = TEST_DIR / 'main.cpp'
    test_obj = OBJ_DIR / 'test_main.o'
    
    if not compile_file(test_main, test_obj, flags, backend):
        return False
    
    # Link with library
    lib_file = LIB_DIR / f'libjin-core-{backend}.a'
    exe_file = BIN_DIR / f'test_jin_core_{backend}'
    
    if not link_executable([test_obj], exe_file, [str(lib_file)], flags, backend):
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

def check_dependencies(backend):
    """Check if backend dependencies are available."""
    if backend == 'cuda':
        nvcc = os.path.join(CUDA_BIN_PATH, 'nvcc')
        if not os.path.exists(nvcc):
            print(f"⚠️  CUDA backend requested but nvcc not found at {nvcc}")
            print("   Set CUDA_BIN_PATH environment variable or install CUDA Toolkit")
            return False
    elif backend == 'vulkan':
        if VULKAN_SDK_PATH and not os.path.exists(os.path.join(VULKAN_SDK_PATH, 'include', 'vulkan', 'vulkan.h')):
            print(f"⚠️  Vulkan backend requested but Vulkan SDK not found at {VULKAN_SDK_PATH}")
            print("   Set VULKAN_SDK_PATH environment variable or install Vulkan SDK")
            return False
    elif backend == 'metal':
        if platform.system() != 'Darwin':
            print("⚠️  Metal backend requested but not on macOS")
            return False
    
    return True

def list_available_backends():
    """List available GPU backends based on system configuration."""
    backends = ['cpu']  # CPU is always available
    
    # Check CUDA
    nvcc = os.path.join(CUDA_BIN_PATH, 'nvcc')
    if os.path.exists(nvcc):
        backends.append('cuda')
    
    # Check Vulkan
    if VULKAN_SDK_PATH and os.path.exists(os.path.join(VULKAN_SDK_PATH, 'include', 'vulkan', 'vulkan.h')):
        backends.append('vulkan')
    else:
        # Try system vulkan
        vulkan_headers = ['/usr/include/vulkan/vulkan.h', '/usr/local/include/vulkan/vulkan.h']
        if any(os.path.exists(header) for header in vulkan_headers):
            backends.append('vulkan')
    
    # Check Metal (macOS only)
    if platform.system() == 'Darwin':
        backends.append('metal')
    
    return backends

def main():
    """Main build entry point."""
    parser = argparse.ArgumentParser(description='Jin Core Build System')
    parser.add_argument('mode', nargs='?', choices=['debug', 'release'], default='release',
                       help='Build mode (default: release)')
    parser.add_argument('target', nargs='?', choices=['lib', 'tests', 'all'], default='all',
                       help='Build target (default: all)')
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'vulkan', 'metal', 'all'], 
                       default='cpu', help='GPU backend to build (default: cpu)')
    parser.add_argument('--clean', action='store_true', help='Clean build directory')
    parser.add_argument('--list-backends', action='store_true', 
                       help='List available GPU backends')
    parser.add_argument('--deps-check', action='store_true',
                       help='Check GPU backend dependencies')
    
    args = parser.parse_args()
    
    if args.list_backends:
        backends = list_available_backends()
        print("Available GPU backends:")
        for backend in backends:
            print(f"  - {backend}")
        return 0
    
    if args.deps_check:
        backends = list_available_backends()
        print("GPU Backend Dependencies:")
        for backend in ['cuda', 'vulkan', 'metal']:
            available = backend in backends
            status = "✓ Available" if available else "✗ Not available"
            print(f"  {backend.upper()}: {status}")
        return 0
    
    if args.clean:
        clean()
        return 0
    
    mode = args.mode
    target = args.target
    backend = args.backend
    
    # Handle 'all' backend option
    if backend == 'all':
        backends = list_available_backends()
        if 'cpu' not in backends:
            backends.insert(0, 'cpu')
    else:
        backends = [backend]
    
    # Check dependencies for each backend
    for b in backends:
        if not check_dependencies(b):
            print(f"Skipping backend {b} due to missing dependencies")
            backends.remove(b)
    
    if not backends:
        print("No available backends to build!")
        return 1
    
    ensure_dirs()
    
    success = True
    
    # Build for each backend
    for b in backends:
        print(f"\n{'='*60}")
        print(f"Building backend: {b.upper()}")
        print(f"{'='*60}")
        
        if target in ['lib', 'all']:
            success = build_library(mode, b) and success
        
        if target in ['tests', 'all']:
            success = build_tests(mode, b) and success
    
    if success:
        print("\n" + "="*50)
        print("✓ Build completed successfully!")
        print("="*50)
        print(f"\nLibraries:")
        for b in backends:
            lib_file = LIB_DIR / f'libjin-core-{b}.a'
            if lib_file.exists():
                print(f"  {b.upper()}: {lib_file}")
        print(f"\nTests:")
        for b in backends:
            exe_file = BIN_DIR / f'test_jin_core_{b}'
            if exe_file.exists():
                print(f"  {b.upper()}: {exe_file}")
        print(f"\nRun tests with: python run_tests.py")
        print()
        return 0
    else:
        print("\n" + "="*50)
        print("✗ Build failed!")
        print("="*50 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
