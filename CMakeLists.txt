# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build for libmorie -- the morie C++ numeric core (v0.9.1 backend
# port). Driven by scikit-build-core; produces the nanobind extension
# module installed as ``morie._core``.

cmake_minimum_required(VERSION 3.15...3.31)
project(morie_core LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Emit compile_commands.json so the clangd language server (see the
# repo-root .clangd) can resolve the nanobind / Python build headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# scikit-build-core points CMake at the correct Python; nanobind is
# installed into the build-isolation environment via build-system
# requires, so its CMake package config is discoverable.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NB_STATIC links nanobind's runtime statically -- one self-contained
# extension, no separate shared nanobind dependency in the wheel.
nanobind_add_module(_core NB_STATIC
  libmorie/_core.cpp
  libmorie/kernels.cpp
  libmorie/hawkes.cpp)

# Land the compiled module inside the morie package: morie/_core*.so,
# importable as ``morie._core``.
install(TARGETS _core LIBRARY DESTINATION morie)
