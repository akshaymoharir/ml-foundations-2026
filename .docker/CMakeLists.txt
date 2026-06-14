cmake_minimum_required(VERSION 3.22)
project(minidetect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # feeds clangd for IDE support

# ── Compiler flags ────────────────────────────────────────
# RTX 3090 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter       # relax during early learning phase
)

# ── GoogleTest ────────────────────────────────────────────
find_package(GTest REQUIRED)
enable_testing()

# ── DSA problems (Track A — C++ practice) ────────────────
# Add each problem file here as you write them.
# Pattern: one test binary per topic group.

file(GLOB_RECURSE DSA_SOURCES "dsa/cpp/**/*.cpp")
if(DSA_SOURCES)
    add_executable(dsa_tests ${DSA_SOURCES})
    target_link_libraries(dsa_tests GTest::gtest_main)
    add_test(NAME dsa COMMAND dsa_tests)
endif()

# ── MiniDetect C++ components (future — Phase 1 C++ plan) ─
# Uncomment when you start the Driveway Detector track:
# add_subdirectory(minidetect)
