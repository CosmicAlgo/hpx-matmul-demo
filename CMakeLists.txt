cmake_minimum_required(VERSION 3.18)
project(hpx_matmul_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(HPX REQUIRED)

add_executable(matmul_hpx matmul_hpx.cpp)
hpx_setup_target(matmul_hpx)

install(TARGETS matmul_hpx DESTINATION bin)
