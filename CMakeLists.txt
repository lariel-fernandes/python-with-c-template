cmake_minimum_required(VERSION 3.18)

project(python_with_c_test C CXX CUDA)

set(PYTHON_VENV_BIN "${CMAKE_SOURCE_DIR}/.venv/bin/python" )

execute_process(
    COMMAND readlink -f ${PYTHON_VENV_BIN}
    OUTPUT_VARIABLE PYTHON_REAL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

get_filename_component(PYTHON_BIN_DIR "${PYTHON_REAL_PATH}" DIRECTORY)
get_filename_component(Python3_ROOT_DIR "${PYTHON_BIN_DIR}" DIRECTORY)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(PkgConfig REQUIRED)

# ============================================================
# MATH (Pure C Extension)
# ============================================================
message(STATUS "Configuring MATH extension")

file(GLOB_RECURSE MATH_EXT_SOURCES
    src/my_proj/ext/_math/*.c
)

add_library(math_ext SHARED ${MATH_EXT_SOURCES})

set_property(TARGET math_ext PROPERTY C_STANDARD 11)

target_compile_definitions(math_ext PRIVATE PY_SSIZE_T_CLEAN)

pkg_check_modules(CURL REQUIRED libcurl)

target_include_directories(math_ext PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    src/my_proj/ext/_math
)

target_link_libraries(math_ext PRIVATE
    ${Python3_LIBRARIES}
    ${CURL_LIBRARIES}
)

message(STATUS "Configuring MATH extension - done")

# ============================================================
# LINALG (Optional C/C++ Extension requiring CUDA)
# ============================================================
message(STATUS "Configuring LINALG extension")

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "Configuring LINALG extension - CUDA found: ${CUDAToolkit_INCLUDE_DIRS}")

    execute_process(
        COMMAND ${PYTHON_VENV_BIN} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Configuring LINALG extension - numpy found: ${NUMPY_INCLUDE_DIR}")

    file(GLOB_RECURSE LINALG_EXT_SOURCES
        src/my_proj/ext/_linalg/*.c
        src/my_proj/ext/_linalg/*.cu
    )

    add_library(linalg_ext SHARED ${LINALG_EXT_SOURCES})

    set_property(TARGET linalg_ext PROPERTY CUDA_STANDARD 11)

    set_property(TARGET linalg_ext PROPERTY C_STANDARD 11)

    target_compile_definitions(linalg_ext PRIVATE
        PY_SSIZE_T_CLEAN
        NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    )

    target_include_directories(linalg_ext PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        src/my_proj/ext/_linalg
    )

    target_link_libraries(linalg_ext PRIVATE
        ${Python3_LIBRARIES}
        CUDA::cudart
    )

    target_compile_options(linalg_ext PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
    )

    message(STATUS "Configuring LINALG extension - done")
else()
    message(WARNING "Configuring LINALG extension - skipped (CUDA not found)")
endif()

# ============================================================
# TORCH (Optional C++ Extension for PyTorch with CPU+CUDA support)
# ============================================================
message(STATUS "Configuring TORCH extension")

execute_process(
    COMMAND ${PYTHON_VENV_BIN} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Configuring TORCH extension - PyTorch found: ${TORCH_PATH}")

execute_process(
    COMMAND ${PYTHON_VENV_BIN} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${PYTHON_VENV_BIN} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.library_paths()))"
    OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(GLOB_RECURSE TORCH_EXT_SOURCES
    src/my_proj/ext/_torch_ext/*.cpp
)

if(CUDAToolkit_FOUND)
    message(STATUS "Configuring TORCH extension - building with CUDA support")
    file(GLOB_RECURSE TORCH_EXT_CU_SOURCES
        src/my_proj/ext/_torch_ext/*.cu
    )
    list(APPEND TORCH_EXT_SOURCES ${TORCH_EXT_CU_SOURCES})
    set(TORCH_WITH_CUDA TRUE)
else()
    message(WARNING "Configuring TORCH extension - building CPU-only")
    set(TORCH_WITH_CUDA FALSE)
endif()

add_library(torch_ext SHARED ${TORCH_EXT_SOURCES})

set_property(TARGET torch_ext PROPERTY CXX_STANDARD 17)
set_property(TARGET torch_ext PROPERTY CXX_STANDARD_REQUIRED ON)

if(TORCH_WITH_CUDA)
    set_property(TARGET torch_ext PROPERTY CUDA_STANDARD 17)
    target_compile_definitions(torch_ext PRIVATE WITH_CUDA)
endif()

target_include_directories(torch_ext PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    src/my_proj/ext/_torch_ext
)

if(TORCH_WITH_CUDA)
    target_include_directories(torch_ext PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_directories(torch_ext PRIVATE ${TORCH_LIBRARY_DIRS})

target_link_libraries(torch_ext PRIVATE
    ${Python3_LIBRARIES}
    torch
    torch_cpu
    c10
)

if(TORCH_WITH_CUDA)
    target_link_libraries(torch_ext PRIVATE
        torch_cuda
        CUDA::cudart
    )
    target_compile_options(torch_ext PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
    )
endif()

message(STATUS "Configuring TORCH extension - done")
