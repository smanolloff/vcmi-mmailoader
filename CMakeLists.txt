cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

option(ENABLE_LIBTORCH "link against libtorch to enable loading of MMAI_MODEL TorchJIT files" ON)

add_definitions(-DVCMI_BIN_DIR="${CMAKE_BINARY_DIR}/bin")
add_definitions(-DVCMI_ROOT_DIR="${CMAKE_SOURCE_DIR}")

add_library(mmailoader SHARED
    Loader.cpp
    Loader.h
    TorchModel.cpp
    TorchModel.h
)

if(ENABLE_LIBTORCH)
    add_definitions(-DENABLE_LIBTORCH=1)

    # Ensure client/ML/libtorch is a symlink to the
    # same libtorch downloaded by pip in the vcmi-gym project
    # (e.g. "VCMI_GYM_DIR/.venv/lib/python3.10/site-packages/torch")

    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
    find_package(Boost REQUIRED)
    find_package(Torch REQUIRED)
    find_package(OpenMP REQUIRED)

    target_link_libraries(mmailoader PRIVATE
        "${Boost_LIBRARIES}"
        "${TORCH_LIBRARIES}"
        OpenMP::OpenMP_CXX
    )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

vcmi_set_output_dir(mmailoader "")
# enable_pch(mmailoader)

install(TARGETS mmailoader DESTINATION ${BIN_DIR})
