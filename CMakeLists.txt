cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

option(ENABLE_LIBTORCH "link against libtorch to enable loading of MMAI_MODEL TorchJIT files" ON)

add_definitions(-DVCMI_BIN_DIR="${CMAKE_BINARY_DIR}/bin")
add_definitions(-DVCMI_ROOT_DIR="${CMAKE_SOURCE_DIR}")

find_package(Boost REQUIRED)

set(mmailoader_FILES
    Loader.cpp
    Loader.h
    TorchModel.h
)

set(mmailoader_LIBS "${Boost_LIBRARIES}")

#[[
About the ENABLE_LIBTORCH flag:

On Mac: use ENABLE_LIBTORCH=1.
On Windows: no idea (ENABLE_LIBTORCH=1 may just work; have not tested it)
On Linux: read below...

VCMI uses a lot of 3rd party libs (like sdl, boost, zlib, etc.) which are
installed by Linux package managers (e.g. `apt install libsdl2-dev`).
Those libs are all compiled with CXX11 ABI. VCMI itself is also compiled
with CXX11 ABI, so it can link to those libs.

For ML, there is an additional 3rd party lib needed: libtorch. However, it is
not installed by a Linux package manager, but by Python's `pip` tool instead.
Unfortunately, it is compiled *without* CXX11 ABI, so VCMI cannot link to it.

There are two solutions to this issue:

1. Compile `libtorch` from source with CXX11 ABI.
    This is not a straight-forward operation.
    If CUDA support is required, it becomes tricky as graphics card drivers
    may also need to be compiled from source.
    There is a pre-compiled `libtorch` CXX11 ABI binary, but it has no CUDA support.
    (see https://github.com/pytorch/pytorch/issues/51039#issuecomment-2111849591)
2. Compile `VCMI` from source without CXX11 ABI.
    This is also not a straight-forward operation.
    It requires compiling all dependencies from scratch (sdl, boost, etc.).
    If using a C++ package manager (like "conan"), it should be possible.
    I was unable to make VCMI deps compile with conan on Linux (Ubuntu 22.04).

Alternative to the above *solutions*, there are two *workarounds* instead:

1. Compile VCMI with `-D ENABLE_LIBTORCH=0`.
* Pros: allows training new models on CPU and GPU
* Cons: prevents loading pre-trained models for gameplay
* Requirements: you have installed the "default" Python torch package with `pip`
  (see `requirements.txt` in vcmi-gym).

2. Compile VCMI with `-D ENABLE_LIBTORCH=1`.
* Pros: allows loading pre-trained models for gameplay
* Cons: prevents training of new models on GPU (can be trained on CPU only).
  Requirements: you have installed the CPU-only cxx11 ABI torch package with `pip`
  (see `requirements.txt` in vcmi-gym).
]]

if(ENABLE_LIBTORCH)
    add_definitions(-DENABLE_LIBTORCH=1)

    # Ensure client/ML/libtorch is a symlink to the
    # same libtorch downloaded by pip in the vcmi-gym project
    # (e.g. "VCMI_GYM_DIR/.venv/lib/python3.10/site-packages/torch")

    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
    find_package(Torch REQUIRED)
    find_package(OpenMP REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    list(APPEND mmailoader_LIBS OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}")
    list(APPEND mmailoader_FILES TorchModel.cpp)
else()
    list(APPEND mmailoader_FILES TorchModelDummy.cpp)
endif()

add_library(mmailoader SHARED ${mmailoader_FILES})
target_link_libraries(mmailoader PRIVATE ${mmailoader_LIBS})

vcmi_set_output_dir(mmailoader "")
# enable_pch(mmailoader)

install(TARGETS mmailoader DESTINATION ${BIN_DIR})
