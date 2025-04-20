set(PYTORCH_VERSION "2.6.0")

include(FetchContent)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${PYTORCH_VERSION}.zip")
endif()

FetchContent_Declare(
    libtorch
    PREFIX libtorch
    DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/libtorch
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/libtorch
    URL ${LIBTORCH_URL}
)

FetchContent_MakeAvailable(libtorch)

find_package(Torch REQUIRED PATHS "${CMAKE_SOURCE_DIR}/libtorch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mobilebert-fill-mask mobilebert-fill-mask/main.cc)

target_include_directories(mobilebert-fill-mask
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${ICU_INCLUDE_DIRS}
)

target_link_libraries(mobilebert-fill-mask 
                      tokenizers
                      "${TORCH_LIBRARIES}")

set_property(TARGET mobilebert-fill-mask PROPERTY CXX_STANDARD 17)