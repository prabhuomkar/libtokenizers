cmake_minimum_required(VERSION 3.22.1)

project(tokenizers VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

set(ICU_ROOT /opt/homebrew/opt/icu4c)
set(ICU_INCLUDE_DIRS ${ICU_ROOT}/include)
set(ICU_LIBRARIES ${ICU_ROOT}/lib)
find_package(ICU REQUIRED COMPONENTS uc i18n data)

add_subdirectory(third_party/simdjson)

set(SOURCES
  src/common.cc
  src/decoder.cc
  src/model.cc
  src/normalizer.cc
  src/post_processor.cc
  src/pre_tokenizer.cc
  src/tokenizer.cc
  src/utils.cc
)

add_library(tokenizers SHARED ${SOURCES})

set_target_properties(tokenizers PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(tokenizers
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ICU_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson/include
)

target_link_libraries(tokenizers
  PRIVATE
    ICU::uc
    ICU::i18n
    ICU::data
    simdjson
)

install(TARGETS tokenizers
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)

if(BUILD_TESTS STREQUAL "ON")
  message(STATUS "Building Tests")
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(GTEST_CREATE_TESTS OFF)
  set(INSTALL_GTEST OFF)
  add_subdirectory(third_party/googletest)
  add_subdirectory(third_party/benchmark)
  enable_testing()
  add_subdirectory(tests)
endif()