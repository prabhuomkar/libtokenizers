file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cc)

add_executable(tokenizers_tests ${TEST_SOURCES})

target_include_directories(tokenizers_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${ICU_INCLUDE_DIRS}
)

target_link_libraries(tokenizers_tests 
                      gtest_main
                      tokenizers
                      ICU::uc
                      ICU::i18n
                      ICU::data)

include(GoogleTest)

gtest_discover_tests(tokenizers_tests)

file(GLOB BENCHMARK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*_benchmark.cc)

add_executable(tokenizers_benchmarks ${BENCHMARK_SOURCES})

target_include_directories(tokenizers_benchmarks
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${ICU_INCLUDE_DIRS}
)

target_link_libraries(tokenizers_benchmarks 
                      benchmark_main
                      tokenizers
                      ICU::uc
                      ICU::i18n
                      ICU::data)
