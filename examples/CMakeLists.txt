add_executable(run run/main.cc)

target_include_directories(run
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(run 
                      tokenizers)

set_property(TARGET run PROPERTY CXX_STANDARD 17)