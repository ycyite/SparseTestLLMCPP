cmake_minimum_required(VERSION 3.27)
set(TARGET llama-visual)
add_executable(${TARGET} visualPrameters.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE (your location)/llama.cpp/common/build/libcommon.a
(your location)/llama.cpp/src/build/libllama.a
(your location)/llama.cpp/ggml/build/src/libggml.so
(your location)/llama.cpp/ggml/build/src/libggml-base.so
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE 
(your location)/llama.cpp/common
(your location)/llama.cpp/include
(your location)/llama.cpp/ggml/include
(your location)/llama.cpp/src
)
