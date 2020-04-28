file(GLOB cppFiles *.cpp)
set(test_filename vcudnn_test)
add_executable(${test_filename} ${cppFiles})
target_link_libraries(${test_filename} cudart cublas curand cudnn vcudnn)
