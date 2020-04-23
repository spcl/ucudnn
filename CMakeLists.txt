cmake_minimum_required(VERSION 3.9)

# vcudnn
project("vcudnn")
set(version "0.0.1")
set(soversion "1")
set(CMAKE_CXX_STANDARD 11)

add_definitions(-DVCUDNN_VERSION=\"${version}\")

option(VCUDNN_DEBUG_OUTPUT "print combination of optimized kernels to stderr" ON)
option(VCUDNN_DEBUG_OUTPUT_ENV "print used environment variables to stderr" OFF)
option(VCUDNN_TEST "build tests" ON)

if(VCUDNN_DEBUG_OUTPUT)
  add_definitions(-DVCUDNN_DEBUG_OUTPUT)
endif()
if(VCUDNN_DEBUG_OUTPUT_ENV)
  add_definitions(-DVCUDNN_DEBUG_OUTPUT_ENV)
endif()

exec_program("git" ARGS "rev-parse HEAD"
  OUTPUT_VARIABLE gitHead
  RETURN_VALUE ret)
if(${ret} EQUAL 0)
  add_definitions(-DVCUDNN_COMMIT_ID=\"${gitHead}\")
endif()

# libraries
find_package(CUDA 9.0 REQUIRED)
find_cuda_helper_libs(cublas)
find_path   (CUDNN_INCLUDE NAMES cudnn.h)
find_library(CUDNN_LIBRARY NAMES cudnn)
find_library(CUDART_LIBRARY NAMES cudart)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE CUDNN_LIBRARY)

file(GLOB cppFiles src/*.cpp)
file(GLOB headerFiles src/*.h)
file(GLOB testCppFiles test/main.cpp)

include_directories(src)

add_library(vcudnn SHARED ${cppFiles})
set_target_properties(vcudnn PROPERTIES VERSION ${version} SOVERSION ${soversion})
target_link_libraries(vcudnn cublas cudnn cudart)

if(VCUDNN_TEST)
  add_subdirectory(test)
endif()

install(TARGETS vcudnn LIBRARY DESTINATION lib)
install(FILES ${headerFiles} DESTINATION include/vcudnn)

enable_testing()
add_test(maintest test/main)


