cmake_minimum_required(VERSION 3.9)

# ucudnn
project("ucudnn")
set(version "1.1.0")
set(soversion "1")
set(CMAKE_CXX_STANDARD 11)

add_definitions(-DUCUDNN_VERSION=\"${version}\")

option(UCUDNN_USE_GLPK "use GNU Linear Programming Kit (GLPK) to run ILP-based optimization" OFF)
option(UCUDNN_USE_SQLITE "use SQLite to cache benchmarking results externally" OFF)
option(UCUDNN_DEBUG_GLPK_PRINT "print ILP information via GLPK function" OFF)
option(UCUDNN_DEBUG_OUTPUT "print combination of optimized kernels to stderr" ON)
option(UCUDNN_DEBUG_OUTPUT_ENV "print used environment variables to stderr" OFF)
option(UCUDNN_DEBUG_EQUIVALENCE_TEST "output distance between cuDNN output and u-cuDNN output (use for evaluation only because this is extremely slow)" OFF)
option(UCUDNN_TEST "build tests" ON)

if(UCUDNN_USE_GLPK)
  add_definitions(-DUCUDNN_USE_GLPK)
endif()
if(UCUDNN_USE_SQLITE)
  add_definitions(-DUCUDNN_USE_SQLITE)
endif()
if(UCUDNN_DEBUG_GLPK_PRINT)
  add_definitions(-DUCUDNN_DEBUG_GLPK_PRINT)
endif()
if(UCUDNN_DEBUG_OUTPUT)
  add_definitions(-DUCUDNN_DEBUG_OUTPUT)
endif()
if(UCUDNN_DEBUG_OUTPUT_ENV)
  add_definitions(-DUCUDNN_DEBUG_OUTPUT_ENV)
endif()
if(UCUDNN_DEBUG_EQUIVALENCE_TEST)
  add_definitions(-DUCUDNN_DEBUG_EQUIVALENCE_TEST)
endif()

exec_program("git" ARGS "rev-parse HEAD"
  OUTPUT_VARIABLE gitHead
  RETURN_VALUE ret)
if(${ret} EQUAL 0)
  add_definitions(-DUCUDNN_COMMIT_ID=\"${gitHead}\")
endif()

# libraries
find_package(CUDA 8.0 REQUIRED)
find_cuda_helper_libs(cublas)
find_path   (CUDNN_INCLUDE NAMES cudnn.h)
find_library(CUDNN_LIBRARY NAMES cudnn)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE CUDNN_LIBRARY)

set(glpkLibrary "")
set(sqliteLibrary "")

if(UCUDNN_USE_GLPK)
  find_path   (GLPK_INCLUDE NAMES glpk.h)
  find_library(GLPK_LIBRARY NAMES glpk)
  find_package_handle_standard_args(GLPK DEFAULT_MSG GLPK_INCLUDE GLPK_LIBRARY)
  set(glpkLibrary "glpk")
endif()

if(UCUDNN_USE_SQLITE)
  find_path   (SQLITE_INCLUDE NAMES sqlite3.h)
  find_library(SQLITE_LIBRARY NAMES sqlite3)
  find_package_handle_standard_args(SQLITE DEFAULT_MSG SQLITE_INCLUDE SQLITE_LIBRARY)
  set(sqliteLibrary "sqlite3")
endif()

file(GLOB cppFiles src/*.cpp)
file(GLOB headerFiles src/*.h)
file(GLOB testCppFiles test/main.cpp)

include_directories(src)

add_library(ucudnn SHARED ${cppFiles})
set_target_properties(ucudnn PROPERTIES VERSION ${version} SOVERSION ${soversion})
target_link_libraries(ucudnn cublas cudnn ${glpkLibrary} ${sqliteLibrary})

if(UCUDNN_TEST)
  add_subdirectory(test)
endif()

install(TARGETS ucudnn LIBRARY DESTINATION lib)
install(FILES ${headerFiles} DESTINATION include/ucudnn)

enable_testing()
add_test(AlexNet test/main)
add_test(DP test/dp)
add_test(desirableSet test/desirableSet)

set_property(TEST AlexNet PROPERTY SKIP_RETURN_CODE 200) # GPU not found
set_property(TEST DP      PROPERTY SKIP_RETURN_CODE 200) # SQLite not found
