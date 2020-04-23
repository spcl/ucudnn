/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_UTIL_H_
#define UCUDNN_UTIL_H_

#include <iostream>
#include <vector>
#include <cudnn.h>

#ifdef UCUDNN_USE_SQLITE
#include <sqlite3.h>
#endif

// This constant is only used when frameworks try to get workspace size without providing workspace limit
#define UCUDNN_DEFAULT_WORKSPACE_LIMIT (64UL * 1024 * 1024)

namespace vcudnn {
  typedef long LayerId;
  static const LayerId LayerIdAny = -1L; // TODO: avoid hash collision
  LayerId stringToLayerId(const std::string s);

#define CUDNN_HAS_MATHTYPE (CUDNN_VERSION >= 7000)

  // Error handling functions
#define UCUDNN_CUDA_CHECK(error)					\
  if((error) != cudaSuccess) {						\
    std::cerr << "CUDA error(" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(error) << std::endl; \
  }

#define UCUDNN_CUBLAS_CHECK(status)					\
  if((status) != CUBLAS_STATUS_SUCCESS) {				\
    std::cerr << "cuBLAS error(" << __FILE__ << ":" << __LINE__ << "): " << status << std::endl; \
  }

#define UCUDNN_CUDNN_CHECK(status)					\
  if((status) != CUDNN_STATUS_SUCCESS) {				\
    std::cerr << "cuDNN error(" << __FILE__ << ":" << __LINE__ << "): " << cudnnGetErrorString(status) << std::endl; \
  }

#ifdef UCUDNN_USE_SQLITE
#define UCUDNN_SQLITE_CHECK(error)					\
  if((error) != SQLITE_OK) {						\
    std::cerr << "SQLite error(" << __FILE__ << ":" << __LINE__ << "): " << sqlite3_errstr(error) << std::endl; \
  }
#endif

#define UCUDNN_WARNING(str)						\
  std::cerr << "u-cuDNN warning(" << __FILE__ << ":" << __LINE__ << "): " << (str) << std::endl;

#define UCUDNN_ERROR_EXIT(str) {					\
    std::cerr << "u-cuDNN error(" << __FILE__ << ":" << __LINE__ << "): " << (str) << std::endl; \
    exit(1);								\
  }

  // type of convolution kernels
  enum ConvType {
    Forward,
    BackwardData,
    BackwardFilter,
    ConvTypeSize
  };

  enum BatchSizePolicy {
    all,        // tries all micro-batch sizes
    powerOfTwo, // tries only power of two micro-batch sizes
    undivided,  // returns the same result as cuDNN
    BatchSizePolicySize
  };

  const std::string getGPUName();
  std::string convDataTypesToString(const cudnnDataType_t xType,
				    const cudnnDataType_t yType,
				    const cudnnDataType_t filterType,
				    const cudnnDataType_t computeType);
  std::string dataTypeToString(const cudnnDataType_t dataType);
#if CUDNN_HAS_MATHTYPE
  std::string mathTypeToString(const cudnnMathType_t mathType);
#endif

  // cuDNN generic wrappers
  cudnnStatus_t cudnnConvolutionGeneric(cudnnHandle_t handle,
					const cudnnTensorDescriptor_t xDesc,
					const cudnnTensorDescriptor_t yDesc,
					const cudnnFilterDescriptor_t wDesc,
					const cudnnConvolutionDescriptor_t convDesc,
					void *x, void *y, void *w, void *workspace,
					const void *alpha, const void *beta,
					const int algo, const size_t workspaceSize,
					const ConvType convType);

  class cudnnConvolutionGenericAlgoPerf_t {
  public:
    cudnnConvolutionGenericAlgoPerf_t() {}
    template<typename AlgoPerfType>
    cudnnConvolutionGenericAlgoPerf_t(const AlgoPerfType &perf) {
      status      = perf.status;
      algo        = perf.algo;
      time        = perf.time;
      memory      = perf.memory;
      determinism = perf.determinism;
#if CUDNN_HAS_MATHTYPE
      mathType    = perf.mathType;
#endif
    }

    std::string toString() const {
      std::string s = std::string("status: ") + std::to_string((int) status)
	+ ", algo: " + std::to_string(algo)
	+ ", time: " + std::to_string(time)
	+ ", memory: " + std::to_string(memory)
	+ ", determinism: " + std::to_string(determinism)
#if CUDNN_HAS_MATHTYPE
	+ ", mathType: " + mathTypeToString(mathType)
#endif
	;
      return s;
    }

    cudnnStatus_t status;
    int algo;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
#if CUDNN_HAS_MATHTYPE
    cudnnMathType_t mathType;
#endif
  };

  cudnnStatus_t cudnnFindConvolutionGenericAlgorithm(cudnnHandle_t handle,
						     const cudnnTensorDescriptor_t xDesc,
						     const cudnnTensorDescriptor_t yDesc,
						     const cudnnFilterDescriptor_t wDesc,
						     const cudnnConvolutionDescriptor_t convDesc,
						     const int requestedAlgoCount,
						     int *returnedAlgoCount,
						     cudnnConvolutionGenericAlgoPerf_t *perfResults,
						     const ConvType convType);

  cudnnStatus_t cudnnFindConvolutionGenericAlgorithmEx(cudnnHandle_t handle,
						       const cudnnTensorDescriptor_t xDesc,
						       const cudnnTensorDescriptor_t yDesc,
						       const cudnnFilterDescriptor_t wDesc,
						       const cudnnConvolutionDescriptor_t convDesc,
						       void *x, void *y, void *w, void *workspace,
						       const size_t workspaceInBytes,
						       const int requestedAlgoCount,
						       int *returnedAlgoCount,
						       cudnnConvolutionGenericAlgoPerf_t *perfResults,
						       const ConvType convType);

  // Returns memory size in which any bytes may be read/written by convolution kernel
  // "getAccessibleTensorSizeInBytes(x) <= getTensorSizeInBytes(x)"
  size_t getAccessibleTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc);
  size_t getTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc);
  size_t getFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc);
  size_t getDataTypeSizeInBytes(const cudnnDataType_t dataType);
  int getBatchSizeFromBottomTensorDescriptor(const cudnnTensorDescriptor_t xDesc);
  cudnnTensorFormat_t getTensorFormat(const cudnnTensorDescriptor_t tensorDesc);
  int getTensorDimensionCount(const cudnnTensorDescriptor_t tensorDesc);
  std::string getTensorFormatName(const cudnnTensorFormat_t format);

  // May modify y to avoid additional GPU memory allocation
  float getL2Distance(const float *x, float *y, const int count);
  float getL2Norm(const float *x, const int count);

  void splitStringToInt(const std::string s, std::vector<int> &ret);

  std::string checkEnvironmentVariable(const std::string name, const std::string alt);

  template<typename T>
  void printVector(const std::vector<T> &vec) {
    for(int i = 0; i < vec.size(); i++)
      std::cerr << (i == 0 ? "" : ",") << std::to_string(vec[i]);
    std::cerr << std::endl;
  }

  template<typename T>
  std::string joinToString(const std::vector<T> &vec, const std::string sep=",") {
    std::string s = "";
    for(auto i = vec.begin(); i != vec.end(); i++) {
      if(i != vec.begin())
	s += sep;
      s += (*i);
    }
    return s;
  }

  size_t getFreeDeviceMemorySize();
  long micros();

}

#endif
