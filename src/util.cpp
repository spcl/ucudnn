/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <string>
#include <assert.h>
#include <cublas_v2.h>
#include <sys/time.h>

#include "util.h"

namespace ucudnn {
  LayerId stringToLayerId(const std::string s) {
    return (LayerId) std::hash<std::string>{}(s);
  }

  const std::string getGPUName() {
    cudaDeviceProp prop;
    UCUDNN_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return std::string(prop.name);
  }

  std::string convDataTypesToString(const cudnnDataType_t xType,
				    const cudnnDataType_t yType,
				    const cudnnDataType_t filterType,
				    const cudnnDataType_t computeType) {
    assert(xType == yType && yType == filterType);
    if(xType == CUDNN_DATA_HALF && computeType == CUDNN_DATA_FLOAT)
      return "PSEUDO_HALF";
    assert(xType == computeType);
    if(xType == CUDNN_DATA_HALF)
      return "TRUE_HALF";
    return dataTypeToString(xType);
  }

  std::string dataTypeToString(const cudnnDataType_t dataType) {
    switch(dataType) {
    case CUDNN_DATA_HALF:
      return "HALF";
    case CUDNN_DATA_FLOAT:
      return "FLOAT";
    case CUDNN_DATA_DOUBLE:
      return "DOUBLE";
    default:
      assert(false);
    }
  }

#if CUDNN_HAS_MATHTYPE
  std::string mathTypeToString(const cudnnMathType_t mathType) {
    return std::string(mathType == CUDNN_TENSOR_OP_MATH
		       ? "TENSOR_OP" : "DEFAULT");
  }
#endif

  cudnnStatus_t cudnnConvolutionGeneric(cudnnHandle_t handle,
					const cudnnTensorDescriptor_t xDesc,
					const cudnnTensorDescriptor_t yDesc,
					const cudnnFilterDescriptor_t wDesc,
					const cudnnConvolutionDescriptor_t convDesc,
					void *x, void *y, void *w, void *workspace,
					const void *alpha, const void *beta,
					const int algo, const size_t workspaceSize,
					const ConvType convType) {

    switch(convType) {
    case Forward:
      return cudnnConvolutionForward(handle,
				     alpha,
				     xDesc, x,
				     wDesc, w,
				     convDesc,
				     (cudnnConvolutionFwdAlgo_t) algo, workspace, workspaceSize,
				     beta,
				     yDesc, y);
      break;

    case BackwardData:
      return cudnnConvolutionBackwardData(handle,
					  alpha,
					  wDesc, w,
					  yDesc, y,
					  convDesc,
					  (cudnnConvolutionBwdDataAlgo_t) algo, workspace, workspaceSize,
					  beta,
					  xDesc, x);
      break;

    case BackwardFilter:
      return cudnnConvolutionBackwardFilter(handle,
					    alpha,
					    xDesc, x,
					    yDesc, y,
					    convDesc,
					    (cudnnConvolutionBwdFilterAlgo_t) algo, workspace, workspaceSize,
					    beta,
					    wDesc, w);
      break;

    default:
      assert(false);
    }
  }

  cudnnStatus_t cudnnFindConvolutionGenericAlgorithm(cudnnHandle_t handle,
						     const cudnnTensorDescriptor_t xDesc,
						     const cudnnTensorDescriptor_t yDesc,
						     const cudnnFilterDescriptor_t wDesc,
						     const cudnnConvolutionDescriptor_t convDesc,
						     const int requestedAlgoCount,
						     int *returnedAlgoCount,
						     cudnnConvolutionGenericAlgoPerf_t *perfResults,
						     const ConvType convType) {

    cudnnStatus_t ret;
    switch(convType) {
    case Forward:
      {
	cudnnConvolutionFwdAlgoPerf_t perfs[requestedAlgoCount];
	ret = cudnnFindConvolutionForwardAlgorithm(handle,
						   xDesc,
						   wDesc,
						   convDesc,
						   yDesc,
						   requestedAlgoCount,
						   returnedAlgoCount,
						   perfs);
	for(int i = 0; i < *returnedAlgoCount; i++)
	  perfResults[i] = perfs[i];
      }
      break;

    case BackwardData:
      {
	cudnnConvolutionBwdDataAlgoPerf_t perfs[requestedAlgoCount];
	ret = cudnnFindConvolutionBackwardDataAlgorithm(handle,
							wDesc,
							yDesc,
							convDesc,
							xDesc,
							requestedAlgoCount,
							returnedAlgoCount,
							perfs);
	for(int i = 0; i < *returnedAlgoCount; i++)
	  perfResults[i] = perfs[i];
      }
      break;

    case BackwardFilter:
      {
	cudnnConvolutionBwdFilterAlgoPerf_t perfs[requestedAlgoCount];
	ret = cudnnFindConvolutionBackwardFilterAlgorithm(handle,
							  xDesc,
							  yDesc,
							  convDesc,
							  wDesc,
							  requestedAlgoCount,
							  returnedAlgoCount,
							  perfs);
	for(int i = 0; i < *returnedAlgoCount; i++)
	  perfResults[i] = perfs[i];
      }
      break;

    default:
      assert(false);
    }

    return ret;
  }

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
						       const ConvType convType) {

    cudnnStatus_t ret;
    switch(convType) {
    case Forward:
      {
	cudnnConvolutionFwdAlgoPerf_t perfs[requestedAlgoCount];
	ret = cudnnFindConvolutionForwardAlgorithmEx(handle,
						     xDesc, x,
						     wDesc, w,
						     convDesc,
						     yDesc, y,
						     requestedAlgoCount,
						     returnedAlgoCount,
						     perfs,
						     workspace, workspaceInBytes);
	for(int i = 0; i < *returnedAlgoCount; i++)
	  perfResults[i] = perfs[i];
      }
      break;

    case BackwardData:
      {
	cudnnConvolutionBwdDataAlgoPerf_t perfs[requestedAlgoCount];
	ret = cudnnFindConvolutionBackwardDataAlgorithmEx(handle,
							  wDesc, w,
							  yDesc, y,
							  convDesc,
							  xDesc, x,
							  requestedAlgoCount,
							  returnedAlgoCount,
							  perfs,
							  workspace, workspaceInBytes);
	for(int i = 0; i < *returnedAlgoCount; i++)
	  perfResults[i] = perfs[i];
      }
      break;

    case BackwardFilter:
      {
	cudnnConvolutionBwdFilterAlgoPerf_t perfs[requestedAlgoCount];
	ret = cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,
							    xDesc, x,
							    yDesc, y,
							    convDesc,
							    wDesc, w,
							    requestedAlgoCount,
							    returnedAlgoCount,
							    perfs,
							    workspace, workspaceInBytes);
	for(int i = 0; i < *returnedAlgoCount; i++)
	  perfResults[i] = perfs[i];
      }
      break;

    default:
      assert(false);
    }

    return ret;
  }

  size_t getAccessibleTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc) {
    int n, c, h, w, nStride, cStride, hStride, wStride;
    cudnnDataType_t dataType;
    UCUDNN_CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc, &dataType,
						  &n, &c, &h, &w,
						  &nStride, &cStride, &hStride, &wStride));
    assert(wStride < hStride && hStride < cStride && cStride < nStride); // assert NCHW for now
    const size_t chw = (size_t) c*h*w;
    return (n*nStride - (nStride - chw))*getDataTypeSizeInBytes(dataType);
  }

  size_t getTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc) {
    size_t size;
    UCUDNN_CUDNN_CHECK(cudnnGetTensorSizeInBytes(tensorDesc, &size));
    return size;
  }

  size_t getFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc) {
    int k, c, h, w;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    UCUDNN_CUDNN_CHECK(cudnnGetFilter4dDescriptor(filterDesc,
						  &dataType, &format,
						  &k, &c, &h, &w));
    return (size_t) k*c*h*w*getDataTypeSizeInBytes(dataType);
  }

  size_t getDataTypeSizeInBytes(const cudnnDataType_t dataType) {
    switch(dataType) {
    case CUDNN_DATA_FLOAT:  return 4;
    case CUDNN_DATA_DOUBLE: return 8;
    case CUDNN_DATA_HALF:   return 2;
    case CUDNN_DATA_INT8:   return 1;
    case CUDNN_DATA_INT32:  return 4;
    default:
      assert(false);
    }
  }

  int getBatchSizeFromBottomTensorDescriptor(const cudnnTensorDescriptor_t xDesc) {
    int n, c, h, w, nStride, cStride, hStride, wStride;
    cudnnDataType_t dataType;
    UCUDNN_CUDNN_CHECK(cudnnGetTensor4dDescriptor(xDesc, &dataType,
						  &n, &c, &h, &w,
						  &nStride, &cStride, &hStride, &wStride));
    return n;
  }

  cudnnTensorFormat_t getTensorFormat(cudnnTensorDescriptor_t tensorDesc) {
    cudnnDataType_t dataType;
    int n, c, h, w, ns, cs, hs, ws;
    UCUDNN_CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc, &dataType, &n, &c, &h, &w,
						  &ns, &cs, &hs, &ws));
    if(   ns == h * w * c
	  && cs == 1
	  && hs == w * c
	  && ws == c)
      return CUDNN_TENSOR_NHWC;
    else if(   ns == c * h * w
	       && cs == h * w
	       && hs == w
	       && ws == 1)
      return CUDNN_TENSOR_NCHW;
    else {
      std::cout << "Invalid tensor format:"
		<< "[" << n
		<< "," << c
		<< "," << h
		<< "," << w
		<< "] "
		<< "[" << ns
		<< "," << cs
		<< "," << hs
		<< "," << ws
		<< "] " << std::endl;
      assert(false);
    }
  }

  int getTensorDimensionCount(const cudnnTensorDescriptor_t tensorDesc) {
    const int nbDimsRequested = 0xFF;
    cudnnDataType_t dataType;
    int nbDims;
    int dimA[nbDimsRequested], strideA[nbDimsRequested];
    UCUDNN_CUDNN_CHECK(cudnnGetTensorNdDescriptor(tensorDesc,
						  nbDimsRequested,
						  &dataType,
						  &nbDims,
						  dimA,
						  strideA));
    return nbDims;
  }

  std::string getTensorFormatName(const cudnnTensorFormat_t format) {
    switch(format) {
    case CUDNN_TENSOR_NCHW:
      return std::string("NCHW");
    case CUDNN_TENSOR_NHWC:
      return std::string("NHWC");
    case CUDNN_TENSOR_NCHW_VECT_C:
      return std::string("NCHW_VECT_C");
    default:
      assert(false);
    }
  }

  float getL2Distance(const float *x, float *y, const int count) {
    float nrm2;
    cublasHandle_t cublas;
    UCUDNN_CUBLAS_CHECK(cublasCreate(&cublas));
    const float minus = -1.0;
    if(y)
      UCUDNN_CUBLAS_CHECK(cublasSaxpy(cublas, count, &minus,
				      x, 1,
				      y, 1)); // y = -1 * x + y = y - x
    UCUDNN_CUBLAS_CHECK(cublasSnrm2(cublas, count, (y ? y : x), 1, &nrm2));
    UCUDNN_CUBLAS_CHECK(cublasDestroy(cublas));
    return nrm2;
  }

  float getL2Norm(const float *x, const int count) {
    getL2Distance(x, nullptr, count);
  }

  void splitStringToInt(const std::string str, std::vector<int> &ret) {
    std::string s = str;
    while(true) {
      const size_t pos = s.find(",");
      const bool isEnd = (pos == std::string::npos);
      ret.push_back(std::stoi(isEnd ? s : s.substr(0, pos)));
      if(isEnd)
	return;
      s = s.substr(pos+1);
    }
  }

  std::string checkEnvironmentVariable(const std::string name, const std::string alt) {
    const char *env = std::getenv(name.c_str());
    if(env) {
#ifdef UCUDNN_DEBUG_OUTPUT_ENV
      std::cerr << "Using environment variable " << name << " = \"" << env << "\"." << std::endl;
#endif
      return std::string(env);
    }
#ifdef UCUDNN_DEBUG_OUTPUT_ENV
    std::cerr << "Environment variable " << name << " is not set. Using \"" << alt << "\" instead." << std::endl;
#endif
    return alt;
  }

  size_t getFreeDeviceMemorySize() {
    size_t free, total;
    UCUDNN_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
  }

  long micros() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec*1e6 + tv.tv_usec;
  }

}
