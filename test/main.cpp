/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <iostream>
#include <fstream>
#include <typeinfo>
#include <regex>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cudnn.h>
#include <assert.h>

#include "vcudnn.h"
#include "util.h"

void cudaCheck(cudaError_t error) {
  if(error != cudaSuccess) {
    std::cerr << "cuda error: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
}

void cudnnCheck(cudnnStatus_t status) {
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cerr << "cudnn error: " << status << std::endl;
    exit(1);
  }
}

void curandCheck(curandStatus_t status) {
  if(status != CURAND_STATUS_SUCCESS) {
    std::cerr << "curand error: " << status << std::endl;
    exit(1);
  }
}

void fillNormal(float *ary, const size_t size) {
  curandGenerator_t generator;
  curandCheck(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  curandCheck(curandGenerateNormal(generator, ary, size, 0.0, 1.0));
  curandCheck(curandDestroyGenerator(generator));
}

cudnnConvolutionFwdAlgo_t test(cudnnHandle_t cudnn, unsigned long workspaceSize,
			       const int iFilter, const int oFilter,
			       const int kernel_w, const int kernel_h,
			       const int stride_w, const int stride_h,
			       const int pad_w, const int pad_h,
			       const int iWidth, const int iHeight,
			       const int batchSize, const int id,
			       const bool getAlgorithm, const bool runConvolution,
			       const bool isHalf, const bool isTensorOp,
			       const cudnnTensorFormat_t xFormat,
			       const cudnnTensorFormat_t yFormat,
			       const cudnnTensorFormat_t wFormat,
			       cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t) 0) {

  const cudnnDataType_t dataType = (isHalf ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT);

  cudnnTensorDescriptor_t xDesc, yDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnCheck(cudnnCreateTensorDescriptor(&xDesc));
  cudnnCheck(cudnnCreateTensorDescriptor(&yDesc));
  cudnnCheck(cudnnCreateFilterDescriptor(&wDesc));
  cudnnCheck(cudnnCreateConvolutionDescriptor(&convDesc));

  cudnnCheck(cudnnSetConvolution2dDescriptor(convDesc,
					     pad_h, pad_w, stride_h, stride_w, 1, 1,
					     CUDNN_CROSS_CORRELATION, dataType));
  if(isTensorOp) {
#if CUDNN_HAS_MATHTYPE
    cudnnCheck(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
#else
    std::cerr << "current cuDNN does not support math types!" << std::endl;
    exit(1);
#endif
  }

  cudnnCheck(cudnnSetFilter4dDescriptor(wDesc, dataType, wFormat,
					oFilter, iFilter, kernel_h, kernel_w));

  if(xFormat == CUDNN_TENSOR_NHWC) {
    cudnnCheck(cudnnSetTensor4dDescriptorEx(xDesc, dataType,
					    batchSize, iFilter, iHeight, iWidth,
					    iHeight*iWidth*iFilter, 1, iWidth*iFilter, iFilter));
  } else if(xFormat == CUDNN_TENSOR_NCHW) {
    cudnnCheck(cudnnSetTensor4dDescriptorEx(xDesc, dataType,
					    batchSize, iFilter, iHeight, iWidth,
					    iFilter*iHeight*iWidth, iHeight*iWidth, iWidth, 1));
  } else {
    assert(false);
  }

  int oWidth, oHeight;
  {
    int n_out;
    int c_out;
    cudnnCheck(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc,
						     &n_out, &c_out,
						     &oHeight, &oWidth));
    assert(n_out == batchSize);
    assert(c_out == oFilter);
  }

  if(yFormat == CUDNN_TENSOR_NHWC) {
    cudnnCheck(cudnnSetTensor4dDescriptorEx(yDesc, dataType,
					    batchSize, oFilter, oHeight, oWidth,
					    oHeight*oWidth*oFilter, 1, oWidth*oFilter, oFilter));
  } else if(yFormat == CUDNN_TENSOR_NCHW) {
    cudnnCheck(cudnnSetTensor4dDescriptorEx(yDesc, dataType,
					    batchSize, oFilter, oHeight, oWidth,
					    oFilter*oHeight*oWidth, oHeight*oWidth, oWidth, 1));
  } else {
    assert(false);
  }

  UcudnnHandle_t handle;
  cudnnCheck(cudnnCreate(&handle));

  if(getAlgorithm)
    cudnnCheck(cudnnGetConvolutionForwardAlgorithm(handle,
						   xDesc, wDesc, convDesc, yDesc,
						   CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
						   workspaceSize,
						   &algo,
						   (ucudnn::LayerId) (unsigned long) id));

  if(runConvolution) {
    void *x, *y, *w, *workspace;

    cudaCheck(cudaMalloc(&x, ucudnn::getTensorSizeInBytes(xDesc)));
    cudaCheck(cudaMalloc(&y, ucudnn::getTensorSizeInBytes(yDesc)));
    cudaCheck(cudaMalloc(&w, ucudnn::getFilterSizeInBytes(wDesc)));
    cudaCheck(cudaMalloc(&workspace, workspaceSize));

    if(dataType == CUDNN_DATA_FLOAT) {
      fillNormal((float *) x, ucudnn::getTensorSizeInBytes(xDesc) / sizeof(float));
      fillNormal((float *) y, ucudnn::getTensorSizeInBytes(yDesc) / sizeof(float));
      fillNormal((float *) w, ucudnn::getFilterSizeInBytes(wDesc) / sizeof(float));
    } else {
      // TODO: set proper random values in half format
      cudaCheck(cudaMemset(x, 1, ucudnn::getTensorSizeInBytes(xDesc)));
      cudaCheck(cudaMemset(y, 2, ucudnn::getTensorSizeInBytes(yDesc)));
      cudaCheck(cudaMemset(w, 3, ucudnn::getFilterSizeInBytes(wDesc)));
      cudaCheck(cudaMemset(workspace, 4, workspaceSize));
    }

    const float alpha = 1.0;
    const float beta = 0.0;
    cudnnCheck(cudnnConvolutionForward(
				       handle,
				       &alpha,
				       xDesc, x,
				       wDesc, w,
				       convDesc,
				       algo,
				       workspace,
				       workspaceSize,
				       &beta,
				       yDesc, y,
				       (ucudnn::LayerId) (unsigned long) id));

    cudaCheck(cudaFree(x));
    cudaCheck(cudaFree(y));
    cudaCheck(cudaFree(w));
    cudaCheck(cudaFree(workspace));
  }

  cudnnCheck(cudnnDestroyTensorDescriptor(xDesc));
  cudnnCheck(cudnnDestroyTensorDescriptor(yDesc));
  cudnnCheck(cudnnDestroyFilterDescriptor(wDesc));
  cudnnCheck(cudnnDestroyConvolutionDescriptor(convDesc));

  cudnnCheck(cudnnDestroy(handle));

  return algo;
}

int main(int argc, char **argv) {
  // ./main [Workspace size [MiB]] ["float" or "half"] ["default" or "tensor_op"] ["NCHW" or "NHWC"]*3 [path/to/bench.csv]

  {
    // Skip this test if no GPUs are found.
    int count;
    cudaCheck(cudaGetDeviceCount(&count));
    if(count == 0)
      exit(200);
  }

  argc--;
  argv++;
  const char *arg_ws    = (argc-- > 0 ? *(argv++) : nullptr);
  const char *arg_type  = (argc-- > 0 ? *(argv++) : nullptr);
  const char *arg_mType = (argc-- > 0 ? *(argv++) : nullptr);
  const char *arg_xFormat = (argc-- > 0 ? *(argv++) : nullptr);
  const char *arg_yFormat = (argc-- > 0 ? *(argv++) : nullptr);
  const char *arg_wFormat = (argc-- > 0 ? *(argv++) : nullptr);
  const char *arg_bench = (argc-- > 0 ? *(argv++) : nullptr);

  const unsigned long workspaceSize = (arg_ws   ? std::atoll(arg_ws) : 64UL)*1024*1024;
  const bool isHalf                 = (arg_type ? (std::strcmp(arg_type, "half") == 0) : false);
  const bool isTensorOp             = (arg_mType ? (std::strcmp(arg_mType, "tensor_op") == 0) : false);
  const cudnnTensorFormat_t xFormat = (arg_xFormat ? (std::strcmp(arg_xFormat, "NCHW") == 0 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC) : CUDNN_TENSOR_NCHW);
  const cudnnTensorFormat_t yFormat = (arg_yFormat ? (std::strcmp(arg_yFormat, "NCHW") == 0 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC) : CUDNN_TENSOR_NCHW);
  const cudnnTensorFormat_t wFormat = (arg_wFormat ? (std::strcmp(arg_wFormat, "NCHW") == 0 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC) : CUDNN_TENSOR_NCHW);
  const std::string benchCSV        = (arg_bench ? std::string(arg_bench) : "../test/bench/alexnet.csv");

  std::cerr << "workspaceSize: " << workspaceSize
	    << ", isHalf: " << isHalf
	    << ", isTensorOp: " << isTensorOp
	    << ", bench CSV file: " << benchCSV
	    << ", xFormat: " << ucudnn::getTensorFormatName(xFormat)
	    << ", yFormat: " << ucudnn::getTensorFormatName(yFormat)
	    << ", wFormat: " << ucudnn::getTensorFormatName(wFormat)
	    << std::endl;

  cudnnHandle_t cudnn;
  cudnnCheck(cudnnCreate(&cudnn));

  // AlexNet

  std::vector<cudnnConvolutionFwdAlgo_t> algos;

  std::ifstream ifs(benchCSV);
  assert(!ifs.fail());
  std::string line;
  std::getline(ifs, line); // Ignore the first line (column names)
  int i = 0;
  while(std::getline(ifs, line)) {
    std::istringstream iss(line);
    std::string elem;
    std::getline(iss, elem, ',');

    // W,H,C_in,N,C_out,kernel_w,kernel_h,pad_w,pad_h,stride_w,stride_h
    std::vector<int> params;
    while(std::getline(iss, elem, ','))
      params.push_back(std::stoi(elem));
    assert(params.size() == 11);

    const int w        = params[0];
    const int h        = params[1];
    const int c_in     = params[2];
    const int n        = params[3];
    const int c_out    = params[4];
    const int kernel_w = params[5];
    const int kernel_h = params[6];
    const int pad_w    = params[7];
    const int pad_h    = params[8];
    const int stride_w = params[9];
    const int stride_h = params[10];

    test(cudnn, workspaceSize,
	 c_in, c_out,
	 kernel_w, kernel_h,
	 stride_w, stride_h,
	 pad_w, pad_h,
	 w, h,
    	 n, i, true, true, isHalf, isTensorOp,
	 xFormat, yFormat, wFormat);
    i++;
  }
  cudnnCheck(cudnnDestroy(cudnn));
}
