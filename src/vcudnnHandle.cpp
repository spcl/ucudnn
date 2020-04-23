/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <stdint.h>
#include <fstream>
#include <string>

#include <cudnn.h>
#ifdef UCUDNN_USE_GLPK
#include <glpk.h>
#endif
#ifdef UCUDNN_USE_SQLITE
#include <sqlite3.h>
#endif

#include "convParam.h"
#include "optimizer.h"
#include "optCache.h"
#include "ilpOptimizer.h"
#include "vcudnnHandle.h"
#include "util.h"
#include "safeWorkspace.h"

namespace ucudnn {
  OptCache UcudnnHandle_t::optCache_;

  void UcudnnHandle_t::init() {
    fo.open("vucudnn.log", std::ofstream::out | std::ofstream::app);

//    optimizerBatchSizePolicy_ = Optimizer::stringToBatchSizePolicy(checkEnvironmentVariable("UCUDNN_BATCH_SIZE_POLICY",
//											    "powerOfTwo"));

//    const std::string deviceEnv = checkEnvironmentVariable("UCUDNN_BENCHMARK_DEVICES",
//							   "");
//    if(deviceEnv.length() > 0) {
//      if(deviceEnv == "all") {
//	int deviceCount;
//	UCUDNN_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
//	for(int i = 0; i < deviceCount; i++)
//	  devices_.push_back(i);
//      } else
//	splitStringToInt(deviceEnv, devices_);
//#ifdef UCUDNN_DEBUG_OUTPUT
//      std::cerr << "Using device "
//		<< std::accumulate(devices_.begin(), devices_.end(),
//				   std::string(""),
//				   [](std::string c1, int c2) {
//				     return (c1 == "" ? "" : c1 + ",") + std::to_string(c2);
//				   })
//		<< "." << std::endl;
//#endif
//    }

//    const long ret = std::atol(checkEnvironmentVariable("UCUDNN_TOTAL_WORKSPACE_SIZE", "0").c_str());
//    ilp_ = (ret > 0);
//    staticWorkspaceSize_ = ret;

//    const std::string dbEnv = checkEnvironmentVariable("UCUDNN_DATABASE",
//						       "");
//    if(dbEnv.length() > 0) {
//      std::cerr << "Using database " << dbEnv << "." << std::endl;
//      database_ = std::shared_ptr<ConvDatabase>(new ConvDatabase(dbEnv));
//    } else
//      database_ = nullptr;
  }

  UcudnnHandle_t::UcudnnHandle_t(const UcudnnHandle_t &handle) {
    // According to "2.4. Thread Safety" of cuDNN documentation,
    // the cuDNN handle can be copied as long as multiple cuDNN functions are called simultaneously.
    // So we can simply copy the internal handle and discriptors.
    handle_ = handle.handle_;
    xDesc_ = handle.xDesc_;
    yDesc_ = handle.yDesc_;
    convDesc_ = handle.convDesc_;
    optimizerBatchSizePolicy_ = handle.optimizerBatchSizePolicy_;
    devices_ = handle.devices_;

    device_ = handle.device_;
    ilp_ = handle.ilp_;
    staticWorkspaceSize_ = handle.staticWorkspaceSize_;

    database_ = handle.database_;
  }

  UcudnnHandle_t::~UcudnnHandle_t() {
  }

  void UcudnnHandle_t::create() {
//    UCUDNN_CUDNN_CHECK(cudnnCreate(&handle_));
//    UCUDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc_));
//    UCUDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc_));
//    UCUDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc_));
//    UCUDNN_CUDA_CHECK(cudaGetDevice(&device_));

//#ifdef UCUDNN_DEBUG_OUTPUT
//    cudaDeviceProp deviceProp;
//    UCUDNN_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device_));
//    std::cerr << "---" << std::endl;
//    std::cerr << "Device: " << deviceProp.name << std::endl;
//    std::cerr << "CUDA version: " << CUDA_VERSION << std::endl;
//    std::cerr << "cuDNN version: " << CUDNN_VERSION << std::endl;
//    std::cerr << "u-cuDNN version: " << UCUDNN_VERSION
//#ifdef UCUDNN_COMMIT_ID
//	      << " (commit " << UCUDNN_COMMIT_ID << ")"
//#endif
//	      << std::endl;
//#ifdef UCUDNN_USE_GLPK
//    std::cerr << "GLPK version: " << GLP_MAJOR_VERSION << "." << GLP_MINOR_VERSION << std::endl;
//#endif
//#ifdef UCUDNN_USE_SQLITE
//    std::cerr << "SQLite version: " << SQLITE_VERSION << std::endl;
//#endif
//    std::cerr << "---" << std::endl << std::endl;
//#endif
  }

  void UcudnnHandle_t::destroy() {
//    if(database_ != nullptr) {
//      database_->setLayerParams(optCache_);
//    }

//    UCUDNN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc_));
//    UCUDNN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc_));
//    UCUDNN_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc_));
//    UCUDNN_CUDNN_CHECK(cudnnDestroy(handle_));
  }

  void UcudnnHandle_t::getAlgorithm(const ConvParam convParam, const ConvType convType, const size_t workspaceSize,
				    const LayerId layerId) {
    findAlgorithmEx(convParam, convType,
		    nullptr, nullptr, nullptr, nullptr,
		    workspaceSize,
		    layerId);
  }

  std::shared_ptr<ConvConfig> UcudnnHandle_t::findAlgorithmEx(const ConvParam convParam, const ConvType convType,
							      void *x, void *y, void *w, void *workspace,
							      const size_t workspaceSize,
							      const LayerId layerId) {
    if(ilp_) {
      optCache_.setConvConfig(convParam, convType, layerId);

      // return dummy ConvConfig
      std::shared_ptr<KernelConfig> kernelConfig =
	std::make_shared<KernelConfig>(convParam.getDefaultBatchSize());
      std::shared_ptr<ConvConfig> convConfig = std::make_shared<ConvConfig>(kernelConfig);
      return convConfig;
    }

    // x, y, w, workspace are nullable since findAlgorithmEx may be called from cudnnGetConvolution*Algorithm
    {
      const std::shared_ptr<ConvConfig> convConfig = optCache_.getConvConfig(convParam, convType, layerId);
      if(convConfig)
	return convConfig;
    }

    FindConvAlgorithmExPointers ptrs;
    ptrs.x = x;
    ptrs.y = y;
    ptrs.w = w;
    ptrs.workspace = workspace;

    Optimizer optimizer(handle_, convParam, database_);
    std::shared_ptr<ConvConfig> convConfig = optimizer.getBestConvConfig(convType,
									 convParam.getDefaultBatchSize(),
									 workspaceSize,
									 optimizerBatchSizePolicy_,
									 devices_,
									 &ptrs);
    optCache_.setConvConfig(convParam, convType, layerId, convConfig);
    return convConfig;
  }

  size_t UcudnnHandle_t::getWorkspaceSize(const ConvParam convParam, const ConvType convType,
					  const LayerId layerId) {
    if(ilp_)
      return 0;
    std::shared_ptr<ConvConfig> convConfig = optCache_.getConvConfig(convParam, convType, layerId);

    if(convConfig == nullptr) {
      // In case that convConfig is nullptr, the workspace size is requested by the frameworks
      // before the maximum workspace size is provided.
      // So we use zero for the workspace limit.

      const std::string limitEnv = checkEnvironmentVariable("UCUDNN_DEFAULT_WORKSPACE_LIMIT", "");
      size_t limit;
      if(limitEnv.length() > 0)
	limit = std::atol(limitEnv.c_str());
      else
	limit = UCUDNN_DEFAULT_WORKSPACE_LIMIT;

      UCUDNN_WARNING("No workspace limit is provided from the framework. Using UCUDNN_DEFAULT_WORKSPACE_LIMIT="
		     + std::to_string(limit) +  " instead.");
      getAlgorithm(convParam, convType, limit, layerId);
      convConfig = optCache_.getConvConfig(convParam, convType, layerId);
    }

    assert(convConfig);

    return convConfig->memory();
  }

  void UcudnnHandle_t::log(const std::string message) {
    if(fo) {
      fo << message << std::endl;
      //std::cout << message << std::endl;
    }
  }

  // cudnnConvolution*
  cudnnStatus_t UcudnnHandle_t::convolution(const ConvParam convParam, const ConvType convType,
					    const cudnnFilterDescriptor_t wDesc,
					    const cudnnConvolutionDescriptor_t convDesc,
					    void *x, void *y, void *w, void *workspace,
					    const size_t defaultWorkspaceSize,
					    const void *alpha, const void *beta,
					    const LayerId layerId) {
    size_t workspaceSize = defaultWorkspaceSize;

    if(ilp_)
      optCache_.optimizeWD(handle_, staticWorkspaceSize_, optimizerBatchSizePolicy_, devices_);

    std::shared_ptr<ConvConfig> convConfig = nullptr;
    std::shared_ptr<SafeWorkspace> safeWorkspace = nullptr;

    cudaStream_t stream;
    UCUDNN_CUDNN_CHECK(cudnnGetStream(handle_, &stream));

    if(ilp_) {
      workspace = nullptr;

      convConfig = optCache_.getConvConfig(convParam, convType, layerId);
      safeWorkspace = optCache_.getWDWorkspace(convParam, convType, layerId, device_);
      workspace = safeWorkspace->ptr();
      workspaceSize = convConfig->memory();
    } else {
      convConfig = optCache_.getConvConfig(convParam, convType, layerId);
      if(workspaceSize < convConfig->memory()) {
	UCUDNN_WARNING("Insufficient workspace size. Using algorithm 0 instead.");
	convConfig = std::make_shared<ConvConfig>(std::make_shared<KernelConfig>(convParam.getDefaultBatchSize()));
      }
    }

    assert(convConfig);

    std::vector<std::shared_ptr<KernelConfig> > kernels = convConfig->kernelConfigs();

#ifdef UCUDNN_DEBUG_EQUIVALENCE_TEST
    convParam.setXDesc(xDesc_, convParam.getDefaultBatchSize());
    convParam.setYDesc(yDesc_, convParam.getDefaultBatchSize());

    size_t outSize;
    void *ucudnnOut, *trueOut;
    cudnnDataType_t outDataType;
    switch(convType) {
    case Forward:
      outSize = getAccessibleTensorSizeInBytes(yDesc_);
      ucudnnOut = y;
      outDataType = convParam.getYDataType();
      break;
    case BackwardData:
      outSize = getAccessibleTensorSizeInBytes(xDesc_);
      ucudnnOut = x;
      outDataType = convParam.getXDataType();
      break;
    case BackwardFilter:
      outSize = getFilterSizeInBytes(wDesc);
      ucudnnOut = w;
      outDataType = convParam.getWDataType();
      break;
    default:
      assert(false);
    }
    assert(outDataType == CUDNN_DATA_FLOAT);
    assert((outSize % getDataTypeSizeInBytes(outDataType)) == 0);
    const int outCount = outSize / getDataTypeSizeInBytes(outDataType);
    UCUDNN_CUDA_CHECK(cudaMalloc(&trueOut, outSize));
    UCUDNN_CUDA_CHECK(cudaMemcpy(trueOut, ucudnnOut, outSize, cudaMemcpyDeviceToDevice));

    UCUDNN_CUDNN_CHECK(cudnnConvolutionGeneric(handle_,
					       xDesc_, yDesc_, wDesc, convDesc,
					       (convType == BackwardData   ? trueOut : x),
					       (convType == Forward        ? trueOut : y),
					       (convType == BackwardFilter ? trueOut : w),
					       nullptr,
					       alpha, beta,
					       0, 0, convType));

    UCUDNN_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

    double beta_one;
    convParam.setScalingFactor(&beta_one, 1.0);

    ConvParam convParamWithCT = convParam;
    int computedBatchSize = 0;
    for(int i = 0; i < kernels.size(); i++) {
      std::shared_ptr<KernelConfig> kernel = kernels[i];

      convParam.setXDesc(xDesc_, kernel->batchSize());
      convParam.setYDesc(yDesc_, kernel->batchSize());

      convParamWithCT.setComputeType(kernel->computeType());
      convParamWithCT.setConvDesc(convDesc_);

      uint8_t *Ux = (uint8_t *) x+convParam.getXOffset(computedBatchSize);
      uint8_t *Uy = (uint8_t *) y+convParam.getYOffset(computedBatchSize);

      UCUDNN_CUDNN_CHECK(cudnnConvolutionGeneric(handle_,
						 xDesc_, yDesc_, wDesc, convDesc_,
						 Ux, Uy, w, workspace,
						 alpha, beta,
						 kernel->algo(), workspaceSize,
						 convType));

      computedBatchSize += kernel->batchSize();

      // synchronize after each convolution is required to protect tensor/filter/convolution descriptors and workspaces
      UCUDNN_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

#ifdef UCUDNN_DEBUG_EQUIVALENCE_TEST
    const float l2nrm = getL2Norm((float *) trueOut, outCount);
    const float l2dist = getL2Distance((float *) ucudnnOut, (float *) trueOut, outCount);
    std::cerr << "Convolution equivalence test (" << convTypeToString(convType)
	      << ", " << convParam.toString() << "):" << std::endl

	      << "   # of u-cuDNN kernel(s): " << kernels.size() << std::endl

	      << std::scientific << std::setprecision(4)
	      << "   alpha = " << convParam.getScalingFactor((double *) alpha)
	      << ", beta = " << convParam.getScalingFactor((double *) beta) << std::endl

	      << "   L2(cuDNN,u-cuDNN)/L2(cuDNN) = " << l2dist << " / " << l2nrm << std::endl
	      << "                               = " << (l2dist / l2nrm) << std::endl;

    UCUDNN_CUDA_CHECK(cudaFree(trueOut));
#endif

    if(ilp_ && safeWorkspace)
      safeWorkspace->setPostKernelRecord(stream);

    return CUDNN_STATUS_SUCCESS;
  }

}
