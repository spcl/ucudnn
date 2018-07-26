/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_OPTIMIZER_H_
#define UCUDNN_OPTIMIZER_H_

#include <vector>
#include <memory>

#include <cudnn.h>

#include "convConfig.h"
#include "convParam.h"
#include "convDatabase.h"

namespace ucudnn {

  typedef struct {
    void *x, *y, *w, *workspace;
  } FindConvAlgorithmExPointers;

  class Optimizer {
  public:

    static std::string batchSizePolicyToString(const BatchSizePolicy batchSizePolicy);
    static BatchSizePolicy stringToBatchSizePolicy(const std::string str);
    static void batchSizePolicyToBatchSizes(const BatchSizePolicy batchSizePolicy,
					    const int batchSize,
					    std::vector<int> &batchSizes);

    Optimizer(const cudnnHandle_t handle, const ConvParam convParam, std::shared_ptr<ConvDatabase> convDatabase=nullptr);
    ~Optimizer();

    // returns all available cuDNN convolution kernel configurations which need less than workspaceSize
    std::vector<std::shared_ptr<KernelConfig> >
    getDesirableKernelConfigs(const cudnnDataType_t computeType,
			      const ConvType convType,
			      const int batchSize,
			      const size_t workspaceSize,
			      FindConvAlgorithmExPointers *ptrs,
			      const bool printResults=false);
    std::vector<std::shared_ptr<KernelConfig> >
    getDesirableKernelConfigs(const ConvType convType,
			      const int batchSize,
			      const size_t workspaceSize,
			      FindConvAlgorithmExPointers *ptrs,
			      const bool printResults=false);

    // returns the fastest combination of cuDNN convolution kernel configurations which need less than workspaceSize
    // if devices (list of device ID) are specified, do behcnmark in parallel with specified devices
    std::shared_ptr<ConvConfig> getBestConvConfig(const ConvType convType,
						  const int batchSize,
						  const size_t workspaceSize,
						  const BatchSizePolicy batchSizePolicy,
						  const std::vector<int> &devices,
						  FindConvAlgorithmExPointers *ptrs);

    // returns "desirable" configuration sets
    // if best is true, return at most one fastest configuration set
    std::pair<std::vector<std::shared_ptr<ConvConfig> >, std::shared_ptr<KernelConfig> >
    getDesirableConvConfigs(const ConvType convType,
			    const int batchSize,
			    const size_t workspaceSize,
			    const BatchSizePolicy batchSizePolicy,
			    const bool best,
			    const std::vector<int> &devices,
			    FindConvAlgorithmExPointers *ptrs=nullptr);

  private:
    std::vector<std::vector<std::shared_ptr<KernelConfig> > > getDesirableKernelConfigBench(const ConvType convType,
											    const int batchSize,
											    const size_t workspaceSize,
											    const BatchSizePolicy batchSizePolicy,
											    const std::vector<int> &devices,
											    FindConvAlgorithmExPointers *ptrs,
											    const bool best);

    const cudnnHandle_t handle_;
    const ConvParam convParam_;
    std::shared_ptr<ConvDatabase> convDatabase_;

    cudnnTensorDescriptor_t xDesc_, yDesc_;
    cudnnFilterDescriptor_t wDesc_;
    cudnnConvolutionDescriptor_t convDesc_;
  };

}

#endif
