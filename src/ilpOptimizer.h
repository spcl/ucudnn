/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_ILP_OPTIMIZER_H_
#define UCUDNN_ILP_OPTIMIZER_H_

#include <cudnn.h>

#include "convConfig.h"
#include "convParam.h"
#include "optimizer.h"

namespace ucudnn {

  class ILPOptimizer {
  public:
    ILPOptimizer(cudnnHandle_t handle,
		 std::vector<std::pair<ConvParam, ConvType> > &kernelList)
      : handle_(handle), kernelList_(kernelList) {}

    std::vector<std::shared_ptr<ConvConfig> > optimize(const size_t workspaceSize,
						       const BatchSizePolicy batchSizePolicy,
						       const std::vector<int> &devices);

  private:
    cudnnHandle_t handle_;
    std::vector<std::pair<ConvParam, ConvType> > kernelList_;

  };

}

#endif
