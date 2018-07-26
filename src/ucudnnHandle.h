/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_UCUDNNHANDLE_H_
#define UCUDNN_UCUDNNHANDLE_H_

#include <unordered_map>
#include <cudnn.h>

#include "convParam.h"
#include "optimizer.h"
#include "optCache.h"
#include "util.h"
#include "convDatabase.h"

namespace ucudnn {

  class UcudnnHandle_t {
  public:

    UcudnnHandle_t() { init(); };
    // Some frameworks including NVIDIA Caffe initializes cudnnHandle_t as nullptr,
    // since cudnnHandle_t is alias type for a cudnnContext pointer.
    // So UcudnnHandle_t have a dummy parameter to support such initialization.
    // ptr is totall ignored in the constructor.
    UcudnnHandle_t(const void *ptr) { init(); }
    UcudnnHandle_t(const UcudnnHandle_t &handle);
    ~UcudnnHandle_t();

    operator cudnnHandle_t() const { return handle_; }
    void allocateTensorDescriptors(const int size);

    void setOptimizerBatchSizePolicy(const BatchSizePolicy optimizerBatchSizePolicy) {
      optimizerBatchSizePolicy_ = optimizerBatchSizePolicy;
    }
    void setDevices(const std::vector<int> devices) {
      devices_ = devices;
    }

    void create();
    void destroy();
    cudnnStatus_t convolution(const ConvParam convParam, const ConvType convType,
			      const cudnnFilterDescriptor_t wDesc,
			      const cudnnConvolutionDescriptor_t convDesc,
			      void *x, void *y, void *w, void *workspace,
			      const size_t workspaceSize,
			      const void *alpha, const void *beta,
			      const LayerId layerId);

    void getAlgorithm(const ConvParam convParam, const ConvType convType, const size_t workspaceSize, const LayerId layerId);
    std::shared_ptr<ConvConfig> findAlgorithmEx(const ConvParam convParam, const ConvType convType,
						void *x, void *y, void *w, void *workspace,
						const size_t workspaceSize, const LayerId layerId);
    size_t getWorkspaceSize(const ConvParam convParam, const ConvType convType, const LayerId layerId);

  private:
    void init();

    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t xDesc_, yDesc_;
    cudnnConvolutionDescriptor_t convDesc_;
    BatchSizePolicy optimizerBatchSizePolicy_;
    std::vector<int> devices_;

    int device_;
    bool ilp_;
    size_t staticWorkspaceSize_;

    std::shared_ptr<ConvDatabase> database_;

    static OptCache optCache_;
  };

}

using ucudnn::UcudnnHandle_t;

#endif
