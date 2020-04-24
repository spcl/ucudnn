/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef VCUDNN_VCUDNNHANDLE_H_
#define VCUDNN_VCUDNNHANDLE_H_

#include <unordered_map>
#include <cudnn.h>
#include <string>

#include "convParam.h"
#include "optimizer.h"
#include "optCache.h"
#include "util.h"
#include "convDatabase.h"

namespace vcudnn {

  class VcudnnHandle_t {
  public:

    VcudnnHandle_t() { init(); };
    // Some frameworks including NVIDIA Caffe initializes cudnnHandle_t as nullptr,
    // since cudnnHandle_t is alias type for a cudnnContext pointer.
    // So VcudnnHandle_t have a dummy parameter to support such initialization.
    // ptr is totall ignored in the constructor.
    VcudnnHandle_t(const void *ptr) { init(); }
    VcudnnHandle_t(const VcudnnHandle_t &handle);
    ~VcudnnHandle_t();

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

    void log(const std::string message);

    cudnnHandle_t handle_;

  private:
    void init();

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

using vcudnn::VcudnnHandle_t;

#endif
