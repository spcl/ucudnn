/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_KERNEL_CONFIG_H_
#define UCUDNN_KERNEL_CONFIG_H_

#include <vector>
#include <numeric>
#include <string>
#include <memory>

#include <stddef.h>
#include <assert.h>

#include <cudnn.h>

#include "util.h"

namespace vcudnn {

  std::string convTypeToString(const ConvType convType);

  // a configuration to execute a cuDNN convolution kernel
  // which contains (micro-)batch size and exact workspace size
  class KernelConfig {
  public:
    KernelConfig(const int algo, const int batchSize, const size_t memory, const float time, const cudnnDataType_t computeType,
		 const cudnnTensorFormat_t xFormat, const cudnnTensorFormat_t yFormat, const cudnnTensorFormat_t wFormat
#if CUDNN_HAS_MATHTYPE
		 , const cudnnMathType_t mathType
#endif
		 )
      : algo_(algo), batchSize_(batchSize), memory_(memory), time_(time), computeType_(computeType),
	xFormat_(xFormat), yFormat_(yFormat), wFormat_(wFormat)
#if CUDNN_HAS_MATHTYPE
      , mathType_(mathType)
#endif
    {}
    template <typename AlgoPerf>
    KernelConfig(const AlgoPerf perf, const int batchSize, const cudnnDataType_t computeType,
		 const cudnnTensorFormat_t xFormat, const cudnnTensorFormat_t yFormat, const cudnnTensorFormat_t wFormat)
      : algo_(perf.algo), batchSize_(batchSize), memory_(perf.memory), time_(perf.time), computeType_(computeType),
	xFormat_(xFormat), yFormat_(yFormat), wFormat_(wFormat)
#if CUDNN_HAS_MATHTYPE
      , mathType_(perf.mathType)
#endif
    {}
    KernelConfig(const KernelConfig &c)
      : algo_(c.algo_), batchSize_(c.batchSize_), memory_(c.memory_), time_(c.time_), computeType_(c.computeType_),
	xFormat_(c.xFormat_), yFormat_(c.yFormat_), wFormat_(c.wFormat_)
#if CUDNN_HAS_MATHTYPE
      , mathType_(c.mathType_)
#endif
    {}

    // this is only for creating a placeholder.
    KernelConfig(const int batchSize)
      : algo_(0), batchSize_(batchSize), memory_(0), time_(0), computeType_(CUDNN_DATA_FLOAT),
	xFormat_(CUDNN_TENSOR_NCHW), yFormat_(CUDNN_TENSOR_NCHW), wFormat_(CUDNN_TENSOR_NCHW)
#if CUDNN_HAS_MATHTYPE
      , mathType_(CUDNN_DEFAULT_MATH)
#endif
    {}

    int algo() const { return algo_; }
    int batchSize() const { return batchSize_; }
    size_t memory() const { return memory_; }
    float time() const { return time_; }
    cudnnDataType_t computeType() const { return computeType_; }
    cudnnTensorFormat_t xFormat() const { return xFormat_; }
    cudnnTensorFormat_t yFormat() const { return yFormat_; }
    cudnnTensorFormat_t wFormat() const { return wFormat_; }
#if CUDNN_HAS_MATHTYPE
    cudnnMathType_t mathType() const { return mathType_; }
#endif

    std::string toString() const {
      std::string s = "{";
      s += "algo: " + std::to_string(algo_);
      s += ", micro-batch: " + std::to_string(batchSize_);
      s += ", workspace: " + std::to_string(memory_);
      s += ", time: " + std::to_string(time_);
      s += ", computeType: " + dataTypeToString(computeType_);
      s += ", xFormat: " + getTensorFormatName(xFormat_);
      s += ", yFormat: " + getTensorFormatName(yFormat_);
      s += ", wFormat: " + getTensorFormatName(wFormat_);
#if CUDNN_HAS_MATHTYPE
      s += ", mathType: " + mathTypeToString(mathType_);
#endif
      s += "}";
      return s;
    }

  private:
    const int algo_;
    const int batchSize_;
    const size_t memory_;
    const float time_;
    const cudnnDataType_t computeType_;
    const cudnnTensorFormat_t xFormat_, yFormat_, wFormat_;
#if CUDNN_HAS_MATHTYPE
    const cudnnMathType_t mathType_;
#endif
  };

}

#endif
