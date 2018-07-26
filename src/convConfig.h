/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_CONV_CONFIG_H_
#define UCUDNN_CONV_CONFIG_H_

#include <vector>
#include <numeric>
#include <string>
#include <memory>

#include <stddef.h>
#include <assert.h>

#include <cudnn.h>

#include "kernelConfig.h"

namespace ucudnn {

  // a set of configurations to execute specific type of a convolution layer
  class ConvConfig {
  public:
    ConvConfig() {}
    ConvConfig(const std::shared_ptr<KernelConfig> kc) {
      kc_ = kc;
      rest_ = std::shared_ptr<ConvConfig>(nullptr);
      time_ = kc->time();
    }
    ConvConfig(const std::shared_ptr<KernelConfig> kc, const std::shared_ptr<ConvConfig> cc) {
      kc_ = kc;
      rest_ = cc;
      time_ = kc->time() + cc->time();
    }

    template <typename AlgoPerf, typename Algo>
    void setPseudoAlgoPerfs(AlgoPerf *perfResults) const {
      perfResults[0].algo = (Algo) 0;
      perfResults[0].status = CUDNN_STATUS_SUCCESS;
      perfResults[0].time = 0.0;
      perfResults[0].memory = memory();
      perfResults[0].determinism = CUDNN_NON_DETERMINISTIC;
    }

    size_t memory() const {
      if(rest_ == nullptr)
	return kc_->memory();
      return std::max(kc_->memory(), rest_->memory());
    }
    float time() const { return time_; }

    std::vector<std::shared_ptr<KernelConfig> > kernelConfigs() const {
      std::vector<std::shared_ptr<KernelConfig> > configs;
      configs.push_back(kc_);
      if(rest_) {
	const auto rests = rest_->kernelConfigs();
	configs.insert(configs.end(), rests.begin(), rests.end());
      }
      return configs;
    }

    std::string toString() const {
      if(rest_ == nullptr)
	return kc_->toString();
      return kc_->toString() + ", " + rest_->toString();
    }

    std::shared_ptr<KernelConfig> kernelConfig() const { return kc_; }
    std::shared_ptr<ConvConfig> rest() const { return rest_; }

  private:
    std::shared_ptr<KernelConfig> kc_;
    std::shared_ptr<ConvConfig> rest_;
    float time_;
  };

}

#endif
