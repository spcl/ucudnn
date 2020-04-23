/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include "optCache.h"
#include "ilpOptimizer.h"

namespace vcudnn {
  void OptCache::setConvConfig(const ConvParam convParam,
			       const ConvType convType,
			       const LayerId layerId,
			       const std::shared_ptr<ConvConfig> convConfig) {
    // std::lock_guard<std::mutex> lock(mutex_);
    LayerId actualLayerId = checkLayerId(layerId, convParam);
    const auto key = std::pair<LayerId, ConvType>(actualLayerId, convType);
    if(convConfig == nullptr && cache_[key] != nullptr) {
      // This will happen if u-cuDNN try to put a dummy config after the optimization.
      // This operation will be ignored.
      return;
    }
    cache_[key] = convConfig;
  }

  std::shared_ptr<ConvConfig> OptCache::getConvConfig(const ConvParam convParam,
						      const ConvType convType,
						      const LayerId layerId) {
    std::lock_guard<std::mutex> lock(mutex_);
    LayerId actualLayerId = checkLayerId(layerId, convParam);
    const auto key = std::pair<LayerId, ConvType>(actualLayerId, convType);
    return cache_[key];
  }

  std::shared_ptr<SafeWorkspace> OptCache::getWDWorkspace(const ConvParam convParam,
							  const ConvType convType,
							  const LayerId layerId,
							  const int deviceId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(workspaces_.size() < deviceId+1)
      workspaces_.resize(deviceId+1);
    assert(layerId != LayerIdAny);
    const auto key = std::pair<LayerId, ConvType>(layerId, convType);

    if(workspaces_[deviceId][key] == nullptr)
      workspaces_[deviceId][key] = std::make_shared<SafeWorkspace>(cache_[key]->memory());
    return workspaces_[deviceId][key];
  }

  void OptCache::optimizeWD(const cudnnHandle_t handle,
			    const size_t totalWorkspaceSize,
			    BatchSizePolicy batchSizePolicy,
			    std::vector<int> &devices) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<OptCacheKey> keys;
    for(const auto k : cache_)
      keys.push_back(k.first);

    if(lastWDSize_ > 0) {
      // This is not the first call, so the last workspaces are destroyed if needed.
      if(keys.size() != lastWDSize_) {
	UCUDNN_WARNING("WD optimization have been performed for several times. This may slow down the cuDNN convolutions due to the optimization overhead.");
	for(const auto wsv : workspaces_)
	  for(const auto ws: wsv)
	    ws.second->destroy();
	workspaces_.resize(0);
      } else {
	// Optimization is not needed because # of layers have not been changed.
	return;
      }
    }

    lastWDSize_ = keys.size();

    std::vector<std::pair<ConvParam, ConvType> > convParams;
    for(const auto k : keys) {
      const ConvParam convParam = layerIdMap.at(k.first);
      convParams.push_back(std::pair<ConvParam, ConvType>(convParam, k.second));
    }
    ILPOptimizer optimizer(handle, convParams);
    auto ret = optimizer.optimize(totalWorkspaceSize, batchSizePolicy, devices);

    assert(keys.size() == ret.size());
    for(auto i = keys.begin(); i != keys.end(); i++) {
      const auto convConfig = ret[std::distance(keys.begin(), i)];
      cache_[*i] = convConfig;
    }

  }

  LayerId OptCache::checkLayerId(const LayerId layerId, const ConvParam convParam) {
    if(layerId == LayerIdAny) {
      // If layerId is not specified, use convPram's hash as its layerId.
      // Therefore layerId is required to perform ILP-based optimization,
      // because u-cuDNN cannot recognize conv. layers that have the same parameters.
      const LayerId randomLayerId = convParam.hash();
      assert(randomLayerId != LayerIdAny);
      return checkLayerId(randomLayerId, convParam);
    }
    const auto i = layerIdMap.find(layerId);
    if(i == layerIdMap.end())
      layerIdMap.emplace(layerId, convParam);
    else
      assert(convParam == (*i).second);
    return layerId;
  }

  std::vector<std::pair<OptCacheKey, ConvParam> > OptCache::getParameters() const {
    std::vector<std::pair<OptCacheKey, ConvParam> > ret;

    for(const auto i : cache_) {
      const LayerId id = (i.first).first;
      ret.push_back(std::pair<OptCacheKey, ConvParam>(i.first, layerIdMap.at(id)));
    }
    return ret;
  }

}
