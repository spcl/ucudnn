/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_OPT_CACHE_H_
#define UCUDNN_OPT_CACHE_H_

#include <map>
#include <unordered_map>
#include <mutex>

#include "util.h"
#include "convParam.h"
#include "safeWorkspace.h"

namespace ucudnn {
  typedef std::pair<LayerId, ConvType> OptCacheKey;
}
namespace std {
  template <>
  struct hash<ucudnn::OptCacheKey> {
  public:
    size_t operator()(const ucudnn::OptCacheKey &key) const {
      return (size_t) key.first + (size_t) key.second;
    }
  };
};

namespace ucudnn {

  class OptCache {
  public:
    OptCache() { lastWDSize_ = 0; }
    OptCache(const OptCache &optCache) { assert(false); }

    void setConvConfig(const ConvParam convParam,
		       const ConvType convType,
		       const LayerId layerId,
		       const std::shared_ptr<ConvConfig> convConfig=nullptr);
    std::shared_ptr<ConvConfig> getConvConfig(const ConvParam convParam,
					      const ConvType convType,
					      const LayerId layerId);
    std::shared_ptr<SafeWorkspace> getWDWorkspace(const ConvParam convParam,
						  const ConvType convType,
						  const LayerId layerId,
						  const int deviceId);

    void optimizeWD(const cudnnHandle_t handle,
		    const size_t totalWorkspaceSize,
		    BatchSizePolicy batchSizePolicy,
		    std::vector<int> &devices);

    std::vector<std::pair<OptCacheKey, ConvParam> > getParameters() const;

  private:
    LayerId checkLayerId(const LayerId layerId, const ConvParam convParam);

    std::mutex mutex_;
    std::map<OptCacheKey, std::shared_ptr<ConvConfig> > cache_;
    std::unordered_map<LayerId, ConvParam> layerIdMap;
    int lastWDSize_;

    std::vector<std::map<OptCacheKey, std::shared_ptr<SafeWorkspace> > > workspaces_;
  };

}

#endif
