/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include "desirableSet.h"
#include "convConfig.h"

namespace ucudnn {

  template<typename ConfigType>
  bool DesirableSet<ConfigType>::insert(const std::shared_ptr<ConfigType> config) {
    // Chech whether config with same memory size is already in the set.
    // If it exists, erase one of them which takes longer time than the other
    {
      auto i = set_.find(config);
      if(i != set_.end()) {
	if(config->time() >= (*i)->time())
	  return false;
	else
	  set_.erase(i);
      }
    }

    // Check whether config is not undesirable
    {
      // ... < config->memory() <= (*i_next_t)->memory() < ...
      const auto i_next_t = set_.lower_bound(config);
      if(i_next_t != set_.begin()) {
	// there are one or more elements on the left.
	// ... < (*i_prev_t)->memory() < config->memory() <= (*i_next_t)->memory() < ...
	const auto i_prev_t = std::prev(i_next_t);
	const bool insertable = (config->time() < (*i_prev_t)->time());
	if(!insertable)
	  return false;
      }
    }

    {
      const auto ret = set_.insert(config);
      const auto i_erase_begin = std::next(ret.first);
      auto i_erase_end = set_.end();
      for(auto i = i_erase_begin; i != set_.end(); i++)
	if((*i)->time() < config->time()) {
	  i_erase_end = i;
	  break;
	}
      // ... < config->memory() <= (*i_erase_begin)->memory() < ... < (*i_erase_end)->memory() < ...
      // ... < (*i_erase_begin)->time() < ... < (*i_erase_end)->time() < config->time() < ...
      set_.erase(i_erase_begin, i_erase_end);
    }

    return true;
  }

  template class DesirableSet<ConvConfig>;
  template class DesirableSet<KernelConfig>;

}
