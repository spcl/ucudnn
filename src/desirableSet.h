/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_DESIRABLE_SET_H_
#define UCUDNN_DESIRABLE_SET_H_

#include <iostream>
#include <memory>
#include <set>

#include "convConfig.h"

namespace vcudnn {

  template<typename ConfigType>
  struct compare_with_time {
    bool operator() (const std::shared_ptr<ConfigType> &a,
		     const std::shared_ptr<ConfigType> &b) const {
      return a->time() < b->time();
    }
  };

  template<typename ConfigType>
  struct compare_with_memory {
    bool operator() (const std::shared_ptr<ConfigType> &a,
		     const std::shared_ptr<ConfigType> &b) const {
      return a->memory() < b->memory();
    }
  };

  template<typename ConfigType>
  class DesirableSet {
  public:
    bool insert(const std::shared_ptr<ConfigType> config);

    typename std::set<std::shared_ptr<ConfigType>, compare_with_memory<ConfigType> >::iterator
    find(std::shared_ptr<ConfigType> &c) {
      for(auto i = begin(); i != end(); i++)
	if(c == *i)
	  return i;
      return end();
    }
    typename std::set<std::shared_ptr<ConfigType>, compare_with_memory<ConfigType> >::iterator begin() { return set_.begin(); }
    typename std::set<std::shared_ptr<ConfigType>, compare_with_memory<ConfigType> >::iterator end() { return set_.end(); }
    const size_t size() const {return set_.size(); }
    std::vector<std::shared_ptr<ConfigType> > getConfigs() {
      std::vector<std::shared_ptr<ConfigType> > configs;
      for(auto i = begin(); i != end(); i++)
	configs.push_back(*i);
      return configs;
    }

    std::string toString() {
      std::string s = "";
      for(auto config : set_)
	s += config->toString() + ",";
      return s;
    }

  private:
    std::set<std::shared_ptr<ConfigType>, compare_with_memory<ConfigType> > set_;
  };

}

#endif
