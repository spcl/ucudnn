/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include "convConfig.h"
#include "desirableSet.h"

#include <stdlib.h>
#include <assert.h>

int main(void) {
  const size_t maxMemory = 64UL*1024*1024;
  const int count = 10000;
  const long seed = 0;

  // generate random configurations with
  //  - memory follows U(0, maxMemory)
  //  - time follows U(1, 2) - (memory/maxMemory)
  std::vector<std::shared_ptr<ucudnn::KernelConfig> > configs;
  srand(seed);
  srand48(seed);
  for(int i = 0; i < count; i++) {
    const size_t memory = rand()%maxMemory;
    const float time = 1.0 - ((float) memory / maxMemory) + drand48();
    configs.push_back(std::make_shared<ucudnn::KernelConfig>(0, 256, memory, time, CUDNN_DATA_FLOAT,
							     CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NCHW
#if CUDNN_HAS_MATHTYPE
							     , CUDNN_DEFAULT_MATH
#endif
							     ));
  }

  // inset configs to DesirableSet
  ucudnn::DesirableSet<ucudnn::KernelConfig> set;
  for(auto config : configs)
    set.insert(config);

  std::cerr << "Contents of the desirable set are:" << std::endl;
  for(auto i = set.begin(); i != set.end(); i++)
    std::cerr << "   " << (*i)->toString() << std::endl;
  std::cout << "Set size: " << set.size() << std::endl;

  // validation
  std::cout << "Start testing..." << std::endl;
  for(auto config : configs) {
    bool isDesirable = true;
    for(auto c2 : set)
      if(config->time() > c2->time() && config->memory() >= c2->memory()) {
	isDesirable = false;
	break;
      }
    const bool isInSet = (set.find(config) != set.end());
    if(isInSet != isDesirable) {
      std::cerr << "Test failed." << std::endl;
      std::cerr << config->toString() << " should " << (isDesirable ? "" : "not") << " be in the desiable set." << std::endl;
      exit(1);
    }
  }

  std::cout << "Test passed." << std::endl;
}
