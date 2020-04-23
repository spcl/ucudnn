/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include "kernelConfig.h"

namespace vcudnn {

  std::string convTypeToString(const ConvType convType) {
    switch(convType) {
    case Forward:
      return "Forward";
    case BackwardData:
      return "BackwardData";
    case BackwardFilter:
      return "BackwardFilter";
    }
  }

}
