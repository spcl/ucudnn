/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_SAFE_WORKSPACE_H_
#define UCUDNN_SAFE_WORKSPACE_H_

#include <vector>
#include <tuple>
#include <stddef.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace vcudnn {

  class SafeWorkspace {
  public:
    SafeWorkspace(const size_t size);
    void destroy();
    bool isAvailable(const cudaStream_t stream) const;
    void setPostKernelRecord(const cudaStream_t stream);
    void *ptr() const { return ptr_; }

  private:
    void * ptr_;
    cudaStream_t stream_;
    cudaEvent_t event_;
    bool used_;
    size_t size_;
  };

}

#endif
