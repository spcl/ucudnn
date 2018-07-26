/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include "safeWorkspace.h"
#include "util.h"

namespace ucudnn {

  SafeWorkspace::SafeWorkspace(const size_t size) {
    UCUDNN_CUDA_CHECK(cudaMalloc(&ptr_, size));
    UCUDNN_CUDA_CHECK(cudaMemset(ptr_, 0, size));
    UCUDNN_CUDA_CHECK(cudaEventCreate(&event_));
    used_ = false;
    size_ = size;
  }

  void SafeWorkspace::destroy() {
    if(used_) {
      UCUDNN_CUDA_CHECK(cudaEventSynchronize(event_));
      UCUDNN_CUDA_CHECK(cudaEventDestroy(event_));
    }
    UCUDNN_CUDA_CHECK(cudaFree(ptr_));
  }

  bool SafeWorkspace::isAvailable(const cudaStream_t stream) const {
    // return true if workspace is not used yet
    if(!used_)
      return true;

    // return true if the next kernel will be enqueued to the same stream
    if(stream == stream_)
      return true;

    // return true if the previous kernel has already been completed
    const cudaError_t ret = cudaEventQuery(event_);
    if(ret == cudaSuccess)
      return true;
    if(ret == cudaErrorNotReady)
      return false;
    UCUDNN_CUDA_CHECK(ret);
    return false;
  }

  void SafeWorkspace::setPostKernelRecord(const cudaStream_t stream) {
    used_ = true;
    UCUDNN_CUDA_CHECK(cudaEventRecord(event_, stream));
    stream_ = stream;
  }

}

