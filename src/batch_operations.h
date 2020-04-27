#ifndef BATCH_OPERATIONS_H
#define BATCH_OPERATIONS_H

#include <cstddef>
#include <cudnn.h>

namespace vcudnn {
  /* Logically shrink the minibatch of input tensors
   * by overwriting lower index, unused values with
   * higher index used values.
   *
   * Returns new minibatch size.
   */
  std::size_t applyBatchMask(cudnnTensorDescriptor_t desc, void* tensor, std::vector<bool> const & mask);
  std::size_t revertBatchMask(cudnnTensorDescriptor_t desc, void* tensor, std::vector<bool> const & mask);
}


#endif // BATCH_OPERATIONS_H
