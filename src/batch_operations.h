#ifndef BATCH_OPERATIONS_H
#define BATCH_OPERATIONS_H

#include <cstddef>
#include <cudnn.h>
#include <utility>

namespace vcudnn {
  typedef std::pair<int, int> pint;

  /* Logically shrink the minibatch of input tensors
   * by overwriting lower index, unused values with
   * higher index used values.
   *
   * Returns new minibatch size.
   */
  std::vector<pint> applyBatchMask(cudnnTensorDescriptor_t desc, void* tensor, std::vector<bool> const & mask);
  std::size_t revertBatchMask(cudnnTensorDescriptor_t desc, void* tensor, std::vector<bool> const & mask, std::vector<pint> const & mapping);
}


#endif // BATCH_OPERATIONS_H
