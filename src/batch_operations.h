#ifndef BATCH_OPERATIONS_H
#define BATCH_OPERATIONS_H

#include <cstddef>
#include <cudnn.h>
#include <utility>
#include <vector>

#include "util.h"

namespace vcudnn {
  typedef std::pair<int, int> pint;

  enum BatchMaskingDirection : uint8_t {
    BatchMaskForward = 0,
    BatchMaskBackward
  };

  /* Logically shrink the minibatch of input tensors
   * by overwriting lower index, unused values with
   * higher index used values.
   *
   * Returns new minibatch size.
   */
  void applyBatchMask(
      const Tensor4DDesc & desc,
      cudnnTensorDescriptor_t * cudnn_desc,
      void* tensor,
      std::vector<bool> const & mask,
      std::size_t new_batch_size,
      BatchMaskingDirection direction = BatchMaskForward
      );

  // internal functions
  void rearrange_by_mask(
      void * data,
      std::vector<bool> const & mask,
      std::size_t size,
      BatchMaskingDirection direction);
}


#endif // BATCH_OPERATIONS_H
