#include <iostream>
#include <cstring>

#include "batch_operations.h"
#include "util.h"

using namespace std;


#ifdef VCUDNN_TEST
#define MEMCPY_FN(a, b, c, d) memcpy(a, b, c);
#else
#define MEMCPY_FN(a, b, c, d) cudaMemcpy(a, b, c, d)
#endif

namespace vcudnn {
void applyBatchMask(
    const Tensor4DDesc & desc,
    cudnnTensorDescriptor_t * cudnn_desc,
    void *tensor,
    const std::vector<bool> &mask,
    std::size_t new_batch_size,
    BatchMaskingDirection direction) {

  // TODO: this only works for NCHW without strides
  // TODO: fix strides
  // TODO: implement for non-NCHW formats

  // apply new mapping
  size_t batch_entry_size = desc.c * desc.h * desc.w;
  rearrange_by_mask(tensor, mask, batch_entry_size, direction);

  // update descriptor
  if(cudnn_desc != nullptr) {
    cudnnSetTensor4dDescriptorEx(
      *cudnn_desc,
      desc.dataType,
      new_batch_size,
      desc.c,
      desc.h,
      desc.w,
      desc.nStride,
      desc.cStride,
      desc.hStride,
      desc.wStride
    );
  }
}

void rearrange_by_mask(
    void * data,
    std::vector<bool> const & mask,
    std::size_t size,
    BatchMaskingDirection direction) {
  int next_empty_idx = 0;
  int last_used_idx = mask.size() - 1;

#ifdef VCUDNN_TEST
  cerr << "*** Warning: compiling with test will use the wrong memcpy function and will not work under CUDA" << endl;
#endif

  while(next_empty_idx < last_used_idx) {
    // find next empty spot
    while(next_empty_idx < mask.size() && mask[next_empty_idx]) {
      ++ next_empty_idx;
    }

    // find next used place item
    while(last_used_idx >= 0 && !mask[last_used_idx]) {
      -- last_used_idx;
    }

    if(next_empty_idx < last_used_idx) {
      // overwrite next_empty_idx with last_used_idx
      size_t dst_idx = direction == BatchMaskForward ? next_empty_idx : last_used_idx;
      size_t src_idx = direction == BatchMaskForward ? last_used_idx : next_empty_idx;

      MEMCPY_FN(
            (uint8_t *)data + size * dst_idx,
            (uint8_t *)data + size * src_idx,
            size,
            cudaMemcpyDeviceToDevice // TODO: this will not always be the case
            );

      ++ next_empty_idx;
      -- last_used_idx;
    }
  }
}

}
