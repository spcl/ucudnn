#include <iostream>

#include "batch_operations.h"
#include "util.h"

using namespace std;

namespace vcudnn {
std::vector<pint> applyBatchMask(cudnnTensorDescriptor_t desc, void *tensor, const std::vector<bool> &mask) {
  std::vector<pint> mapping;

  Tensor4DDesc d;

  if(read_4d_desc(desc, &d)) {
    // TODO: this only works for NCHW without strides
    // TODO: fix strides
    // TODO: implement for non-NCHW formats

    size_t batch_entry_size = d.c * d.h * d.w;

    int next_empty_idx = 0;
    int last_used_idx = mask.size() - 1;

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
        cudaMemcpy(
              (uint8_t *)tensor + batch_entry_size * next_empty_idx,
              (uint8_t *)tensor + batch_entry_size * last_used_idx,
              batch_entry_size,
              cudaMemcpyDeviceToDevice // TODO: this will not always be the case
              );

        mapping.emplace_back(last_used_idx, next_empty_idx);

        ++ next_empty_idx;
        -- last_used_idx;
      }
    }
  } else {
    cerr << "could not read desc to apply mask" << endl;
  }

  return mapping;
}

std::size_t revertBatchMask(cudnnTensorDescriptor_t desc, void *tensor, const std::vector<bool> &mask, const std::vector<pint> &mapping) {
  Tensor4DDesc d;

  if(read_4d_desc(desc, &d)) {
    size_t batch_entry_size = d.c * d.h * d.w;

    for(auto e : mapping) {
      int src_idx = e.first;
      int dst_idx = e.second;

      // overwrite src_idx with dst_idx
      cudaMemcpy(
            (uint8_t *)tensor + batch_entry_size * src_idx,
            (uint8_t *)tensor + batch_entry_size * dst_idx,
            batch_entry_size,
            cudaMemcpyDeviceToDevice // TODO: this will not always be the case
            );
    }
  } else {
    cerr << "could not read desc to revert mask" << endl;
  }

  return 0;
}

}
