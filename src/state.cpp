#include "state.h"
#include "batch_operations.h"

namespace vcudnn {
  State::State()
    :batch_size(0), should_apply_mask(false) {
  }

  void State::setBatchSize(std::size_t batch_size) {
    this->batch_size = batch_size;
  }

  void State::setApplyState(ApplyMaskState new_state) {
    this->apply_state = new_state;
  }

  ApplyMaskState State::applyState() const {
    return this->apply_state;
  }

  std::size_t State::applyMask() {
    if(should_apply_mask) {
      size_t new_batch_size = applyMask();
      batch_size = new_batch_size;
    }

    return batch_size;
  }

  State *get_state() {
    // C++11 standard guarantees that the opject initialization is thread safe
    // https://stackoverflow.com/a/19907903/1218284
    static State state;

    return & state;
  }

}
