#include "state.h"
#include "batch_operations.h"

namespace vcudnn {
  State::State()
    :batch_size(0), should_apply_mask(false) {
  }

  void State::reinit(std::size_t batch_size) {
    mask.resize(batch_size, false);
    // TODO: this is a mock test so that we can trim down the batch size
    for(int idx = 0; idx < batch_size; ++ idx) {
      mask[idx] = i % 2;
    }
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

  const std::vector<bool> &State::getMask() const {
    return this->batch_mask;
  }


  State *get_state() {
    // C++11 standard guarantees that the opject initialization is thread safe
    // https://stackoverflow.com/a/19907903/1218284
    static State state;

    return & state;
  }

}
