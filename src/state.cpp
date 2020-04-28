#include "state.h"
#include "batch_operations.h"

namespace vcudnn {
  State::State()
    :batch_size(0) {
  }

  void State::reinit(std::size_t batch_size) {
    batch_mask.resize(batch_size, false);

    // TODO: this is a mock test so that we can trim down the batch size
    int bs = 0;
    for(int idx = 0; idx < batch_size; ++ idx) {
      batch_mask[idx] = idx % 2;
      bs += batch_mask[idx];
    }

    this->batch_size = bs;
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

  const std::vector<bool> & State::getMask() const {
    return this->batch_mask;
  }

  std::size_t State::removeFromMask(int idx) {
    this->batch_size -= this->batch_mask[idx];
    this->batch_mask[idx] = false;
    return this->batch_size;
  }

  std::size_t State::getBatchSize() const {
    return this->batch_size;
  }

  std::size_t State::getOldBatchSize() const {
    return this->batch_mask.size();
  }

  State *get_state() {
    // C++11 standard guarantees that the opject initialization is thread safe
    // https://stackoverflow.com/a/19907903/1218284
    static State state;

    return & state;
  }

}
