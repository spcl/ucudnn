#include "state.h"
#include "batch_operations.h"

namespace vcudnn {
  State::State()
    :batch_size(0) {
  }

  // TODO: this function should not exist
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
    this->batch_mask.resize(batch_size, true);
  }

  void State::setMaskParams(MaskedParam params) {
    this->mask_params = params;
  }

  void State::setUnmaskParams(MaskedParam params) {
    this->unmask_params = params;
  }

  MaskedParam State::getMaskParams() const {
    return this->mask_params;
  }

  MaskedParam State::getUnmaskParams() const {
    return this->unmask_params;
  }

  const std::vector<bool> & State::getMask() const {
    return this->batch_mask;
  }

  std::size_t State::removeFromMask(int idx) {
    this->batch_size -= this->batch_mask.at(idx);
    this->batch_mask[idx] = false;
    return this->batch_size;
  }

  std::size_t State::getReducedBatchSize() const {
    return this->batch_size;
  }

  std::size_t State::getFullBatchSize() const {
    return this->batch_mask.size();
  }

  State *getState() {
    // C++11 standard guarantees that the opject initialization is thread safe
    // https://stackoverflow.com/a/19907903/1218284
    static State state;

    return & state;
  }

}
