#ifndef STATE_H
#define STATE_H

#include <vector>

namespace vcudnn {
  enum ApplyMaskState {
    None = 0,
    Input,
    Output,
    Both
  };

  class State {
  private:
    std::vector<bool> batch_mask;
    std::size_t batch_size;
    ApplyMaskState apply_state;

    State();

  public:
    void reinit(std::size_t batch_size);

    // TODO: should these operations be thread safe?
    void setBatchSize(std::size_t batch_size);
    void setApplyState(ApplyMaskState new_state);
    ApplyMaskState applyState() const;
    std::vector<bool> const & getMask() const;
    std::size_t removeFromMask(int idx);
    std::size_t getBatchSize() const;
    std::size_t getOldBatchSize() const;

    friend State * get_state();
  };

  State * get_state();
}

#endif // STATE_H
