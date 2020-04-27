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

    // TODO: should these operations be thread safe?
    void setBatchSize(std::size_t batch_size);
    void setApplyState(ApplyMaskState new_state);
    ApplyMaskState applyState() const;
    std::size_t applyMask();

  };

  State * get_state();
}

#endif // STATE_H
