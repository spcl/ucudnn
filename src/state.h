#ifndef STATE_H
#define STATE_H

#include <vector>

namespace vcudnn {
  enum MaskedParam {
    None = 0,
    Input,    // input parameter
    Output,   // output or input/output parameter
    Both
  };

  class State {
  private:
    std::vector<bool> batch_mask;

    // size of the reduced batch i.e. the active batch
    std::size_t batch_size;

    // parameters to apply the mask to before calling into cudnn
    MaskedParam mask_params;

    // parameters to revert the mask from after calling into cudnn
    MaskedParam unmask_params;

    State();

  public:
    // TODO: should these operations be thread safe?
    void setBatchSize(std::size_t batch_size);

    void setMaskParams(MaskedParam params);
    void setUnmaskParams(MaskedParam params);
    MaskedParam getMaskParams() const;
    MaskedParam getUnmaskParams() const;

    std::vector<bool> const & getMask() const;
    std::size_t removeFromMask(int idx);

    std::size_t getReducedBatchSize() const;
    std::size_t getFullBatchSize() const;

    friend State * getState();
  };

  State * getState();
}

#endif // STATE_H
