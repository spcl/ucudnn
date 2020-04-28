#include "../extern/catch.hpp"

#include "../src/batch_operations.h"

#include <vector>
#include <algorithm>

using namespace std;
using namespace vcudnn;

TEST_CASE("Rearranging by empty mask it a no-op", "[rearrange]") {
  vector<bool> mask(10, false);
  vector<int> data(10);
  iota(data.begin(), data.end(), 0);
  vector<int> data_copy(data);
  void * pdata = (void*) data.data();

  // precondition
  REQUIRE(data == data_copy);

  // forward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskForward);
  REQUIRE(data == data_copy);

  // backward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskBackward);
  REQUIRE(data == data_copy);
}

TEST_CASE("Rearranging by full mask it a no-op", "[rearrange]") {
  vector<bool> mask(10, true);
  vector<int> data(10);
  iota(data.begin(), data.end(), 0);
  vector<int> data_copy(data);
  void * pdata = (void*) data.data();

  // precondition
  REQUIRE(data == data_copy);

  // forward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskForward);
  REQUIRE(data == data_copy);

  // backward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskBackward);
  REQUIRE(data == data_copy);
}

TEST_CASE("Can rearrange by mask", "[rearrange]") {
  vector<bool> mask =    {1, 1, 0, 0, 1, 0, 1, 0, 1, 1};
  vector<int> data(10);
  iota(data.begin(), data.end(), 0);
  void * pdata = (void*) data.data();
  vector<int> expected = {0, 1, 9, 8, 4, 6, 6, 7, 8, 9};

  // precondition
  REQUIRE(data != expected);

  // forward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskForward);
  REQUIRE(data == expected);

  // backward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskBackward);
  REQUIRE(data == expected);
}

TEST_CASE("Can undo rearrange", "[rearrange]") {
  vector<bool> mask =    {1, 1, 0, 0, 1, 0, 1, 0, 1, 1};
  vector<int> data(10);
  iota(data.begin(), data.end(), 0);
  void * pdata = (void*) data.data();
  vector<int> expected = {0, 1, 9, 8, 4, 6, 6, 7, 8, 9};

  // precondition
  REQUIRE(data != expected);

  // forward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskForward);
  REQUIRE(data == expected);

  // change the active values
  for(int i = 0; i < 6; ++ i) {
    data[i] += 10;
  }
  // mask =  { 1,  1,  0,  0,  1,  0,  1,  0,  1,  1};
  expected = {10, 11, 19, 18, 14, 16, 16,  7, 18, 19};

  // backward application of mask
  rearrange_by_mask(pdata, mask, sizeof(int), BatchMaskBackward);
  REQUIRE(data == expected);
}


