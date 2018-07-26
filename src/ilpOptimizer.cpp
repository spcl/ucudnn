/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifdef UCUDNN_USE_GLPK
#include <glpk.h>
#include <sys/time.h>
#endif

#include "optimizer.h"
#include "ilpOptimizer.h"
#include "util.h"

namespace ucudnn {

  std::vector<std::shared_ptr<ConvConfig> > ILPOptimizer::optimize(const size_t workspaceSize,
								   const BatchSizePolicy batchSizePolicy,
								   const std::vector<int> &devices) {
#ifdef UCUDNN_USE_GLPK

#ifdef UCUDNN_DEBUG_OUTPUT
    std::cerr << "=== Convolutional kernels optimization using ILP ===" << std::endl;
    std::cerr << "   Number of kernels: " << kernelList_.size() << std::endl;
    std::cerr << "   Maximum total workspace size: " << workspaceSize << std::endl;
    long us = micros();
#endif

    // list up all of configuration sets
    std::vector<std::vector<std::shared_ptr<ConvConfig> > > configLists;
    std::vector<std::pair<int, int> > variableSubs; // map from variable ID to (layer ID, configuration set ID)
    std::vector<std::shared_ptr<KernelConfig> > bestKernelConfigs; // best single (cuDNN) kernels to compare execution time
    for(auto i = kernelList_.begin(); i != kernelList_.end(); i++) {
      const ConvParam convParam = (*i).first;
      const ConvType convType = (*i).second;
      Optimizer optimizer(handle_, convParam);
      const int idx = configLists.size();
      const auto ret = optimizer.getDesirableConvConfigs(convType,
							 convParam.getDefaultBatchSize(),
							 workspaceSize,
							 batchSizePolicy,
							 false,
							 devices);
      configLists.push_back(ret.first);
      bestKernelConfigs.push_back(ret.second);
      for(int j = 0; j < configLists[idx].size(); j++)
	variableSubs.push_back(std::make_pair(idx, j));
    }

#ifdef UCUDNN_DEBUG_OUTPUT
    std::cerr << "Time to benchmark and optimize all kernels: " << (micros()-us) << "[us]" << std::endl;
#endif

    const int kernelCount = kernelList_.size();
    const int variableCount = variableSubs.size();

    // create a GLPK problem object
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_prob_name(lp, "u-cuDNN ILP");
    glp_set_obj_dir(lp, GLP_MIN);

    // set up LP rows
    glp_add_rows(lp, kernelCount+1);
    for(int row = 0; row < kernelCount; row++) {
      const std::string name = "kernel " + std::to_string(row);
      glp_set_row_name(lp, row+1, name.c_str());
      glp_set_row_bnds(lp, row+1, GLP_FX, 1.0, 1.0);
    }
    glp_set_row_name(lp, kernelCount+1, "Total workspace size");
    glp_set_row_bnds(lp, kernelCount+1, GLP_DB, 0.0, workspaceSize);

    // set up LP columns
    glp_add_cols(lp, variableCount);
    for(int col = 0; col < variableCount; col++) {
      auto sub = variableSubs[col];
      const int kernel = sub.first;
      const int config = sub.second;
      const float time = configLists[kernel][config]->time();
      const std::string name = "x_k" + std::to_string(kernel)
	+ ",c" + std::to_string(config);
      glp_set_col_name(lp, col+1, name.c_str());
      glp_set_col_kind(lp, col+1, GLP_BV);
      glp_set_col_bnds(lp, col+1, GLP_DB, 0.0, 1.0);
      glp_set_obj_coef(lp, col+1, time);
    }

    // set up LP matrix
    {
      std::vector<int> rowIndices;
      std::vector<int> colIndices;
      std::vector<double> coeffs;
      rowIndices.push_back(0);
      colIndices.push_back(0);
      coeffs.push_back(0);

      for(int i = 0; i < variableCount; i++) {
	auto sub = variableSubs[i];
	const int kernel = sub.first;
	const int config = sub.second;
	const size_t memory = configLists[kernel][config]->memory();

	rowIndices.push_back(kernel+1);
	colIndices.push_back(i+1);
	coeffs.push_back(1.0);

	rowIndices.push_back(kernelCount+1);
	colIndices.push_back(i+1);
	coeffs.push_back(memory);

	glp_load_matrix(lp, coeffs.size()-1,
			&rowIndices[0],
			&colIndices[0],
			&coeffs[0]);
      }
    }

#ifdef UCUDNN_DEBUG_OUTPUT
    std::cerr << "--- GLPK ---" << std::endl;
    us = micros();
#endif

    // solve ILP
    glp_simplex(lp, nullptr);
    glp_intopt(lp, nullptr);

#ifdef UCUDNN_DEBUG_OUTPUT
    std::cerr << "--- GLPK end ---" << std::endl << std::endl;
    std::cerr << "Time to solve ILP: " << (micros()-us) << "[us]" << std::endl;
#endif

#ifdef UCUDNN_DEBUG_GLPK_PRINT
    const std::string t = std::to_string(time(nullptr));
    glp_print_sol(lp, (std::string("glpk_sol_")+t).c_str());
    glp_print_mip(lp, (std::string("glpk_mip_")+t).c_str());
    glp_write_prob(lp, 0, (std::string("glpk_prob_")+t).c_str());
#endif

    // we should get optimal solution here; the worst case is using algorithm ID 0 (which doesn't require WS) for all layers
    assert(glp_mip_status(lp) == GLP_OPT);

    std::vector<int> usedVariables;
    for(int col = 0; col < variableCount; col++) {
      if(glp_mip_col_val(lp, col+1) == 1.0)
	usedVariables.push_back(col);
    }
    assert(usedVariables.size() == kernelCount);

    const float totalTime = glp_mip_obj_val(lp);
    const size_t usedWorkspaceSize = glp_mip_row_val(lp, kernelCount+1);
    assert(usedWorkspaceSize <= workspaceSize);

    std::vector<std::shared_ptr<ConvConfig> > ret;
    for(int i = 0; i < kernelCount; i++) {
      const int configId = variableSubs[usedVariables[i]].second;
      ret.push_back(configLists[i][configId]);
    }

#ifdef UCUDNN_DEBUG_OUTPUT
    assert(bestKernelConfigs.size() > 0);
    const float time_singleKernels = std::accumulate(bestKernelConfigs.begin(), bestKernelConfigs.end(),
						     0.0,
						     [](float t, std::shared_ptr<KernelConfig> c) {
						       return t + c->time();
						     });
    const size_t memory_singleKernels = std::accumulate(bestKernelConfigs.begin(), bestKernelConfigs.end(),
							0,
							[](size_t m, std::shared_ptr<KernelConfig> c) {
							  return m + c->memory();
							});
    std::cerr << "Expected total execution time: " << totalTime << "[ms]" << std::endl;
    std::cerr << "Expected speedup: " << (time_singleKernels / totalTime)
	      << " (" << time_singleKernels << "[ms] -> " << totalTime << "[ms])" << std::endl;
    std::cerr << "Total workspace size: " << usedWorkspaceSize << "[bytes]" << std::endl;
    std::cerr << "Workspace size ratio: " << ((float) usedWorkspaceSize / memory_singleKernels)
	      << " (" << memory_singleKernels << "[bytes] -> " << usedWorkspaceSize << "[bytes])"
	      << std::endl;
    std::cerr << "Workspace utilization: " << ((float) usedWorkspaceSize / workspaceSize)
	      << "(" << usedWorkspaceSize << "[bytes] / " << workspaceSize << "[bytes])" << std::endl;

    for(auto i = kernelList_.begin(); i != kernelList_.end(); i++) {
      const int idx = std::distance(kernelList_.begin(), i);
      std::cerr << "---" << std::endl;
      const auto config = ret[idx];
      std::cerr <<  "Best cuDNN kernel(s) for (" << (*i).first.toString()
		<< ", " << convTypeToString((*i).second) <<"): " << std::endl;
      std::cerr << "   " << bestKernelConfigs[idx]->toString() << std::endl;
      std::cerr <<  "Selected u-cuDNN kernel(s): " << std::endl;
      for(const auto kc : config->kernelConfigs())
	std::cerr << "   " << kc->toString() << std::endl;
      std::cerr << "Workspace size: " << config->memory() << "[bytes]" << std::endl;
    }
    std::cerr << "=== ILP optimization end ===" << std::endl << std::endl;
#endif

    return ret;
#else
    UCUDNN_ERROR_EXIT("Failed to run ILP-based optimization. Reinstall u-cuDNN with CUDNN_USE_GLPK option.");
#endif
  }

}
