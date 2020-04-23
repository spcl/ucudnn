/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <set>
#include <numeric>
#include <memory>
#include <algorithm>
#include <thread>
#include <assert.h>

#include "util.h"

#include "convConfig.h"
#include "convParam.h"
#include "optimizer.h"
#include "desirableSet.h"

namespace vcudnn {

  /// --- public static ---

  std::string Optimizer::batchSizePolicyToString(const BatchSizePolicy batchSizePolicy) {
    switch(batchSizePolicy) {
    case BatchSizePolicy::all:
      return "all";
    case BatchSizePolicy::powerOfTwo:
      return "powerOfTwo";
    case BatchSizePolicy::undivided:
      return "undivided";
    default:
      assert(false);
    }
  }

  BatchSizePolicy Optimizer::stringToBatchSizePolicy(const std::string str) {
    for(int i = 0; i < BatchSizePolicySize; i++)
      if(str == batchSizePolicyToString((BatchSizePolicy) i))
	return (BatchSizePolicy) i;
    std::cerr << "Invalid batch size policy: " << str << std::endl;
    assert(false);
  }

  void Optimizer::batchSizePolicyToBatchSizes(const BatchSizePolicy batchSizePolicy,
					      const int batchSize,
					      std::vector<int> &batchSizes) {
    switch(batchSizePolicy) {
    case all:
      for(int b = 1; b <= batchSize; b++)
	batchSizes.push_back(b);
      break;

    case powerOfTwo:
      for(int b = 1; b <= batchSize; b *= 2)
	batchSizes.push_back(b);
      if(std::find(batchSizes.begin(), batchSizes.end(), batchSize) == batchSizes.end())
	batchSizes.push_back(batchSize);
      break;

    case undivided:
      batchSizes.push_back(batchSize);
      break;

    default:
      assert(false);
    }
  }

  // -- public --

  Optimizer::Optimizer(const cudnnHandle_t handle, const ConvParam convParam, std::shared_ptr<ConvDatabase> convDatabase)
    : handle_(handle), convParam_(convParam), convDatabase_(convDatabase) {
    // TODO: don't reuse descriptors to prevent bugs
    UCUDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc_));
    UCUDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc_));
    UCUDNN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc_));
    UCUDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc_));

    convParam_.setWDesc(wDesc_);
  }

  Optimizer::~Optimizer() {
    UCUDNN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc_));
    UCUDNN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc_));
    UCUDNN_CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc_));
    UCUDNN_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc_));
  }

  std::vector<std::shared_ptr<KernelConfig> >
  Optimizer::getDesirableKernelConfigs(const ConvType convType,
				       const int batchSize,
				       const size_t workspaceSize,
				       FindConvAlgorithmExPointers *ptrs,
				       const bool printResults) {
    const cudnnDataType_t defaultComputeType = convParam_.getComputeType();
    auto ret = getDesirableKernelConfigs(defaultComputeType, convType, batchSize,
					 workspaceSize, ptrs, printResults);
    if(defaultComputeType == CUDNN_DATA_HALF) {
      auto ret_pseudoHalf = getDesirableKernelConfigs(CUDNN_DATA_FLOAT, convType, batchSize,
						      workspaceSize, ptrs, printResults);
      ret.insert(ret.end(), ret_pseudoHalf.begin(), ret_pseudoHalf.end());
    }
    return ret;
  }

  std::vector<std::shared_ptr<KernelConfig> >
  Optimizer::getDesirableKernelConfigs(const cudnnDataType_t computeType,
				       const ConvType convType,
				       const int batchSize,
				       const size_t workspaceSize,
				       FindConvAlgorithmExPointers *ptrs,
				       const bool printResults) {

    ConvParam convParamWithCT = convParam_;
    convParamWithCT.setComputeType(computeType);

    convParamWithCT.setXDesc(xDesc_, batchSize);
    convParamWithCT.setYDesc(yDesc_, batchSize);
    convParamWithCT.setConvDesc(convDesc_);

    const int maximumAlgoCount = 0xFF;
    int algoCount;

    DesirableSet<KernelConfig> candidates;

    std::vector<cudnnConvolutionGenericAlgoPerf_t> availablePerfs;

    // if the benckmarking results exist in the database, just uses them
    if(convDatabase_) {
      const auto ret = convDatabase_->selectPerfResults(convParamWithCT, convType, batchSize, workspaceSize);
      if(ret.size() > 0) {
	std::cerr << "Using existing " << std::to_string(ret.size()) << " result(s) in the database: "
		  << convParamWithCT.toString()
		  << ", " << convTypeToString(convType)
		  << ", " << std::to_string(batchSize) << std::endl;
	availablePerfs.insert(availablePerfs.end(), ret.begin(), ret.end());
      }
    }

    // if not available, performs actual benchmarks
    if(availablePerfs.size() == 0) {
      cudnnConvolutionGenericAlgoPerf_t perfs[maximumAlgoCount];

      if(ptrs && ptrs->x && ptrs->y && ptrs->w && ptrs->workspace) {
	// Benchmarking using already allocated pointers
	UCUDNN_CUDNN_CHECK(cudnnFindConvolutionGenericAlgorithmEx(handle_,
								  xDesc_, yDesc_, wDesc_, convDesc_,
								  ptrs->x, ptrs->y, ptrs->w, ptrs->workspace,
								  workspaceSize,
								  maximumAlgoCount,
								  &algoCount,
								  perfs,
								  convType));
      } else {
	// Benchmarking without pointers
	UCUDNN_CUDNN_CHECK(cudnnFindConvolutionGenericAlgorithm(handle_,
								xDesc_, yDesc_, wDesc_, convDesc_,
								maximumAlgoCount,
								&algoCount,
								perfs,
								convType));
      }

      // pack available algorithms
      for(int i = 0; i < algoCount; i++) {
	if(perfs[i].status == CUDNN_STATUS_SUCCESS && perfs[i].memory <= workspaceSize)
	  availablePerfs.push_back(perfs[i]);
      }

      // if the database is available, puts the results
      if(convDatabase_)
	convDatabase_->insertPerfResults(convParamWithCT, convType, batchSize, workspaceSize,
					 availablePerfs);
    }

    for(auto i = availablePerfs.begin(); i != availablePerfs.end(); i++) {
      const auto kc = std::shared_ptr<KernelConfig>(new KernelConfig((*i), batchSize, computeType,
								     convParam_.getXFormat(),
								     convParam_.getYFormat(),
								     convParam_.getWFormat()));
      candidates.insert(kc);
#ifdef UCUDNN_DEBUG_OUTPUT
      if(printResults)
	std::cerr << "Available cuDNN kernel: " << kc->toString() << std::endl;
#endif
    }

    return candidates.getConfigs();
  }

  std::shared_ptr<ConvConfig> Optimizer::getBestConvConfig(const ConvType convType,
							   const int batchSize,
							   const size_t workspaceSize,
							   const BatchSizePolicy batchSizePolicy,
							   const std::vector<int> &devices,
							   FindConvAlgorithmExPointers *ptrs) {

    const auto ret = getDesirableConvConfigs(convType, batchSize, workspaceSize,
					     batchSizePolicy, true, devices, ptrs).first;
    assert(ret.size() == 1);
    return ret[0];
  }

  std::pair<std::vector<std::shared_ptr<ConvConfig> >, std::shared_ptr<KernelConfig> >
  Optimizer::getDesirableConvConfigs(const ConvType convType,
				     const int batchSize,
				     const size_t workspaceSize,
				     const BatchSizePolicy batchSizePolicy,
				     const bool best,
				     const std::vector<int> &devices,
				     FindConvAlgorithmExPointers *ptrs) {
#ifdef UCUDNN_DEBUG_OUTPUT
    std::cerr << "--- Convolutional layer optimization using DP (" << convTypeToString(convType)
	      << ", " << convParam_.toString() << ") ---" << std::endl;
    std::cerr << "Batch size policy: " << batchSizePolicyToString(batchSizePolicy) << std::endl;
    std::cerr << "Maximum workspace size: " << workspaceSize << std::endl;
    long us = micros();
#endif

    std::vector<std::vector<std::shared_ptr<KernelConfig> > > kernelConfigBench =
      getDesirableKernelConfigBench(convType, batchSize, workspaceSize, batchSizePolicy, devices, ptrs, best);

#ifdef UCUDNN_DEBUG_OUTPUT
    const long benchmarkTime = micros()-us;
    us = micros();
#endif

    std::vector<std::vector<std::shared_ptr<ConvConfig> > > dpTable;
    dpTable.resize(batchSize+1);

    for(int b = 1; b <= batchSize; b++) {
      DesirableSet<ConvConfig> candidates;

      for(auto kernelConfig : kernelConfigBench[b])
	candidates.insert(std::shared_ptr<ConvConfig>(new ConvConfig(kernelConfig)));

      // In any set of multiple configs, it should contain one or more config in which
      // micro-batch size is less than b/2. Hence we don't have to test bSingle >= b/2
      for(int bSingle = 1; bSingle <= b/2; bSingle++) {
	const int bRest = b-bSingle;
	if(bRest <= 0 || bRest >= b)
	  continue;
	for(auto kernelConfig : kernelConfigBench[bSingle])
	  for(auto restConfig : dpTable[bRest])
	    candidates.insert(std::shared_ptr<ConvConfig>(new ConvConfig(kernelConfig, restConfig)));
      }

      if(best) {
	if(candidates.size() > 0)
	  dpTable[b].push_back(std::accumulate(candidates.begin(), candidates.end(),
					       *candidates.begin(),
					       [](std::shared_ptr<ConvConfig> c1,
						  std::shared_ptr<ConvConfig> c2) {
						 return (c1->time() < c2->time()) ? c1 : c2;
					       }));
      } else
	dpTable[b].insert(dpTable[b].end(), candidates.begin(), candidates.end());
    }

    assert(kernelConfigBench[batchSize].size() >= 1);
    assert(dpTable[batchSize].size() >= 1);

    const auto defaultComputeType = convParam_.getComputeType();
    std::vector<std::shared_ptr<KernelConfig> > kernelsWithDefaultCT;
    for(auto i : kernelConfigBench[batchSize])
      if(i->computeType() == defaultComputeType)
	kernelsWithDefaultCT.push_back(i);

    // This happen if the user wants to use TRUE_HALF but it is not available, for example on Kepler GPUs.
    if(kernelsWithDefaultCT.size() == 0) {
      UCUDNN_WARNING("No algorithm with default computeType is found.");
      kernelsWithDefaultCT.insert(kernelsWithDefaultCT.end(),
				  kernelConfigBench[batchSize].begin(), kernelConfigBench[batchSize].end());
    }

    const auto best_kernel = std::accumulate(kernelsWithDefaultCT.begin(), kernelsWithDefaultCT.end(),
					     *kernelsWithDefaultCT.begin(),
					     [](std::shared_ptr<KernelConfig> c1,
						std::shared_ptr<KernelConfig> c2) {
					       return (c1->time() < c2->time()) ? c1 : c2;
					     });

#ifdef UCUDNN_DEBUG_OUTPUT
    const float time_kernel = best_kernel->time();
    const size_t memory_kernel = best_kernel->memory();
    std::cerr << "Best cuDNN kernel: " << best_kernel->toString() << std::endl;

    std::cerr << "Time to benchmark all existing kernels: " << benchmarkTime << "[us]" << std::endl;
    std::cerr << "Time to optimize combination of kernels using DP: " << (micros()-us) << "[us]" << std::endl;
    std::cerr << "Number of optimized configuration sets: " << dpTable[batchSize].size() << std::endl;

    for(const auto best_dp : dpTable[batchSize]) {
      std::cerr << "---" << std::endl;

      const float time_dp = best_dp->time();
      const size_t memory_dp = best_dp->memory();
      auto kernels = best_dp->kernelConfigs();
      for(auto i = kernels.begin(); i != kernels.end(); i++)
	std::cerr << (i == kernels.begin()
		      ? std::string(best ? "Best" : "Desirable") + " u-cuDNN kernel(s): "
		      : "                        ")
		  << (*i)->toString() << std::endl;

      const float speedup = time_kernel / time_dp;
      const float memoryRatio = (float) memory_dp / memory_kernel;

      std::cerr << "Expected speedup: " << speedup
		<< " (" << time_kernel << "[ms] -> " << time_dp << "[ms])"
		<< std::endl;
      std::cerr << "Workspace size ratio: " << memoryRatio
		<< " (" << memory_kernel << "[bytes] -> " << memory_dp << "[bytes])"
		<< std::endl;
      std::cerr << "Workspace utilization: " << ((float) memory_dp / workspaceSize)
		<< " (" << memory_dp << "[bytes] / " << workspaceSize << "[bytes])"
		<< std::endl;

      assert(speedup >= 1 || (memoryRatio < 1 || (memory_kernel == 0 && memory_dp == 0)));
    }
    std::cerr << "--- DP optimization end ---" << std::endl << std::endl;
#endif

    return std::pair<std::vector<std::shared_ptr<ConvConfig> >, std::shared_ptr<KernelConfig> >(dpTable[batchSize], best_kernel);
  }

  // --- private ---

  std::vector<std::vector<std::shared_ptr<KernelConfig> > > Optimizer::getDesirableKernelConfigBench(const ConvType convType,
												     const int batchSize,
												     const size_t workspaceSize,
												     const BatchSizePolicy batchSizePolicy,
												     const std::vector<int> &devices,
												     FindConvAlgorithmExPointers *ptrs,
												     const bool best) {
    std::vector<std::vector<std::shared_ptr<KernelConfig> > > kernelConfigBench;
    kernelConfigBench.resize(batchSize+1);

    std::vector<int> kernelBatchSizes;
    batchSizePolicyToBatchSizes(batchSizePolicy, batchSize, kernelBatchSizes);

    // to prevent excessive GPU buffer re-allocation
    std::reverse(kernelBatchSizes.begin(), kernelBatchSizes.end());
    if(devices.size() == 0) {
      // sequential benchmarking
      for(auto b : kernelBatchSizes) {
	const auto ret = getDesirableKernelConfigs(convType, b, workspaceSize, ptrs, b == batchSize);
	kernelConfigBench[b].insert(kernelConfigBench[b].end(), ret.begin(), ret.end());
      }

    } else {
      // parallel benchnarking with multiple devices
      int defaultDevice;
      UCUDNN_CUDA_CHECK(cudaGetDevice(&defaultDevice));

      std::vector<std::vector<int> > dividedBatchSizes;
      dividedBatchSizes.resize(devices.size());
      {
	int i = 0;
	for(auto b : kernelBatchSizes) {
	  dividedBatchSizes[i].push_back(b);
	  i = (i+1)%devices.size();
	}
      }

      std::vector<std::thread *> threads;
      for(int i = 0; i < devices.size(); i++) {
	std::thread *thread = new std::thread([&, i] {
	    cudnnHandle_t handle;
	    UCUDNN_CUDA_CHECK(cudaSetDevice(devices[i]));
	    UCUDNN_CUDNN_CHECK(cudnnCreate(&handle));
	    {
	      // auto newDatabase = std::shared_ptr<ConvDatabase>(new ConvDatabase(*convDatabase_));
	      Optimizer optimizer(handle, convParam_, convDatabase_);
	      for(auto b : dividedBatchSizes[i]) {
		// since the pointers in ptrs cannot be necessarily used by other GPUs, we just ignore the pointers here.
		const auto ret = optimizer.getDesirableKernelConfigs(convType, b, workspaceSize, nullptr);
		kernelConfigBench[b].insert(kernelConfigBench[b].end(), ret.begin(), ret.end());
	      }
	    }
	    UCUDNN_CUDNN_CHECK(cudnnDestroy(handle));
	  });
	threads.push_back(thread);
      }

      for(auto thread : threads) {
	thread->join();
	delete thread;
      }

      UCUDNN_CUDA_CHECK(cudaSetDevice(defaultDevice));
    }

    return kernelConfigBench;
  }

}
