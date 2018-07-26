/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cudnn.h>
#include <assert.h>

#include "convParam.h"
#include "convDatabase.h"
#include "optimizer.h"
#include "kernelConfig.h"

void cudnnCheck(cudnnStatus_t status) {
  if(status != CUDNN_STATUS_SUCCESS) {
    std::cerr << "cudnn error: " << status << std::endl;
    exit(1);
  }
}

ucudnn::ConvParam createConvParam(const int batchSize) {
  const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
  const cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  const int pad = 1;
  const int stride = 1;
  const int iFilter = 3;
  const int oFilter = 64;
  const int kernel = 3;

  const int iWidth = 224;
  const int oWidth = (iWidth-kernel+pad*2)/stride+1;

  cudnnTensorDescriptor_t xDesc, yDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnCheck(cudnnCreateTensorDescriptor(&xDesc));
  cudnnCheck(cudnnCreateTensorDescriptor(&yDesc));
  cudnnCheck(cudnnCreateFilterDescriptor(&wDesc));
  cudnnCheck(cudnnCreateConvolutionDescriptor(&convDesc));

  cudnnCheck(cudnnSetTensor4dDescriptorEx(xDesc, dataType,
					  batchSize, iFilter, iWidth, iWidth,
					  iFilter*iWidth*iWidth, iWidth*iWidth, iWidth, 1));
  cudnnCheck(cudnnSetTensor4dDescriptorEx(yDesc, dataType,
					  batchSize, oFilter, oWidth, oWidth,
					  oFilter*oWidth*oWidth, oWidth*oWidth, oWidth, 1));
  cudnnCheck(cudnnSetFilter4dDescriptor(wDesc, dataType, format,
					oFilter, iFilter, kernel, kernel));
  cudnnCheck(cudnnSetConvolution2dDescriptor(convDesc,
					     pad, pad, stride, stride, 1, 1,
					     CUDNN_CROSS_CORRELATION, dataType));

  return ucudnn::ConvParam(xDesc, yDesc, wDesc, convDesc);
}

std::vector<ucudnn::cudnnConvolutionGenericAlgoPerf_t> createPerfs(const float time_fast, const float time_slow,
								   const size_t memory_fast, const size_t memory_slow,
								   const size_t workspaceSize) {
  assert(time_fast <= time_slow);
  std::vector<ucudnn::cudnnConvolutionGenericAlgoPerf_t> perfs;
  for(int i = 0; i < 2; i++) {
    ucudnn::cudnnConvolutionGenericAlgoPerf_t perf;
    perf.status = CUDNN_STATUS_SUCCESS;
    perf.algo = i;
    perf.time = (i == 0 ? time_fast : time_slow);
    perf.memory = (i == 0 ? memory_fast : memory_slow);
    perf.determinism = CUDNN_DETERMINISTIC;
    if(perf.memory < workspaceSize)
      perfs.push_back(perf);
  }
  return perfs;
}

int main() {
#ifndef UCUDNN_USE_SQLITE
  // Since this test requires SQLite3 to generate pseudo bencharmk results,
  // skip the test if it is not installed.
  exit(200);
#endif

  const char *dbFileName = "db.sqlite";
  remove(dbFileName);

  cudnnHandle_t cudnn;
  cudnnCheck(cudnnCreate(&cudnn));

  const ucudnn::ConvType convType = ucudnn::Forward;
  const size_t workspaceSize = 64UL*1024*1024;
  const int batchSize = 256;

  auto convParam = createConvParam(batchSize);
  auto db = std::make_shared<ucudnn::ConvDatabase>(dbFileName);

  const float time_fast_a = 0.1;
  const float time_fast_b = 0.01;
  const float time_slow_a = 10.0;
  const float time_slow_b = 0.1;
  const size_t memory_fast_a = 1024UL*1024;
  const size_t memory_slow_a = 1;

  for(int b = 1; b <= batchSize; b*=2) {
    const auto perfs = createPerfs(b * time_fast_a + time_fast_b,
				   b * time_slow_a + time_slow_b,
				   b * memory_fast_a,
				   b * memory_slow_a,
				   workspaceSize);
    db->insertPerfResults(convParam, convType, b, workspaceSize, perfs);
  }

  ucudnn::Optimizer optimizer(cudnn, convParam, db);

  std::vector<int> devices;
  std::vector<std::shared_ptr<ucudnn::ConvConfig> >
    configs(optimizer.getDesirableConvConfigs(convType, batchSize, workspaceSize,
					      ucudnn::powerOfTwo,
					      false, devices).first);

  std::sort(configs.begin(), configs.end(),
  	    [](std::shared_ptr<ucudnn::ConvConfig> c1, std::shared_ptr<ucudnn::ConvConfig> c2) {
  	      return c1->time() < c2->time();
  	    });

  const std::vector<std::string> trueOutputs = {
    "25.68 33554432 8",
    "25.76 16777216 16",
    "25.92 8388608 32",
    "26.24 4194304 64",
    "26.88 2097152 128",
    "28.16 1048576 256",
    "2560.10 256 1",
    "2560.20 128 2",
    "2560.40 64 4",
    "2560.80 32 8",
    "2561.60 16 16",
    "2563.20 8 32",
    "2566.40 4 64",
    "2572.80 2 128",
    "2585.60 1 256"};

  assert(configs.size() == trueOutputs.size());
  for(int i = 0; i < configs.size(); i++) {
    const auto config = configs[i];

    std::ostringstream sout;
    sout << std::fixed << std::setprecision(2) << config->time()
	 << " " << config->memory() << " " << config->kernelConfigs().size();
    const std::string output = sout.str();

    assert(output == trueOutputs[i]);
  }
}
