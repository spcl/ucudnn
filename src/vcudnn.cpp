/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <iostream>
#include <assert.h>
#include <stdint.h>
#include <sstream>

#include <cudnn.h>

#include "convParam.h"
#include "optimizer.h"
#include "vcudnn.h"
#include "vcudnnHandle.h"
#include "batch_operations.h"
#include "util.h"
#include "state.h"

using vcudnn::ConvType;
using vcudnn::ConvParam;
using vcudnn::ConvConfig;
using vcudnn::getFreeDeviceMemorySize;
using vcudnn::LayerId;
using namespace vcudnn;
using namespace std;

cudnnStatus_t cudnnCreate(VcudnnHandle_t *handle) {
  return cudnnCreate(& handle->handle_);

//  handle->create();
//  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(VcudnnHandle_t handle) {
  return cudnnDestroy(handle.handle_);

//  handle.destroy();
//  return CUDNN_STATUS_SUCCESS;
}

// cudnnConvolution*
cudnnStatus_t cudnnConvolutionForward(
				      VcudnnHandle_t                     handle,
				      const void                         *alpha,
				      const cudnnTensorDescriptor_t      xDesc,
				      const void                         *x,
				      const cudnnFilterDescriptor_t      wDesc,
				      const void                         *w,
				      const cudnnConvolutionDescriptor_t convDesc,
				      cudnnConvolutionFwdAlgo_t          algo,
				      void                               *workSpace,
				      size_t                             workSpaceSizeInBytes,
				      const void                         *beta,
				      const cudnnTensorDescriptor_t      yDesc,
				      void                               *y,
				      const LayerId layerId) {
  // forward all conv calls for now
  Tensor4DDesc dx;
  Tensor4DDesc dy;
  vector<pint> mapping;

  // TODO: this can cause undefined behaviors if the original object x is pointing to is indeed const
  void * nx = const_cast<void*>(x);
  cudnnTensorDescriptor_t & tdx = const_cast<cudnnTensorDescriptor_t&>(xDesc);
  cudnnTensorDescriptor_t & tdy = const_cast<cudnnTensorDescriptor_t&>(yDesc);

  bool can_apply_shrink = read_4d_desc(xDesc, &dx) && read_4d_desc(yDesc, &dy);

  if(can_apply_shrink) {
    stringstream ss;
    ss << "cudnnConvolutionForward on ";
    ss << dx.n << " x " << dx.c << " x " << dx.w << " x " << dx.h << ", DT = " << dx.dataType;
    handle.log(ss.str());

    State * s = get_state();
    s->reinit(dx.n);
    mapping = applyBatchMask(xDesc, nx, s->getMask());

    // change descriptors
    size_t new_batch_size = s->getBatchSize();
    cudnnSetTensor4dDescriptorEx(
      tdx,
      dx.dataType,
      new_batch_size,
      dx.c,
      dx.h,
      dx.w,
      dx.nStride,
      dx.cStride,
      dx.hStride,
      dx.wStride
    );

    cudnnSetTensor4dDescriptorEx(
      tdy,
      dy.dataType,
      new_batch_size,
      dy.c,
      dy.h,
      dy.w,
      dy.nStride,
      dy.cStride,
      dy.hStride,
      dy.wStride
    );
    handle.log("compressed x and y to " + to_string(new_batch_size) + " from " + to_string(s->getOldBatchSize()));
  } else {
    handle.log("cudnnConvolutionForward -- getting desc failed");
  }

  auto result = cudnnConvolutionForward(
        handle.handle_,
        alpha,
        tdx,
        x,
        wDesc,
        w,
        convDesc,
        algo,
        workSpace,
        workSpaceSizeInBytes,
        beta,
        tdy,
        y
        );

  if(can_apply_shrink) {
    // undo mapping for conv input and output
    State * s = get_state();
    revertBatchMask(xDesc, nx, s->getMask(), mapping);
    revertBatchMask(yDesc, y, s->getMask(), mapping);

    // undo descriptor changes
    size_t old_batch_size = s->getOldBatchSize();
    cudnnSetTensor4dDescriptorEx(
      tdx,
      dx.dataType,
      old_batch_size,
      dx.c,
      dx.h,
      dx.w,
      dx.nStride,
      dx.cStride,
      dx.hStride,
      dx.wStride
    );

    cudnnSetTensor4dDescriptorEx(
      tdy,
      dy.dataType,
      old_batch_size,
      dy.c,
      dy.h,
      dy.w,
      dy.nStride,
      dy.cStride,
      dy.hStride,
      dy.wStride
    );

    handle.log("reverted mask for x and y to " + to_string(s->getOldBatchSize()) + " from " + to_string(s->getBatchSize()));
  }

  return result;
//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  return handle.convolution(convParam, convType,
//			    wDesc, convDesc,
//			    (void *) x, y, (void *) w, workSpace, workSpaceSizeInBytes,
//			    alpha, beta,
//			    layerId);
}
cudnnStatus_t cudnnConvolutionBackwardData(
					   VcudnnHandle_t                     handle,
					   const void                         *alpha,
					   const cudnnFilterDescriptor_t      wDesc,
					   const void                         *w,
					   const cudnnTensorDescriptor_t      dyDesc,
					   const void                         *dy,
					   const cudnnConvolutionDescriptor_t convDesc,
					   cudnnConvolutionBwdDataAlgo_t      algo,
					   void                               *workSpace,
					   size_t                             workSpaceSizeInBytes,
					   const void                         *beta,
					   const cudnnTensorDescriptor_t      dxDesc,
					   void                               *dx,
					   const LayerId layerId) {
  handle.log("cudnnConvolutionBackwardData");
  return cudnnConvolutionBackwardData(
        handle.handle_,
        alpha,
        wDesc,
        w,
        dyDesc,
        dy,
        convDesc,
        algo,
        workSpace,
        workSpaceSizeInBytes,
        beta,
        dxDesc,
        dx
  );
//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  return handle.convolution(convParam, convType,
//			    wDesc, convDesc,
//			    dx, (void *) dy, (void *) w, workSpace, workSpaceSizeInBytes,
//			    alpha, beta,
//			    layerId);
}
cudnnStatus_t cudnnConvolutionBackwardFilter(
					     VcudnnHandle_t                     handle,
					     const void                         *alpha,
					     const cudnnTensorDescriptor_t      xDesc,
					     const void                         *x,
					     const cudnnTensorDescriptor_t      dyDesc,
					     const void                         *dy,
					     const cudnnConvolutionDescriptor_t convDesc,
					     cudnnConvolutionBwdFilterAlgo_t    algo,
					     void                               *workSpace,
					     size_t                             workSpaceSizeInBytes,
					     const void                         *beta,
					     const cudnnFilterDescriptor_t      dwDesc,
					     void                               *dw,
					     const LayerId layerId) {
  handle.log("cudnnConvolutionBackwardFilter");
  return cudnnConvolutionBackwardFilter(
        handle.handle_,
        alpha,
        xDesc,
        x,
        dyDesc,
        dy,
        convDesc,
        algo,
        workSpace,
        workSpaceSizeInBytes,
        beta,
        dwDesc,
        dw
  );

//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, dwDesc, convDesc);
//  return handle.convolution(convParam, convType,
//			    dwDesc, convDesc,
//			    (void *) x, (void *) dy, dw, workSpace, workSpaceSizeInBytes,
//			    alpha, beta,
//			    layerId);
}

// cudnnGetConvolution*Algorithm
cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
						  VcudnnHandle_t                     handle,
						  const cudnnTensorDescriptor_t      xDesc,
						  const cudnnFilterDescriptor_t      wDesc,
						  const cudnnConvolutionDescriptor_t convDesc,
						  const cudnnTensorDescriptor_t      yDesc,
						  cudnnConvolutionFwdPreference_t    preference,
						  size_t                             memoryLimitInBytes,
						  cudnnConvolutionFwdAlgo_t          *algo,
						  const LayerId layerId) {
  handle.log("cudnnGetConvolutionForwardAlgorithm");
  return cudnnGetConvolutionForwardAlgorithm(
        handle.handle_,
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        preference,
        memoryLimitInBytes,
        algo
        );
//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  handle.getAlgorithm(convParam, convType, memoryLimitInBytes, layerId);
//  *algo = (cudnnConvolutionFwdAlgo_t) 0;
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t  cudnnGetConvolutionBackwardDataAlgorithm(
							VcudnnHandle_t                      handle,
							const cudnnFilterDescriptor_t       wDesc,
							const cudnnTensorDescriptor_t       dyDesc,
							const cudnnConvolutionDescriptor_t  convDesc,
							const cudnnTensorDescriptor_t       dxDesc,
							cudnnConvolutionBwdDataPreference_t preference,
							size_t                              memoryLimitInBytes,
							cudnnConvolutionBwdDataAlgo_t       *algo,
							const LayerId layerId) {
  handle.log("cudnnGetConvolutionBackwardDataAlgorithm");
  return cudnnGetConvolutionBackwardDataAlgorithm(
        handle.handle_,
        wDesc,
        dyDesc,
        convDesc,
        dxDesc,
        preference,
        memoryLimitInBytes,
        algo
        );
//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  handle.getAlgorithm(convParam, convType, memoryLimitInBytes, layerId);
//  *algo = (cudnnConvolutionBwdDataAlgo_t) 0;
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
							 VcudnnHandle_t                        handle,
							 const cudnnTensorDescriptor_t         xDesc,
							 const cudnnTensorDescriptor_t         dyDesc,
							 const cudnnConvolutionDescriptor_t    convDesc,
							 const cudnnFilterDescriptor_t         dwDesc,
							 cudnnConvolutionBwdFilterPreference_t preference,
							 size_t                                memoryLimitInBytes,
							 cudnnConvolutionBwdFilterAlgo_t       *algo,
							 const LayerId layerId) {
  handle.log("cudnnGetConvolutionBackwardFilterAlgorithm");
  return cudnnGetConvolutionBackwardFilterAlgorithm(
        handle.handle_,
        xDesc,
        dyDesc,
        convDesc,
        dwDesc,
        preference,
        memoryLimitInBytes,
        algo
  );
//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, dwDesc, convDesc);
//  handle.getAlgorithm(convParam, convType, memoryLimitInBytes, layerId);
//  *algo = (cudnnConvolutionBwdFilterAlgo_t) 0;
//  return CUDNN_STATUS_SUCCESS;
}


// cudnnGetConvolution*Algorithm_v7
cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
						     VcudnnHandle_t                     handle,
						     const cudnnTensorDescriptor_t      srcDesc,
						     const cudnnFilterDescriptor_t      filterDesc,
						     const cudnnConvolutionDescriptor_t convDesc,
						     const cudnnTensorDescriptor_t      destDesc,
						     const int                          requestedAlgoCount,
						     int                               *returnedAlgoCount,
						     cudnnConvolutionFwdAlgoPerf_t     *perfResults,
						     const LayerId layerId) {
  handle.log("cudnnGetConvolutionForwardAlgorithm_v7");
  return cudnnGetConvolutionForwardAlgorithm_v7(
        handle.handle_,
        srcDesc,
        filterDesc,
        convDesc,
        destDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults
        );

//  return cudnnFindConvolutionForwardAlgorithm(handle, srcDesc, filterDesc, convDesc, destDesc,
//					      requestedAlgoCount, returnedAlgoCount, perfResults, layerId);
}
cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
							  VcudnnHandle_t                      handle,
							  const cudnnFilterDescriptor_t       filterDesc,
							  const cudnnTensorDescriptor_t       diffDesc,
							  const cudnnConvolutionDescriptor_t  convDesc,
							  const cudnnTensorDescriptor_t       gradDesc,
							  const int                           requestedAlgoCount,
							  int                                *returnedAlgoCount,
							  cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							  const LayerId layerId) {
  handle.log("cudnnGetConvolutionBackwardDataAlgorithm_v7");
  return cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle.handle_,
        filterDesc,
        diffDesc,
        convDesc,
        gradDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults
        );

//  return cudnnFindConvolutionBackwardDataAlgorithm(handle, filterDesc, diffDesc, convDesc, gradDesc,
//						   requestedAlgoCount, returnedAlgoCount, perfResults, layerId);
}
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
							    VcudnnHandle_t                        handle,
							    const cudnnTensorDescriptor_t         srcDesc,
							    const cudnnTensorDescriptor_t         diffDesc,
							    const cudnnConvolutionDescriptor_t    convDesc,
							    const cudnnFilterDescriptor_t         gradDesc,
							    const int                             requestedAlgoCount,
							    int                                  *returnedAlgoCount,
							    cudnnConvolutionBwdFilterAlgoPerf_t  *perfResults,
							    const LayerId layerId) {
  handle.log("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
  return cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle.handle_,
        srcDesc,
        diffDesc,
        convDesc,
        gradDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults
        );

//  return cudnnFindConvolutionBackwardFilterAlgorithm(handle, srcDesc, diffDesc, convDesc, gradDesc,
//						     requestedAlgoCount, returnedAlgoCount, perfResults, layerId);
}

// cudnnFindConvolution*Algorithm
cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
						   VcudnnHandle_t                     handle,
						   const cudnnTensorDescriptor_t      xDesc,
						   const cudnnFilterDescriptor_t      wDesc,
						   const cudnnConvolutionDescriptor_t convDesc,
						   const cudnnTensorDescriptor_t      yDesc,
						   const int                          requestedAlgoCount,
						   int                                *returnedAlgoCount,
						   cudnnConvolutionFwdAlgoPerf_t      *perfResults,
						   const LayerId layerId) {
  handle.log("cudnnFindConvolutionForwardAlgorithm");
  return cudnnFindConvolutionForwardAlgorithm(
        handle.handle_,
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults
        );

//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  std::shared_ptr<ConvConfig> convConfig = handle.findAlgorithmEx(convParam, convType,
//								  nullptr, nullptr, nullptr, nullptr,
//								  getFreeDeviceMemorySize(),
//								  layerId);
//  convConfig->setPseudoAlgoPerfs<cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t>(perfResults);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
							VcudnnHandle_t                     handle,
							const cudnnFilterDescriptor_t      wDesc,
							const cudnnTensorDescriptor_t      dyDesc,
							const cudnnConvolutionDescriptor_t convDesc,
							const cudnnTensorDescriptor_t      dxDesc,
							const int                          requestedAlgoCount,
							int                                *returnedAlgoCount,
							cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							const LayerId layerId) {

  handle.log("cudnnFindConvolutionBackwardDataAlgorithm");
  return cudnnFindConvolutionBackwardDataAlgorithm(
        handle.handle_,
        wDesc,
        dyDesc,
        convDesc,
        dxDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults
        );

//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  std::shared_ptr<ConvConfig> convConfig = handle.findAlgorithmEx(convParam, convType,
//								  nullptr, nullptr, nullptr, nullptr,
//								  getFreeDeviceMemorySize(),
//								  layerId);
//  convConfig->setPseudoAlgoPerfs<cudnnConvolutionBwdDataAlgoPerf_t, cudnnConvolutionBwdDataAlgo_t>(perfResults);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
							  VcudnnHandle_t                      handle,
							  const cudnnTensorDescriptor_t       xDesc,
							  const cudnnTensorDescriptor_t       dyDesc,
							  const cudnnConvolutionDescriptor_t  convDesc,
							  const cudnnFilterDescriptor_t       dwDesc,
							  const int                           requestedAlgoCount,
							  int                                 *returnedAlgoCount,
							  cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
							  const LayerId layerId) {
  handle.log("cudnnFindConvolutionBackwardFilterAlgorithm");
  return cudnnFindConvolutionBackwardFilterAlgorithm(
        handle.handle_,
        xDesc,
        dyDesc,
        convDesc,
        dwDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults
        );

//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, dwDesc, convDesc);
//  std::shared_ptr<ConvConfig> convConfig = handle.findAlgorithmEx(convParam, convType,
//								  nullptr, nullptr, nullptr, nullptr,
//								  getFreeDeviceMemorySize(),
//								  layerId);
//  convConfig->setPseudoAlgoPerfs<cudnnConvolutionBwdFilterAlgoPerf_t, cudnnConvolutionBwdFilterAlgo_t>(perfResults);
//  return CUDNN_STATUS_SUCCESS;
}

// cudnnFindConvolution*AlgorithmEx
cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
						     VcudnnHandle_t                     handle,
						     const cudnnTensorDescriptor_t      xDesc,
						     const void                         *x,
						     const cudnnFilterDescriptor_t      wDesc,
						     const void                         *w,
						     const cudnnConvolutionDescriptor_t convDesc,
						     const cudnnTensorDescriptor_t      yDesc,
						     void                               *y,
						     const int                          requestedAlgoCount,
						     int                                *returnedAlgoCount,
						     cudnnConvolutionFwdAlgoPerf_t      *perfResults,
						     void                               *workSpace,
						     size_t                             workSpaceSizeInBytes,
						     const LayerId layerId) {
  handle.log("cudnnFindConvolutionForwardAlgorithmEx");
  return cudnnFindConvolutionForwardAlgorithmEx(
        handle.handle_,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        yDesc,
        y,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults,
        workSpace,
        workSpaceSizeInBytes
        );

//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  std::shared_ptr<ConvConfig> convConfig = handle.findAlgorithmEx(convParam, convType,
//								  (void *) x, y, (void *) w, workSpace,
//								  workSpaceSizeInBytes,
//								  layerId);
//  convConfig->setPseudoAlgoPerfs<cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t>(perfResults);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
							  VcudnnHandle_t                     handle,
							  const cudnnFilterDescriptor_t      wDesc,
							  const void                         *w,
							  const cudnnTensorDescriptor_t      dyDesc,
							  const void                         *dy,
							  const cudnnConvolutionDescriptor_t convDesc,
							  const cudnnTensorDescriptor_t      dxDesc,
							  void                               *dx,
							  const int                          requestedAlgoCount,
							  int                                *returnedAlgoCount,
							  cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							  void                               *workSpace,
							  size_t                             workSpaceSizeInBytes,
							  const LayerId layerId) {
  handle.log("cudnnFindConvolutionBackwardDataAlgorithmEx");
  return cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle.handle_,
        wDesc,
        w,
        dyDesc,
        dy,
        convDesc,
        dxDesc,
        dx,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults,
        workSpace,
        workSpaceSizeInBytes
        );

//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  std::shared_ptr<ConvConfig> convConfig = handle.findAlgorithmEx(convParam, convType,
//								  dx, (void *) dy, (void *) w, workSpace,
//								  workSpaceSizeInBytes,
//								  layerId);
//  convConfig->setPseudoAlgoPerfs<cudnnConvolutionBwdDataAlgoPerf_t, cudnnConvolutionBwdDataAlgo_t>(perfResults);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
							    VcudnnHandle_t                      handle,
							    const cudnnTensorDescriptor_t       xDesc,
							    const void                          *x,
							    const cudnnTensorDescriptor_t       dyDesc,
							    const void                          *y,
							    const cudnnConvolutionDescriptor_t  convDesc,
							    const cudnnFilterDescriptor_t       dwDesc,
							    void                                *dw,
							    const int                           requestedAlgoCount,
							    int                                 *returnedAlgoCount,
							    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
							    void                                *workSpace,
							    size_t                              workSpaceSizeInBytes,
							    const LayerId layerId) {
  handle.log("cudnnFindConvolutionBackwardFilterAlgorithmEx");
  return cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle.handle_,
        xDesc,
        x,
        dyDesc,
        y,
        convDesc,
        dwDesc,
        dw,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults,
        workSpace,
        workSpaceSizeInBytes
        );

//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, dwDesc, convDesc);
//  std::shared_ptr<ConvConfig> convConfig = handle.findAlgorithmEx(convParam, convType,
//								  (void *) x, (void *) y, dw, workSpace,
//								  workSpaceSizeInBytes,
//								  layerId);
//  convConfig->setPseudoAlgoPerfs<cudnnConvolutionBwdFilterAlgoPerf_t, cudnnConvolutionBwdFilterAlgo_t>(perfResults);
//  return CUDNN_STATUS_SUCCESS;
}

// cudnnGetConvolution*WorkspaceSize
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
						      VcudnnHandle_t                     handle,
						      const cudnnTensorDescriptor_t      xDesc,
						      const cudnnFilterDescriptor_t      wDesc,
						      const cudnnConvolutionDescriptor_t convDesc,
						      const cudnnTensorDescriptor_t      yDesc,
						      cudnnConvolutionFwdAlgo_t          algo,
						      size_t                             *sizeInBytes,
						      const LayerId layerId) {
  handle.log("cudnnGetConvolutionForwardWorkspaceSize");
  return cudnnGetConvolutionForwardWorkspaceSize(
        handle.handle_,
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        algo,
        sizeInBytes
        );
//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  *sizeInBytes = handle.getWorkspaceSize(convParam, convType, layerId);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
							   VcudnnHandle_t                      handle,
							   const cudnnFilterDescriptor_t       wDesc,
							   const cudnnTensorDescriptor_t       dyDesc,
							   const cudnnConvolutionDescriptor_t  convDesc,
							   const cudnnTensorDescriptor_t       dxDesc,
							   cudnnConvolutionBwdDataAlgo_t       algo,
							   size_t                              *sizeInBytes,
							   const LayerId layerId) {
  handle.log("cudnnGetConvolutionBackwardDataWorkspaceSize");
  return cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle.handle_,
        wDesc,
        dyDesc,
        convDesc,
        dxDesc,
        algo,
        sizeInBytes
        );
//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  *sizeInBytes = handle.getWorkspaceSize(convParam, convType, layerId);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
							     VcudnnHandle_t                      handle,
							     const cudnnTensorDescriptor_t       xDesc,
							     const cudnnTensorDescriptor_t       dyDesc,
							     const cudnnConvolutionDescriptor_t  convDesc,
							     const cudnnFilterDescriptor_t       gradDesc,
							     cudnnConvolutionBwdFilterAlgo_t     algo,
							     size_t                              *sizeInBytes,
							     const LayerId layerId) {
  handle.log("cudnnGetConvolutionBackwardFilterWorkspaceSize");
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle.handle_,
        xDesc,
        dyDesc,
        convDesc,
        gradDesc,
        algo,
        sizeInBytes
        );
//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, gradDesc, convDesc);
//  *sizeInBytes = handle.getWorkspaceSize(convParam, convType, layerId);
//  return CUDNN_STATUS_SUCCESS;
}
