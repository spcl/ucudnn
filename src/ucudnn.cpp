/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <iostream>
#include <assert.h>
#include <stdint.h>

#include <cudnn.h>

#include "convParam.h"
#include "optimizer.h"
#include "ucudnn.h"
#include "ucudnnHandle.h"

using ucudnn::ConvType;
using ucudnn::ConvParam;
using ucudnn::ConvConfig;
using ucudnn::getFreeDeviceMemorySize;
using ucudnn::LayerId;

cudnnStatus_t cudnnCreate(UcudnnHandle_t *handle) {
  return cudnnCreate(handle->handle_);

//  handle->create();
//  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(UcudnnHandle_t handle) {
  return cudnnDestroy(handle.handle_);

//  handle.destroy();
//  return CUDNN_STATUS_SUCCESS;
}

// cudnnConvolution*
cudnnStatus_t cudnnConvolutionForward(
				      UcudnnHandle_t                     handle,
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
  handle.log("cudnnConvolutionForward");

  return cudnnConvolutionForward(
        handle.handle_,
        alpha,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        algo,
        workSpace,
        workSpaceSizeInBytes,
        beta,
        yDesc,
        y,
        layerId
        );

//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  return handle.convolution(convParam, convType,
//			    wDesc, convDesc,
//			    (void *) x, y, (void *) w, workSpace, workSpaceSizeInBytes,
//			    alpha, beta,
//			    layerId);
}
cudnnStatus_t cudnnConvolutionBackwardData(
					   UcudnnHandle_t                     handle,
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
        wdesc,
        w,
        dyDesc,
        dy,
        convDesc,
        algo,
        workSpace,
        workSpaceSizeInBytes,
        beta,
        dxDesc,
        dx,
        layerId
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
					     UcudnnHandle_t                     handle,
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
        dw,
        layerId
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
						  UcudnnHandle_t                     handle,
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
        algo,
        layerId
        );
//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  handle.getAlgorithm(convParam, convType, memoryLimitInBytes, layerId);
//  *algo = (cudnnConvolutionFwdAlgo_t) 0;
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t  cudnnGetConvolutionBackwardDataAlgorithm(
							UcudnnHandle_t                      handle,
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
        algo,
        layerId
        );
//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  handle.getAlgorithm(convParam, convType, memoryLimitInBytes, layerId);
//  *algo = (cudnnConvolutionBwdDataAlgo_t) 0;
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
							 UcudnnHandle_t                        handle,
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
        algo,
        layerId
  );
//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, dwDesc, convDesc);
//  handle.getAlgorithm(convParam, convType, memoryLimitInBytes, layerId);
//  *algo = (cudnnConvolutionBwdFilterAlgo_t) 0;
//  return CUDNN_STATUS_SUCCESS;
}


// cudnnGetConvolution*Algorithm_v7
cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
						     UcudnnHandle_t                     handle,
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
        perfResults,
        layerId
        );

//  return cudnnFindConvolutionForwardAlgorithm(handle, srcDesc, filterDesc, convDesc, destDesc,
//					      requestedAlgoCount, returnedAlgoCount, perfResults, layerId);
}
cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
							  UcudnnHandle_t                      handle,
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
        handle.handle,
        filterDesc,
        diffDesc,
        convDesc,
        gradDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults,
        layerId
        );

//  return cudnnFindConvolutionBackwardDataAlgorithm(handle, filterDesc, diffDesc, convDesc, gradDesc,
//						   requestedAlgoCount, returnedAlgoCount, perfResults, layerId);
}
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
							    UcudnnHandle_t                        handle,
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
        perfResults,
        layerId
        );

//  return cudnnFindConvolutionBackwardFilterAlgorithm(handle, srcDesc, diffDesc, convDesc, gradDesc,
//						     requestedAlgoCount, returnedAlgoCount, perfResults, layerId);
}

// cudnnFindConvolution*Algorithm
cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
						   UcudnnHandle_t                     handle,
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
        perfResults,
        layerId
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
							UcudnnHandle_t                     handle,
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
        perfResults,
        layerId
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
							  UcudnnHandle_t                      handle,
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
        perfResults,
        layerId
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
						     UcudnnHandle_t                     handle,
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
        workSpaceSizeInBytes,
        layerId
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
							  UcudnnHandle_t                     handle,
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
        workSpaceSizeInBytes,
        layerId
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
							    UcudnnHandle_t                      handle,
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
        workSpaceSizeInBytes,
        layerId
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
						      UcudnnHandle_t                     handle,
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
        sizeInBytes,
        layerId
        );
//  const ConvType convType = ConvType::Forward;
//  const ConvParam convParam(xDesc, yDesc, wDesc, convDesc);
//  *sizeInBytes = handle.getWorkspaceSize(convParam, convType, layerId);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
							   UcudnnHandle_t                      handle,
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
        sizeInBytes,
        layerId
        );
//  const ConvType convType = ConvType::BackwardData;
//  const ConvParam convParam(dxDesc, dyDesc, wDesc, convDesc);
//  *sizeInBytes = handle.getWorkspaceSize(convParam, convType, layerId);
//  return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
							     UcudnnHandle_t                      handle,
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
        sizeInBytes,
        layerId
        );
//  const ConvType convType = ConvType::BackwardFilter;
//  const ConvParam convParam(xDesc, dyDesc, gradDesc, convDesc);
//  *sizeInBytes = handle.getWorkspaceSize(convParam, convType, layerId);
//  return CUDNN_STATUS_SUCCESS;
}
