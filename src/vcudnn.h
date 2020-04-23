/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_UCUDNN_H_
#define UCUDNN_UCUDNN_H_

#include <unordered_map>
#include <cudnn.h>

#include "convParam.h"
#include "optimizer.h"
#include "util.h"
#include "vcudnnHandle.h"

cudnnStatus_t cudnnCreate(UcudnnHandle_t *handle);
cudnnStatus_t cudnnDestroy(UcudnnHandle_t handle);

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
				      const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
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
					   const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
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
				const ucudnn::LayerId layerId=ucudnn::LayerIdAny);

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
						  const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t  cudnnGetConvolutionBackwardDataAlgorithm(
							UcudnnHandle_t                      handle,
							const cudnnFilterDescriptor_t       wDesc,
							const cudnnTensorDescriptor_t       dyDesc,
							const cudnnConvolutionDescriptor_t  convDesc,
							const cudnnTensorDescriptor_t       dxDesc,
							cudnnConvolutionBwdDataPreference_t preference,
							size_t                              memoryLimitInBytes,
							cudnnConvolutionBwdDataAlgo_t       *algo,
							const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
							 UcudnnHandle_t                        handle,
							 const cudnnTensorDescriptor_t         xDesc,
							 const cudnnTensorDescriptor_t         dyDesc,
							 const cudnnConvolutionDescriptor_t    convDesc,
							 const cudnnFilterDescriptor_t         dwDesc,
							 cudnnConvolutionBwdFilterPreference_t preference,
							 size_t                                memoryLimitInBytes,
							 cudnnConvolutionBwdFilterAlgo_t       *algo,
							 const ucudnn::LayerId layerId=ucudnn::LayerIdAny);

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
						     const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
							  UcudnnHandle_t                      handle,
							  const cudnnFilterDescriptor_t       filterDesc,
							  const cudnnTensorDescriptor_t       diffDesc,
							  const cudnnConvolutionDescriptor_t  convDesc,
							  const cudnnTensorDescriptor_t       gradDesc,
							  const int                           requestedAlgoCount,
							  int                                *returnedAlgoCount,
							  cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							  const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
							    UcudnnHandle_t                        handle,
							    const cudnnTensorDescriptor_t         srcDesc,
							    const cudnnTensorDescriptor_t         diffDesc,
							    const cudnnConvolutionDescriptor_t    convDesc,
							    const cudnnFilterDescriptor_t         gradDesc,
							    const int                             requestedAlgoCount,
							    int                                  *returnedAlgoCount,
							    cudnnConvolutionBwdFilterAlgoPerf_t  *perfResults,
							    const ucudnn::LayerId layerId=ucudnn::LayerIdAny);

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
						   const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
							UcudnnHandle_t                     handle,
							const cudnnFilterDescriptor_t      wDesc,
							const cudnnTensorDescriptor_t      dyDesc,
							const cudnnConvolutionDescriptor_t convDesc,
							const cudnnTensorDescriptor_t      dxDesc,
							const int                          requestedAlgoCount,
							int                                *returnedAlgoCount,
							cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
							  UcudnnHandle_t                      handle,
							  const cudnnTensorDescriptor_t       xDesc,
							  const cudnnTensorDescriptor_t       dyDesc,
							  const cudnnConvolutionDescriptor_t  convDesc,
							  const cudnnFilterDescriptor_t       dwDesc,
							  const int                           requestedAlgoCount,
							  int                                 *returnedAlgoCount,
							  cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
							  const ucudnn::LayerId layerId=ucudnn::LayerIdAny);

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
						     const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
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
							  const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
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
							    const ucudnn::LayerId layerId=ucudnn::LayerIdAny);

// cudnnGetConvolution*WorkspaceSize
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
						      UcudnnHandle_t                     handle,
						      const cudnnTensorDescriptor_t      xDesc,
						      const cudnnFilterDescriptor_t      wDesc,
						      const cudnnConvolutionDescriptor_t convDesc,
						      const cudnnTensorDescriptor_t      yDesc,
						      cudnnConvolutionFwdAlgo_t          algo,
						      size_t                             *sizeInBytes,
						      const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
							   UcudnnHandle_t                      handle,
							   const cudnnFilterDescriptor_t       wDesc,
							   const cudnnTensorDescriptor_t       dyDesc,
							   const cudnnConvolutionDescriptor_t  convDesc,
							   const cudnnTensorDescriptor_t       dxDesc,
							   cudnnConvolutionBwdDataAlgo_t       algo,
							   size_t                              *sizeInBytes,
							   const ucudnn::LayerId layerId=ucudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
							     UcudnnHandle_t                      handle,
							     const cudnnTensorDescriptor_t       xDesc,
							     const cudnnTensorDescriptor_t       dyDesc,
							     const cudnnConvolutionDescriptor_t  convDesc,
							     const cudnnFilterDescriptor_t       gradDesc,
							     cudnnConvolutionBwdFilterAlgo_t     algo,
							     size_t                              *sizeInBytes,
							     const ucudnn::LayerId layerId=ucudnn::LayerIdAny);

#endif
