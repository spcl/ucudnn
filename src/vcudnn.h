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

cudnnStatus_t cudnnCreate(VcudnnHandle_t *handle);
cudnnStatus_t cudnnDestroy(VcudnnHandle_t handle);

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
				      const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
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
					   const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
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
				const vcudnn::LayerId layerId=vcudnn::LayerIdAny);

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
						  const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t  cudnnGetConvolutionBackwardDataAlgorithm(
							VcudnnHandle_t                      handle,
							const cudnnFilterDescriptor_t       wDesc,
							const cudnnTensorDescriptor_t       dyDesc,
							const cudnnConvolutionDescriptor_t  convDesc,
							const cudnnTensorDescriptor_t       dxDesc,
							cudnnConvolutionBwdDataPreference_t preference,
							size_t                              memoryLimitInBytes,
							cudnnConvolutionBwdDataAlgo_t       *algo,
							const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
							 VcudnnHandle_t                        handle,
							 const cudnnTensorDescriptor_t         xDesc,
							 const cudnnTensorDescriptor_t         dyDesc,
							 const cudnnConvolutionDescriptor_t    convDesc,
							 const cudnnFilterDescriptor_t         dwDesc,
							 cudnnConvolutionBwdFilterPreference_t preference,
							 size_t                                memoryLimitInBytes,
							 cudnnConvolutionBwdFilterAlgo_t       *algo,
							 const vcudnn::LayerId layerId=vcudnn::LayerIdAny);

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
						     const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
							  VcudnnHandle_t                      handle,
							  const cudnnFilterDescriptor_t       filterDesc,
							  const cudnnTensorDescriptor_t       diffDesc,
							  const cudnnConvolutionDescriptor_t  convDesc,
							  const cudnnTensorDescriptor_t       gradDesc,
							  const int                           requestedAlgoCount,
							  int                                *returnedAlgoCount,
							  cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							  const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
							    VcudnnHandle_t                        handle,
							    const cudnnTensorDescriptor_t         srcDesc,
							    const cudnnTensorDescriptor_t         diffDesc,
							    const cudnnConvolutionDescriptor_t    convDesc,
							    const cudnnFilterDescriptor_t         gradDesc,
							    const int                             requestedAlgoCount,
							    int                                  *returnedAlgoCount,
							    cudnnConvolutionBwdFilterAlgoPerf_t  *perfResults,
							    const vcudnn::LayerId layerId=vcudnn::LayerIdAny);

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
						   const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
							VcudnnHandle_t                     handle,
							const cudnnFilterDescriptor_t      wDesc,
							const cudnnTensorDescriptor_t      dyDesc,
							const cudnnConvolutionDescriptor_t convDesc,
							const cudnnTensorDescriptor_t      dxDesc,
							const int                          requestedAlgoCount,
							int                                *returnedAlgoCount,
							cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
							const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
							  VcudnnHandle_t                      handle,
							  const cudnnTensorDescriptor_t       xDesc,
							  const cudnnTensorDescriptor_t       dyDesc,
							  const cudnnConvolutionDescriptor_t  convDesc,
							  const cudnnFilterDescriptor_t       dwDesc,
							  const int                           requestedAlgoCount,
							  int                                 *returnedAlgoCount,
							  cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
							  const vcudnn::LayerId layerId=vcudnn::LayerIdAny);

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
						     const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
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
							  const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
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
							    const vcudnn::LayerId layerId=vcudnn::LayerIdAny);

// cudnnGetConvolution*WorkspaceSize
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
						      VcudnnHandle_t                     handle,
						      const cudnnTensorDescriptor_t      xDesc,
						      const cudnnFilterDescriptor_t      wDesc,
						      const cudnnConvolutionDescriptor_t convDesc,
						      const cudnnTensorDescriptor_t      yDesc,
						      cudnnConvolutionFwdAlgo_t          algo,
						      size_t                             *sizeInBytes,
						      const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
							   VcudnnHandle_t                      handle,
							   const cudnnFilterDescriptor_t       wDesc,
							   const cudnnTensorDescriptor_t       dyDesc,
							   const cudnnConvolutionDescriptor_t  convDesc,
							   const cudnnTensorDescriptor_t       dxDesc,
							   cudnnConvolutionBwdDataAlgo_t       algo,
							   size_t                              *sizeInBytes,
							   const vcudnn::LayerId layerId=vcudnn::LayerIdAny);
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
							     VcudnnHandle_t                      handle,
							     const cudnnTensorDescriptor_t       xDesc,
							     const cudnnTensorDescriptor_t       dyDesc,
							     const cudnnConvolutionDescriptor_t  convDesc,
							     const cudnnFilterDescriptor_t       gradDesc,
							     cudnnConvolutionBwdFilterAlgo_t     algo,
							     size_t                              *sizeInBytes,
							     const vcudnn::LayerId layerId=vcudnn::LayerIdAny);

// vcudnn API

/* Set the initial batch size */
void vcudnnSetBatchSize(std::size_t size);

/* Mark a item from the batch for removal for subsequent operations */
std::size_t vcudnnRemoveFromBatch(std::size_t idx);

/* Retrieve the size of the reduced batch i.e. after performing the reduction */
std::size_t vcudnnGetReducedBatchSize();



#endif
