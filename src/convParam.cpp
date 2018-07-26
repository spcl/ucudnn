/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include <assert.h>

#include "util.h"

#include "convConfig.h"
#include "convParam.h"

namespace ucudnn {

  void ConvParam::initialize(const cudnnTensorDescriptor_t xDesc,
			     const cudnnTensorDescriptor_t yDesc,
			     const cudnnFilterDescriptor_t wDesc,
			     const cudnnConvolutionDescriptor_t convDesc) {

    // xDesc
    {
      if(getTensorDimensionCount(xDesc) != 4)
	UCUDNN_ERROR_EXIT("u-cuDNN does not support nbDims != 4.");
      int n, c, h, w, nStride, cStride, hStride, wStride;
      cudnnDataType_t dataType;
      UCUDNN_CUDNN_CHECK(cudnnGetTensor4dDescriptor(xDesc, &dataType,
						    &n, &c, &h, &w,
						    &nStride, &cStride, &hStride, &wStride));
      this->xType_ = dataType;
      this->n_ = n;
      this->c_in_ = c;
      this->h_in_ = h;
      this->w_in_ = w;
      this->xStride_.set(nStride, cStride, hStride, wStride);
    }

    // yDesc
    {
      if(getTensorDimensionCount(xDesc) != 4)
	UCUDNN_ERROR_EXIT("u-cuDNN does not support nbDims != 4.");
      int n, c, h, w, nStride, cStride, hStride, wStride;
      cudnnDataType_t dataType;
      UCUDNN_CUDNN_CHECK(cudnnGetTensor4dDescriptor(yDesc, &dataType,
						    &n, &c, &h, &w,
						    &nStride, &cStride, &hStride, &wStride));
      this->yType_ = dataType;
      assert(n == this->n_);
      this->c_out_ = c;
      this->h_out_ = h;
      this->w_out_ = w;
      this->yStride_.set(nStride, cStride, hStride, wStride);
    }

    // wDesc
    {
      int k, c, h, w;
      cudnnDataType_t dataType;
      cudnnTensorFormat_t format;
      UCUDNN_CUDNN_CHECK(cudnnGetFilter4dDescriptor(wDesc,
						    &dataType, &format,
						    &k, &c, &h, &w));
      this->filterType_ = dataType;
      this->filterFormat_ = format;
      assert(this->c_in_ == c);
      assert(this->c_out_ == k);
      this->h_kernel_ = h;
      this->w_kernel_ = w;
    }

    // convDesc
    {
      int pad_h, pad_w;
      int u, v;
      int dilation_h, dilation_w;
      cudnnConvolutionMode_t mode;
      cudnnDataType_t computeType;
      UCUDNN_CUDNN_CHECK(cudnnGetConvolution2dDescriptor(convDesc,
							 &pad_h, &pad_w,
							 &u, &v,
							 &dilation_h, &dilation_w,
							 &mode, &computeType));
      this->h_pad_ = pad_h;
      this->w_pad_ = pad_w;
      this->h_stride_ = u;
      this->w_stride_ = v;
      this->h_dilation_ = dilation_h;
      this->w_dilation_ = dilation_w;
      this->convMode_ = mode;
      this->computeType_ = computeType;
    }
#if CUDNN_HAS_MATHTYPE
    UCUDNN_CUDNN_CHECK(cudnnGetConvolutionMathType(convDesc, &this->mathType_));
#endif
  }

  void ConvParam::setXDesc(cudnnTensorDescriptor_t xDesc, const int batchSize) const {
    UCUDNN_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(xDesc, xType_,
						    batchSize, c_in_, h_in_, w_in_,
						    xStride_.n(),
						    xStride_.c(),
						    xStride_.h(),
						    xStride_.w()));
  }

  void ConvParam::setYDesc(cudnnTensorDescriptor_t yDesc, const int batchSize) const {
    UCUDNN_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(yDesc, yType_,
						    batchSize, c_out_, h_out_, w_out_,
						    yStride_.n(),
						    yStride_.c(),
						    yStride_.h(),
						    yStride_.w()));
  }

  void ConvParam::setWDesc(cudnnFilterDescriptor_t wDesc) const {
    UCUDNN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc,
						  filterType_, filterFormat_,
						  c_out_, c_in_, h_kernel_, w_kernel_));
  }

  void ConvParam::setConvDesc(cudnnConvolutionDescriptor_t convDesc) const {
    UCUDNN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
						       h_pad_, w_pad_,
						       h_stride_, w_stride_,
						       h_dilation_, w_dilation_,
						       convMode_, computeType_));
#if CUDNN_HAS_MATHTYPE
    UCUDNN_CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, mathType_));
#endif
  }


  void ConvParam::setScalingFactor(double *factor, const double val) const {
    switch(filterType_) {
    case CUDNN_DATA_HALF:
    case CUDNN_DATA_FLOAT:
      {
	void *ptr = factor;
	*((float *) ptr) = val;
	break;
      }
    case CUDNN_DATA_DOUBLE:
      *factor = val;
      break;
    default:
      assert(false);
    }
  }

  double ConvParam::getScalingFactor(const double *factor) const {
    switch(filterType_) {
    case CUDNN_DATA_HALF:
    case CUDNN_DATA_FLOAT:
      {
	float val;
	UCUDNN_CUDA_CHECK(cudaMemcpy(&val, factor, sizeof(float), cudaMemcpyDefault));
	return val;
      }
    case CUDNN_DATA_DOUBLE:
      {
	double val;
	UCUDNN_CUDA_CHECK(cudaMemcpy(&val, factor, sizeof(double), cudaMemcpyDefault));
	return val;
      }
    default:
      assert(false);
    }
  }

}
