/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_CONV_PARAM_H_
#define UCUDNN_CONV_PARAM_H_

#include <algorithm>
#include <vector>
#include <cudnn.h>
#include "convConfig.h"
#include "util.h"

namespace vcudnn {

  class TensorStride {
  public:
    void set(const int n, const int c, const int h, const int w) {
      n_ = n;
      c_ = c;
      h_ = h;
      w_ = w;
    }

    std::string toString() const {
      return "[" + std::to_string(n_)
	+ ", " + std::to_string(c_)
	+ ", " + std::to_string(h_)
	+ ", " + std::to_string(w_) + "]";
    }

    size_t hash() const {
      return std::hash<std::string>()(toString());
    }

    inline bool operator==(const TensorStride &rhs) const {
      return true
	&& n_ == rhs.n_
	&& c_ == rhs.c_
	&& h_ == rhs.h_
	&& w_ == rhs.w_;
    }

    int n() const { return n_; }
    int c() const { return c_; }
    int h() const { return h_; }
    int w() const { return w_; }

  private:
    int n_, c_, h_, w_;
  };

  // an object to store all parameters of a convolution layer
  class ConvParam {
  public:
    ConvParam(const cudnnTensorDescriptor_t xDesc,
	      const cudnnTensorDescriptor_t yDesc,
	      const cudnnFilterDescriptor_t wDesc,
	      const cudnnConvolutionDescriptor_t convDesc) {
      initialize(xDesc, yDesc, wDesc, convDesc);
    }

    void initialize(const cudnnTensorDescriptor_t xDesc,
		    const cudnnTensorDescriptor_t yDesc,
		    const cudnnFilterDescriptor_t wDesc,
		    const cudnnConvolutionDescriptor_t convDesc);

    void setXDesc(cudnnTensorDescriptor_t xDesc, const int batchSize) const;
    void setYDesc(cudnnTensorDescriptor_t yDesc, const int batchSize) const;
    void setWDesc(cudnnFilterDescriptor_t wDesc) const;
    void setConvDesc(cudnnConvolutionDescriptor_t convDesc) const;

    cudnnDataType_t getXDataType() const { return xType_; }
    cudnnDataType_t getYDataType() const { return yType_; }
    cudnnDataType_t getWDataType() const { return filterType_; }

    static cudnnTensorFormat_t getTensorFormat(const int n, const int c, const int h, const int w, const TensorStride stride) {
      if(stride.w() == 1 &&  stride.h() == w && stride.c() == w*h && stride.n() == w*h*c) return CUDNN_TENSOR_NCHW;
      if(stride.c() == 1 &&  stride.w() == c && stride.h() == c*w && stride.n() == c*w*h) return CUDNN_TENSOR_NHWC;
      UCUDNN_ERROR_EXIT("Invalid tensor format: ["
			+ std::to_string(n) + ","
			+ std::to_string(c) + ","
			+ std::to_string(h) + ","
			+ std::to_string(w) + "], stride: ["
			+ std::to_string(stride.n()) + ","
			+ std::to_string(stride.c()) + ","
			+ std::to_string(stride.h()) + ","
			+ std::to_string(stride.w()) + "]");
    }

    cudnnTensorFormat_t getXFormat() const { return getTensorFormat(n_, c_in_,  h_in_,  w_in_,  xStride_); }
    cudnnTensorFormat_t getYFormat() const { return getTensorFormat(n_, c_out_, h_out_, w_out_, yStride_); }
    cudnnTensorFormat_t getWFormat() const { return filterFormat_; }

    size_t getXOffset(const int idx) const { return ((size_t) idx)*xStride_.n()*getDataTypeSizeInBytes(xType_); }
    size_t getYOffset(const int idx) const { return ((size_t) idx)*yStride_.n()*getDataTypeSizeInBytes(yType_); }
    int getDefaultBatchSize() const { return n_; }

    void setScalingFactor(double *factor, const double val) const;
    double getScalingFactor(const double *factor) const;

    void setComputeType(const cudnnDataType_t computeType) { computeType_ = computeType; }
    cudnnDataType_t getComputeType() const { return computeType_; }

    size_t hash() const {
      std::vector<std::string> ary;
      // only using NCHW for simplicity
      ary.push_back(std::to_string(n_));
      ary.push_back(std::to_string(c_in_));
      ary.push_back(std::to_string(c_out_));
      ary.push_back(std::to_string(h_in_));
      ary.push_back(std::to_string(w_in_));
      std::string str = "";
      for(auto s : ary)
	str += "," + s;
      return std::hash<std::string>()(str);
    }

    std::string databaseHash() const {
      return toString();
    }

    inline bool operator==(const ConvParam &rhs) const {
      return true
	&& xType_       == rhs.xType_
	&& yType_       == rhs.yType_
	&& filterType_  == rhs.filterType_
	&& computeType_ == rhs.computeType_

#if CUDNN_HAS_MATHTYPE
	&& mathType_    == rhs.mathType_
#endif

	&& xStride_      == rhs.xStride_
	&& yStride_      == rhs.yStride_
	&& filterFormat_ == rhs.filterFormat_

	&& n_  == rhs.n_
	&& c_in_  == rhs.c_in_
	&& c_out_ == rhs.c_out_
	&& h_in_  == rhs.h_in_
	&& h_out_ == rhs.h_out_
	&& w_in_  == rhs.w_in_
	&& w_out_ == rhs.w_out_

	&& h_kernel_ == rhs.h_kernel_
	&& w_kernel_ == rhs.w_kernel_

	&& h_pad_      == rhs.h_pad_
	&& w_pad_      == rhs.w_pad_
	&& h_stride_   == rhs.h_stride_
	&& w_stride_   == rhs.w_stride_
	&& h_dilation_ == rhs.h_dilation_
	&& w_dilation_ == rhs.w_dilation_

	&& convMode_ == rhs.convMode_;
    }

    inline bool operator!=(const ConvParam &rhs) const { return !(this->operator==(rhs)); }

    std::string toString() const {
      std::string s = "";
      s += "[" + std::to_string(n_)
	+"," + std::to_string(c_in_)
	+"," + std::to_string(h_in_)
	+"," + std::to_string(w_in_) + "]";
      s += " -> ";
      s += "[" + std::to_string(n_)
	+"," + std::to_string(c_out_)
	+"," + std::to_string(h_out_)
	+"," + std::to_string(w_out_) + "]";
      s += " ";
      s += "(" + std::to_string(h_kernel_) + "x" + std::to_string(w_kernel_)
	+((h_stride_ > 1 || w_stride_ > 1)
	  ? ", stride " + std::to_string(h_stride_) + "x" + std::to_string(w_stride_)
	  : "")
	+((h_pad_ >= 1 || w_pad_ >= 1)
	  ? ", pad " + std::to_string(h_pad_) + "x" + std::to_string(w_pad_)
	  : "")
	+(", "+getTensorFormatName(getXFormat()))
	+(", "+getTensorFormatName(getYFormat()))
	+(", "+getTensorFormatName(filterFormat_))
    +(convMode_ == CUDNN_CROSS_CORRELATION ? ", CCR" : "")
    +(", "+convDataTypesToString(xType_, yType_, filterType_, computeType_))
#if CUDNN_HAS_MATHTYPE
    +(", "+mathTypeToString(mathType_))
#endif
    + ")";
    return s;
  }

private:
  cudnnDataType_t xType_, yType_, filterType_, computeType_;
#if CUDNN_HAS_MATHTYPE
  cudnnMathType_t mathType_;
#endif
  TensorStride xStride_, yStride_;
  cudnnTensorFormat_t filterFormat_;
  int c_in_, c_out_, h_in_, h_out_, w_in_, w_out_;
  int n_; // batch size specified at its constructor
  int h_kernel_, w_kernel_;
  int h_pad_, w_pad_, h_stride_, w_stride_, h_dilation_, w_dilation_;
  cudnnConvolutionMode_t convMode_;
};

}

namespace std {
  template <>
  struct hash<vcudnn::ConvParam> {
    size_t operator()(const vcudnn::ConvParam &convParam) const {
      return convParam.hash();
    }
  };
};

#endif
