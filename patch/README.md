You can install &mu;-cuDNN to [Caffe](https://github.com/BVLC/caffe) or [TensorFlow](https://github.com/tensorflow/tensorflow) by applying these patches as follows:
`cd /path/to/framework_repo && patch -p1 < /path/to/ucudnn_repo/patch/[framework]_[version].patch`

---

* `caffe_v1.0.patch`: A patch file to [Caffe v1.0](https://github.com/BVLC/caffe/tree/1.0)
   * commit: `eeebdab16155d34ff8f5f42137da7df4d1c7eab0`
   * A build command example:
      1. `mkdir buikd && cd build`
	  2. `cmake ..`
	  3. `make`
* `tensorflow_v1.4.1.patch`: A patch to [TensorFlow v1.4.1](https://github.com/tensorflow/tensorflow/tree/v1.4.1)
   * commit: `438604fc885208ee05f9eef2d0f2c630e1360a83`
   * A build command example:
      1. `./configure` and select preferable configurations
	  2. `bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`
	  3. `bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg`
	  4. `pip install ./tensorflow_pkg/tensorflow-*.whl`
  * The build details are described at [here](https://www.tensorflow.org/install/install_sources).
* `tensorflow_v1.6.0.patch`: A patch to [TensorFlow v1.6.0](https://github.com/tensorflow/tensorflow/tree/v1.6.0)
   * commit: `d2e24b6039433bd83478da8c8c2d6c58034be607`
