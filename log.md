## 1. 环境配置
#### 1.1 cuda cudnn pytorch torchvision
```
conda install pytorch torchvision cuda90 cudnn=6.0.21
```
> 和自己电脑上的cuda和cudnn版本对应，不然不能使用pytorch
> pytorch的版本为0.3
```
# 测试
>>import torch
>>torch.cuda.is_available()
true
```

#### 1.2 theano0.9版本不兼容numpy
> AttributeError: ('The following error happened while compiling the node', GpuCAReduce{add}{1}(<CudaNdarrayType(float32, vector)>), '\n', "module 'numpy.core.multiarray' has no attribute '_get_ndarray_c_version'")
> 换用theano1.0.4解决

#### 1.3 theano版本1.0.4 theano.sandbox.cuda已经被去除
> https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29
> 使用theano.gpuarray 代替所有的 theano.sandbox.cuda

## 2. 环境变量
1. /lib/config.py line17
> gpu0 --> cuda

2. ~/.theanorc
> device=gpu -->> device = cuda0

3. /experiments/scripts/res_gru_net.sh line18
> export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'" --->>>
> export THEANO_FLAGS="floatX=float32,device=cuda,assert_no_cpu_op='raise'"

## 3. mkl2018 bug
RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.
> ~/.bashrc 添加
> export MKL_THREADING_LAYER=GNU

## 4. gpu-out-of-memory
> gpudata_alloc: cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY: out of memory
+ allow_pre_alloc = False
+ optimizer = fast_run
+ [gpuarray]
+ preallocate = -1
> 后来发现是gpu环境不足以支撑整个的运算
> 选择减小batch_size的尺寸 24->8

## 5. LSTM
> LSTM单元是RNN隐藏状态最成功的实现之一。LSTM单元明确控制从输入到输出的流量，允许网络克服消失的梯度问题。 

> 具体来说，LSTM单元由四个部分组成：
+ 存储单元（存储单元和隐藏状态
+ 控制从输入到隐藏状态（输入门）
+ 从隐藏状态到输出的信息流的三个门 （输出门）
+ 从前一个隐藏状态到当前隐藏状态（忘记门）。