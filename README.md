# inferentia-workshop-cn


## 0-Prerequisite

AWS帐号，可以开启EC2。


## 1-Setup Neuron Environment

1. Launch EC2 instance
   - AMI：**Deep Learning AMI (Ubuntu 18.04) Version 46.0** 
   - Instance Type: **inf1.6xlarge**
   - EBS：**150GB**
2. 激活环境，更新Neuron SDK

对于pytorch：

```shell
source activate aws_neuron_pytorch_p36
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade torch-neuron neuron-cc[tensorflow] torchvision
```

对于TensorFlow：

```shell
source activate aws_neuron_tensorflow_p36
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade tensorflow-neuron tensorboard-neuron neuron-cc
```



## 2-TensorFlow Resnet 50 model for image classification

### 2.1 功能验证

首先我们通过 TensorFlow下载一个预训练的Resnet50模型，并下载一个示例图片用于推理。

图片下载：

```shell
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
```

将如下代码保存为 **infer_resnet50_cpu.py**：

```python
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Instantiate Keras ResNet50 model
keras.backend.set_learning_phase(0)
tf.keras.backend.set_image_data_format('channels_last')
model = ResNet50(weights='imagenet')

# Export SavedModel
model_dir = 'resnet50'
shutil.rmtree(model_dir, ignore_errors=True)

tf.saved_model.simple_save(
    session            = keras.backend.get_session(),
    export_dir         = model_dir,
    inputs             = {'input': model.inputs[0]},
    outputs            = {'output': model.outputs[0]})

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = preprocess_input(img_arr2)

# Run inference, Display results
preds = model.predict(img_arr3)
print(decode_predictions(preds, top=5)[0])
```

执行该脚本，将下载预训练的图像分类模型并保存到**resnet50**目录以及如下输出：

```shell
[('n02123045', 'tabby', 0.68324643), ('n02127052', 'lynx', 0.12829497), ('n02123159', 'tiger_cat', 0.089623705), ('n02124075', 'Egyptian_cat', 0.06437764), ('n02128757', 'snow_leopard', 0.009918912)]
```

接下来，我们将使用Neuron SDK对上一步的模型进行编译，将如下代码保存为**compile_resnet50.py**：

```python
import shutil
import tensorflow.neuron as tfn

model_dir = 'resnet50'

# Prepare export directory (old one removed)
compiled_model_dir = 'resnet50_neuron'
shutil.rmtree(compiled_model_dir, ignore_errors=True)

# Compile using Neuron
tfn.saved_model.compile(model_dir, compiled_model_dir)
```

执行该代码，将对上一步下载的模型进行编译，并将编译后的模型保存到**resnet50_neuron**目录下，同时得到如下输出：

```shell
...
Compiler status PASS
INFO:tensorflow:Number of operations in TensorFlow session: 4638
INFO:tensorflow:Number of operations after tf.neuron optimizations: 876
INFO:tensorflow:Number of operations placed on Neuron runtime: 874
INFO:tensorflow:Successfully converted resnet50 to resnet50_neuron
```

可见，模型原来有4638个op，编译后（算子融合，计算图优化）有876个op，其中874个op是可以跑在Inferentia中，剩余2个将以框架的方式跑在CPU上，显然越多的op跑在inferentia上面，获得的加速越多。

接下来，我们来加载该编译后的模型，并对同一张图片进行推理，**infer_resnet50_neuron.py**脚本如下：

```python
import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load model
compiled_model_dir = 'resnet50_neuron'
predictor_inferentia = tf.contrib.predictor.from_saved_model(compiled_model_dir)

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = preprocess_input(img_arr2)

# Run inference, Display results
model_feed_dict={'input': img_arr3}
infa_rslts = predictor_inferentia(model_feed_dict)
print(decode_predictions(infa_rslts["output"], top=5)[0])
```

执行该脚本，结果如下：

```shell
[('n02123045', 'tabby', 0.68817204), ('n02127052', 'lynx', 0.12701613), ('n02123159', 'tiger_cat', 0.08736559), ('n02124075', 'Egyptian_cat', 0.063844085), ('n02128757', 'snow_leopard', 0.009240591)]
```

可见，推理结果和CPU推理（基于框架）的结果一致。

至此，我们已经完成Neuron的基本使用。

### 2.2 性能验证

接下来，我们来看下inferentia的性能：

首先看下，跑1000次Inference的结果：

保存脚本**infer_resnet50_perf.py**如下：

```python
import osimport timeimport numpy as npimport tensorflow as tffrom tensorflow.keras.preprocessing import imagefrom tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions# added for utilizing 4 neuron coresos.environ['NEURONCORE_GROUP_SIZES'] = "4x1"# Load modelsmodel_dir = 'resnet50'predictor_cpu = tf.contrib.predictor.from_saved_model(model_dir)compiled_model_dir = 'resnet50_neuron'predictor_inferentia = tf.contrib.predictor.from_saved_model(compiled_model_dir)# Create input from imageimg_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))img_arr = image.img_to_array(img_sgl)img_arr2 = np.expand_dims(img_arr, axis=0)img_arr3 = preprocess_input(img_arr2)model_feed_dict={'input': img_arr3}# warmupinfa_rslts = predictor_cpu(model_feed_dict)infa_rslts = predictor_inferentia(model_feed_dict)num_inferences = 2000# Run inference on CPUs, Display resultsstart = time.time()for _ in range(num_inferences):    infa_rslts = predictor_cpu(model_feed_dict)elapsed_time = time.time() - startprint('By CPU         - num_inferences:{:>6}[images], elapsed_time:{:6.2f}[sec], Throughput:{:8.2f}[images/sec]'.format(num_inferences, elapsed_time, num_inferences / elapsed_time))# Run inference on Neuron Cores, Display resultsstart = time.time()for _ in range(num_inferences):    infa_rslts = predictor_inferentia(model_feed_dict)elapsed_time = time.time() - startprint('By Neuron Core - num_inferences:{:>6}[images], elapsed_time:{:6.2f}[sec], Throughput:{:8.2f}[images/sec]'.format(num_inferences, elapsed_time, num_inferences / elapsed_time))
```

执行结果如下：

```python
By CPU         - num_inferences:  2000[images], elapsed_time: 60.97[sec], Throughput:   32.80[images/sec]By Neuron Core - num_inferences:  2000[images], elapsed_time:  7.76[sec], Throughput:  257.59[images/sec]
```

在如上脚本执行过程中，我们可以新开一个terminal，查看Neuron Core的使用率：

```shell
neuron-top
```

结果如下：

```shell
neuron-top - 10:11:40Models: 4 loaded, 4 running. NeuronCores: 4 used.0000:00:1c.0 Utilizations: NC0 0.00%, NC1 0.00%, NC2 0.00%, NC3 0.00%, Model ID   Device    NeuronCore%   Device Mem   Host Mem   Model Name10012      nd0:nc3   18.00            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c78073310011      nd0:nc2   18.00            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c78073310010      nd0:nc1   18.00            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c78073310009      nd0:nc0   18.00            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c780733
```

可见NeuronCore的使用率并不高，接下来我们通过Python多线程的方式加大并发以提高NeuronCore的使用率。

使用如下脚本**infer_resnet50_perf2.py**：

```python
import osimport timeimport numpy as npimport tensorflow as tffrom tensorflow.keras.preprocessing import imagefrom tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictionsfrom concurrent import futures# added for utilizing 4 neuron coresos.environ['NEURONCORE_GROUP_SIZES'] = '4x1'# Load modelsmodel_dir = 'resnet50'predictor_cpu = tf.contrib.predictor.from_saved_model(model_dir)compiled_model_dir = 'resnet50_neuron'predictor_inferentia = tf.contrib.predictor.from_saved_model(compiled_model_dir)# Create input from imageimg_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))img_arr = image.img_to_array(img_sgl)img_arr2 = np.expand_dims(img_arr, axis=0)img_arr3 = preprocess_input(img_arr2)model_feed_dict={'input': img_arr3}# warmupinfa_rslts = predictor_cpu(model_feed_dict)infa_rslts = predictor_inferentia(model_feed_dict)num_inferences = 3000# Run inference on CPUs, Display resultsstart = time.time()with futures.ThreadPoolExecutor(8) as exe:    fut_list = []    for _ in range (num_inferences):        fut = exe.submit(predictor_cpu, model_feed_dict)        fut_list.append(fut)    for fut in fut_list:        infa_rslts = fut.result()elapsed_time = time.time() - startprint('By CPU         - num_inferences:{:>6}[images], elapsed_time:{:6.2f}[sec], Throughput:{:8.2f}[images/sec]'.format(num_inferences, elapsed_time, num_inferences / elapsed_time))# Run inference on Neuron Cores, Display resultsstart = time.time()with futures.ThreadPoolExecutor(16) as exe:    fut_list = []    for _ in range (num_inferences):        fut = exe.submit(predictor_inferentia, model_feed_dict)        fut_list.append(fut)    for fut in fut_list:        infa_rslts = fut.result()elapsed_time = time.time() - startprint('By Neuron Core - num_inferences:{:>6}[images], elapsed_time:{:6.2f}[sec], Throughput:{:8.2f}[images/sec]'.format(num_inferences, elapsed_time, num_inferences / elapsed_time))
```

输出结果如下：

```shell
By CPU         - num_inferences:  1000[images], elapsed_time: 13.13[sec], Throughput:   76.15[images/sec]By Neuron Core - num_inferences:  1000[images], elapsed_time:  1.26[sec], Throughput:  794.77[images/sec]
```

同时在运行该脚本的时候在另一个Terminal窗口下查看NeuronCore的使用率：

```shell
neuron-top - 10:16:24Models: 4 loaded, 4 running. NeuronCores: 4 used.0000:00:1c.0 Utilizations: NC0 0.00%, NC1 0.00%, NC2 0.00%, NC3 0.00%, Model ID   Device    NeuronCore%   Device Mem   Host Mem   Model Name10012      nd0:nc3   99.77            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c78073310011      nd0:nc2   99.68            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c78073310010      nd0:nc1   99.89            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c78073310009      nd0:nc0   99.98            50 MB         1 MB    p/tmpvdz0l2ob/neuron_op_d6f098c01c780733
```

可见利用率基本打满，并且此时的throughput高达794 images/sec.

### 2.2 模型编译与多batch size尝试：

前面 2.1中我们讨论了batch size为1的时候，我们知道通常把batch size调大可以获得更高的资源利用率和吞吐，接下来我们看下改变batch size带来的结果。

[Neuron Batch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/technotes/neuroncore-batching.html)是提高Inferentia吞吐的重要方法。

Neuron是预先编译器，也就是说在编译时需要指定input tensor shape（含batch size），并且一旦编译完成，使用编译后的模型推理时，对于input tensor shape要求与编译时指定的一致。

脚本如下**compile_resnet50_batch.py**：

```python
import shutilimport tensorflow.neuron as tfnmodel_dir = 'resnet50'for batch_size in [1, 2, 4, 8, 16]:# Prepare export directory (old one removed)    compiled_model_dir = 'resnet50_neuron_batch' + str(batch_size)    shutil.rmtree(compiled_model_dir, ignore_errors=True)# Compile using Neuron    tfn.saved_model.compile(model_dir, compiled_model_dir, batch_size=batch_size, dynamic_batch_size=True)
```

在该脚本中，我们一口气编译了5个模型，batch size分别为1，2，4，8，16，执行该脚本，我们将获得5个目录，存储不同batch size的模型。

除了以上在 **tfn.saved_model.compile()** 中设定batch size方式外，我们还可以用如下方式 **new_batch_compile.py**：(该模型的input tensor Name为**input_1:0**，可以通过 *saved_model_cli* 得到该模型的input，output tensor相关信息)

```python
import shutilimport tensorflow.neuron as tfnimport numpy as npmodel_dir = 'resnet50'for batch_size in [1, 2, 4, 8, 16]:		model_feed_dict = {'input_1:0': np.zeros([batch_size, 224, 224, 3], dtype='float16')}# Prepare export directory (old one removed)		compiled_model_dir = 'resnet50_neuron_batch' + str(batch_size)		shutil.rmtree(compiled_model_dir, ignore_errors=True)# Compile using Neuron    tfn.saved_model.compile(model_dir, compiled_model_dir, feed_dict=model_feed_dict, dynamic_batch_size=True)
```

上面脚本中，我们通过设置 *feed_dict* 的参数值，在该值中指定了input tensor的shape



接下来我们看下不同batch size时推理的性能，**infer_resnet50_batch.py**：

```python
import osimport timeimport numpy as npimport tensorflow as tffrom tensorflow.keras.preprocessing import imagefrom tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictionsfrom concurrent import futures# added for utilizing 4 neuron coresos.environ['NEURONCORE_GROUP_SIZES'] = '4x1'# measure the performance per batch sizefor batch_size in [1, 2, 4, 8, 16]:    USER_BATCH_SIZE = batch_size    print("batch_size: {}, USER_BATCH_SIZE: {}". format(batch_size, USER_BATCH_SIZE))# Load model    compiled_model_dir = 'resnet50_neuron_batch' + str(batch_size)    predictor_inferentia = tf.contrib.predictor.from_saved_model(compiled_model_dir)# Create input from image    img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))    img_arr = image.img_to_array(img_sgl)    img_arr2 = np.expand_dims(img_arr, axis=0)    img_arr3 = preprocess_input(np.repeat(img_arr2, USER_BATCH_SIZE, axis=0))    model_feed_dict={'input': img_arr3}# warmup    infa_rslts = predictor_inferentia(model_feed_dict)     num_loops = 1000    num_inferences = num_loops * USER_BATCH_SIZE# Run inference on Neuron Cores, Display results    start = time.time()    with futures.ThreadPoolExecutor(8) as exe:        fut_list = []        for _ in range (num_loops):            fut = exe.submit(predictor_inferentia, model_feed_dict)            fut_list.append(fut)        for fut in fut_list:            infa_rslts = fut.result()    elapsed_time = time.time() - start    print('By Neuron Core - num_inferences:{:>6}[images], elapsed_time:{:6.2f}[sec], Throughput:{:8.2f}[images/sec]'.format(num_inferences, elapsed_time, num_inferences / elapsed_time))
```

执行该脚本，获得如下结果：

```shell
By Neuron Core - num_inferences:  1000[images], elapsed_time:  1.32[sec], Throughput:  757.93[images/sec]batch_size: 2, USER_BATCH_SIZE: 2By Neuron Core - num_inferences:  2000[images], elapsed_time:  1.45[sec], Throughput: 1378.71[images/sec]batch_size: 4, USER_BATCH_SIZE: 4By Neuron Core - num_inferences:  4000[images], elapsed_time:  2.69[sec], Throughput: 1489.27[images/sec]batch_size: 8, USER_BATCH_SIZE: 8By Neuron Core - num_inferences:  8000[images], elapsed_time:  5.31[sec], Throughput: 1507.61[images/sec]batch_size: 16, USER_BATCH_SIZE: 16By Neuron Core - num_inferences: 16000[images], elapsed_time: 10.97[sec], Throughput: 1459.13[images/sec]
```

可见随着batch size的提高，吞吐同样有提高，但是到一定性能后，将不在提升，即 batch size 也**不是越大越好**。

### 2.3 Pipeline尝试：

接下来我们探索一下Neuron 另一个杀手锏功能 —— [Neuron Pipeline](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/technotes/neuroncore-pipeline.html)：

Pipeline模式下，通过指定使用多个NeuronCore，可以将模型的权重分区尽可能保存到片上cache，以降低权重参数加载与交换的时间损耗，而使用多少NeuronCore通常根据如下[公式](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/technotes/neuroncore-pipeline.html)决定所需的core数：

```
neuroncore-pipeline-cores = 4 * round( number-of-weights-in-model/(2 * 10**7) )
```

对于ResNet50：46,159,168，计算出来8个core最合适，需要至少inf1.6xlarge机器

创建如下脚本**compile_resnet50_pipeline.py**：

```python
import shutilimport tensorflow.neuron as tfnmodel_dir = 'resnet50'# Prepare export directory (old one removed)compiled_model_dir = 'resnet50_neuron_pipeline'shutil.rmtree(compiled_model_dir, ignore_errors=True)tfn.saved_model.compile(model_dir, compiled_model_dir, compiler_args=['--neuroncore-pipeline-cores', '8'])
```

执行如上脚本完成编译后，用如下推理的脚本**infer_resnet50_pipeline.py**：

```python
import osimport timeimport numpy as npimport tensorflow as tffrom tensorflow.keras.preprocessing import imagefrom tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictionsfrom concurrent import futures# added for utilizing neuron coresos.environ['NEURONCORE_GROUP_SIZES'] = '8'USER_BATCH_SIZE = 1compiled_model_dir = 'resnet50_neuron_pipeline'predictor_inferentia = tf.contrib.predictor.from_saved_model(compiled_model_dir)# Create input from imageimg_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))img_arr = image.img_to_array(img_sgl)img_arr2 = np.expand_dims(img_arr, axis=0)img_arr3 = preprocess_input(np.repeat(img_arr2, USER_BATCH_SIZE, axis=0))model_feed_dict={'input': img_arr3}# warmupinfa_rslts = predictor_inferentia(model_feed_dict)num_inferences = 8000# Run inference on Neuron Cores, Display resultsstart = time.time()with futures.ThreadPoolExecutor(8) as exe:    fut_list = []    for _ in range (num_inferences):        fut = exe.submit(predictor_inferentia, model_feed_dict)        fut_list.append(fut)    for fut in fut_list:        infa_rslts = fut.result()elapsed_time = time.time() - startprint('By Neuron Core Pipeline - num_inferences:{:>6}[images], elapsed_time:{:6.2f}[sec], Throughput:{:8.2f}[images/sec]'.format(num_inferences*USER_BATCH_SIZE, elapsed_time, num_inferences*USER_BATCH_SIZE/ elapsed_time))
```

执行该脚本输出结果如下：

```shell
By Neuron Core Pipeline - num_inferences:  8000[images], elapsed_time:  9.17[sec], Throughput:  871.95[images/sec]
```

同样您可以实验一下，修改batch size，以及NeuronCore的数量，对比下效果，此处我们不再展开，相信您已经掌握了该技能。

**注意：**

在如上推理脚本中，有处设置环境变量 *os.environ['NEURONCORE_GROUP_SIZES'] = '8'*， 此处表示将用8个NeuronCore，那么为什么是8呢？答案是前面编译的时候我们指定了要用8个NeuronCore，如果您前面改了该值，在推理脚本中记得修改该环境变量的值。

### 2.5 TensorFlow Serving

最后我们看下Neuron编译后的模型在TensorFlow Serving中的使用。

Neuron TensorFlow Serving 提供了与原生 TensorFlow Serving 一样的API，接下来我们介绍下使用方法。

执行如下命令把2.1生成的模型复制到新的目录。

```shell
mkdir -p resnet50_inf1_servecp -rf resnet50_neuron resnet50_inf1_serve/1
```

执行如下命令启动 tensorflow_model_server_neuron:（注意此处命令是 *tensorflow_model_server_neuron* 不是 *tensorflow_model_server*）

```shell
tensorflow_model_server_neuron --model_name=resnet50_inf1_serve --model_base_path=$(pwd)/resnet50_inf1_serve/ --port=8500
```

该命令执行后，可以看到如下输出，表面 model server已经启动完毕，可以接收推理请求：

```
Successfully loaded servable version {name: resnet50_inf1_serve version: 1}Running gRPC ModelServer at 0.0.0.0:8500 ...
```



接下来执行如下客户端脚本，该客户端脚本将向前面启动的model Server发送推理请求，**tfs_client.py**：

```python
import numpy as npimport grpcimport tensorflow as tffrom tensorflow.keras.preprocessing import imagefrom tensorflow.keras.applications.resnet50 import preprocess_inputfrom tensorflow.keras.applications.resnet50 import decode_predictionsfrom tensorflow_serving.apis import predict_pb2from tensorflow_serving.apis import prediction_service_pb2_grpctf.keras.backend.set_image_data_format('channels_last')if __name__ == '__main__':    channel = grpc.insecure_channel('localhost:8500')    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)    img_file = tf.keras.utils.get_file(        "./kitten_small.jpg",        "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg")    img = image.load_img(img_file, target_size=(224, 224))    img_array = preprocess_input(image.img_to_array(img)[None, ...])    request = predict_pb2.PredictRequest()    request.model_spec.name = 'resnet50_inf1_serve'    request.inputs['input'].CopyFrom(        tf.contrib.util.make_tensor_proto(img_array, shape=img_array.shape))    result = stub.Predict(request)    prediction = tf.make_ndarray(result.outputs['output'])    print(decode_predictions(prediction))
```

执行结果如下，符合预期：

```shell
[[('n02123045', 'tabby', 0.68817204), ('n02127052', 'lynx', 0.12701613), ('n02123159', 'tiger_cat', 0.08736559), ('n02124075', 'Egyptian_cat', 0.063844085), ('n02128757', 'snow_leopard', 0.009240591)]]
```



### 2.6 Neuron Tools

最后我们简单介绍几个常用的Neuron CLI

**neuron-top**：可以查看NeuronCore的使用率；

**neuron-ls**：可以查看当前机器上 Inferentia的情况；

**neuron-cli list-model**：列出当前加载到 Inferentia中的模型；

**neuron-cli list-ncg**：列出当前NeuronCore Group；

**neuron-cli reset**：重置/清空 inferentia中的模型，该命令执行后，再执行前面两个命令输出将为空。



## 3-Reference

**EC2型号**： https://aws.amazon.com/cn/ec2/instance-types/

**Neuron SDK**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html

**TensorFlow Neuron**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/index.html

**TensorFlow Neuron Compile SDK**：https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/api-compilation-python-api.html

**Neuron-TensorFlow-Serving**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/tutorial-tensorflow-serving.html

**Inf1 workshop**: https://introduction-to-inferentia.workshop.aws/

**Roadmap**: https://github.com/aws/aws-neuron-sdk/projects/2

