![](https://github.com/InternLM/Tutorial/assets/25839884/48108bed-1bbd-4781-9edc-ecdf7d1bca02#id=gNIes&originHeight=383&originWidth=900&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
<a name="069a3749"></a>
# 1.LMDeploy环境部署
<a name="1a39cb29"></a>
## 1.1 创建开发机
打开InternStudio平台，创建开发机。<br />填写开发机名称；选择镜像`Cuda12.2-conda`；选择`10% A100*1`GPU；点击“立即创建”。**注意请不要选择**`**Cuda11.7-conda**`**的镜像，新版本的lmdeploy会出现兼容性问题。**<br />排队等待一小段时间，点击“进入开发机”。<br />点击左上角图标，切换为终端(Terminal)模式。

<a name="b25eb7c0"></a>
## 1.2 创建conda环境
<a name="577062ea"></a>
### InternStudio开发机创建conda环境（推荐）
由于环境依赖项存在torch，下载过程可能比较缓慢。InternStudio上提供了快速创建conda环境的方法。打开命令行终端，创建一个名为`lmdeploy`的环境：
```shell
studio-conda -t lmdeploy -o pytorch-2.1.2
```
环境创建成功后，提示如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714916783358-f8bf9043-989f-47af-a810-11adaf1e8fdd.png#averageHue=%23201f1d&clientId=u9700ae3c-65a1-4&from=paste&height=232&id=u2ede85ea&originHeight=321&originWidth=800&originalType=binary&ratio=1&rotation=0&showTitle=false&size=266259&status=done&style=none&taskId=u2bd4306a-f555-41ff-8f0d-7b24439cf50&title=&width=579)

<a name="9c2496fc"></a>
### 本地环境创建conda环境
注意，如果你在上一步已经在InternStudio开发机上创建了conda环境，这一步就没必要执行了。

打开命令行终端，让我们来创建一个名为`lmdeploy`的conda环境，python版本为3.10。
```shell
conda create -n lmdeploy -y python=3.10
```


<a name="f5704781"></a>
## 1.3 安装LMDeploy
接下来，激活刚刚创建的虚拟环境。
```shell
conda activate lmdeploy
```

安装0.3.0版本的lmdeploy。
```shell
pip install lmdeploy[all]==0.3.0
```
等待安装结束就OK了！

<a name="9fb6cbde"></a>
# 2.LMDeploy模型对话(chat)

<a name="b8bc17f7"></a>
## 2.1 Huggingface与TurboMind
<a name="HuggingFace"></a>
### HuggingFace
[HuggingFace](https://huggingface.co/)是一个高速发展的社区，包括Meta、Google、Microsoft、Amazon在内的超过5000家组织机构在为HuggingFace开源社区贡献代码、数据集和模型。可以认为是一个针对深度学习模型和数据集的在线托管社区，如果你有数据集或者模型想对外分享，网盘又不太方便，就不妨托管在HuggingFace。<br />托管在HuggingFace社区的模型通常采用HuggingFace格式存储，简写为**HF格式**。<br />但是HuggingFace社区的服务器在国外，国内访问不太方便。国内可以使用阿里巴巴的[MindScope](https://www.modelscope.cn/home)社区，或者上海AI Lab搭建的[OpenXLab](https://openxlab.org.cn/home)社区，上面托管的模型也通常采用**HF格式**。

<a name="TurboMind"></a>
### TurboMind
TurboMind是LMDeploy团队开发的一款关于LLM推理的高效推理引擎，它的主要功能包括：LLaMa 结构模型的支持，continuous batch 推理模式和可扩展的 KV 缓存管理器。<br />TurboMind推理引擎仅支持推理TurboMind格式的模型。因此，TurboMind在推理HF格式的模型时，会首先自动将HF格式模型转换为TurboMind格式的模型。**该过程在新版本的LMDeploy中是自动进行的，无需用户操作。**

几个容易迷惑的点：

- TurboMind与LMDeploy的关系：LMDeploy是涵盖了LLM 任务全套轻量化、部署和服务解决方案的集成功能包，TurboMind是LMDeploy的一个推理引擎，是一个子模块。LMDeploy也可以使用pytorch作为推理引擎。
- TurboMind与TurboMind模型的关系：TurboMind是推理引擎的名字，TurboMind模型是一种模型存储格式，TurboMind引擎只能推理TurboMind格式的模型。

<a name="87e758f5"></a>
## 2.2 下载模型
本次实战营已经在开发机的共享目录中准备好了常用的预训练模型，可以运行如下命令查看：
```shell
ls /root/share/new_models/Shanghai_AI_Laboratory/
```

显示如下，每一个文件夹都对应一个预训练模型。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917224026-51c9bc0f-0d42-4284-9bbc-c7647a809acb.png#averageHue=%23253d30&clientId=u723146a6-54e6-4&from=paste&height=54&id=u8cec2e4c&originHeight=54&originWidth=838&originalType=binary&ratio=1&rotation=0&showTitle=false&size=100157&status=done&style=none&taskId=ue172178d-46e5-437b-a1dd-92c5cf2dd1f&title=&width=838)<br />以InternLM2-Chat-1.8B模型为例，从官方仓库下载模型。
<a name="56abbead"></a>
### InternStudio开发机上下载模型（推荐）
如果你是在InternStudio开发机上，可以按照如下步骤快速下载模型。<br />首先进入一个你想要存放模型的目录，本教程统一放置在Home目录。执行如下指令：

```shell
cd ~
```

然后执行如下指令由开发机的共享目录**软链接**或**拷贝**模型：
```shell
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
# cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
```

执行完如上指令后，可以运行“ls”命令。可以看到，当前目录下已经多了一个`internlm2-chat-1_8b`文件夹，即下载好的预训练模型。
```shell
ls
```

![](./imgs/2.2_2.jpg#id=b1VrU&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="95e3528f"></a>
### 由OpenXLab平台下载模型
注意，如果你在上一步已经从InternStudio开发机上下载了模型，这一步就没必要执行了。

上一步介绍的方法只适用于在InternStudio开发机上下载模型，如果是在自己电脑的开发环境上，也可以由[OpenXLab](https://openxlab.org.cn/usercenter/OpenLMLab?vtab=create&module=datasets)下载。_在开发机上还是建议通过拷贝的方式，因为从OpenXLab平台下载会占用大量带宽~_<br />首先进入一个你想要存放模型的目录，本教程统一放置在Home目录。执行如下指令：
```shell
cd ~
```
 <br />OpenXLab平台支持通过Git协议下载模型。首先安装git-lfs组件。

- 对于root用于请执行如下指令：
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt update
apt install git-lfs   
git lfs install  --system
```
 

- 对于非root用户需要加sudo，请执行如下指令：
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt update
sudo apt install git-lfs   
sudo git lfs install  --system
```
 <br />安装好git-lfs组件后，由OpenXLab平台下载InternLM2-Chat-1.8B模型：
```shell
git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b.git
```
 <br />这一步骤可能耗时较长，主要取决于网速，耐心等待即可。<br /> ![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917297874-647c2368-c7a1-4d23-a026-3388a6f9e8ad.png#averageHue=%23181818&clientId=u723146a6-54e6-4&from=paste&height=129&id=u2d034e4f&originHeight=129&originWidth=803&originalType=binary&ratio=1&rotation=0&showTitle=false&size=95316&status=done&style=none&taskId=u70aa3a2a-44a3-40be-9b01-8742eb90894&title=&width=803)<br />下载完成后，运行`ls`指令，可以看到当前目录下多了一个`internlm2-chat-1.8b`文件夹，即下载好的预训练模型。
```shell
ls
```
注意！从OpenXLab平台下载的模型文件夹命名为`1.8b`，而从InternStudio开发机直接拷贝的模型文件夹名称是`1_8b`，为了后续文档统一，这里统一命名为`1_8b`。<br /> 
```shell
mv /root/internlm2-chat-1.8b /root/internlm2-chat-1_8b
```
 

<a name="39ce0a08"></a>
## 2.3 使用Transformer库运行模型

Transformer库是Huggingface社区推出的用于运行HF模型的官方库。<br />在2.2中，我们已经下载好了InternLM2-Chat-1.8B的HF模型。下面我们先用Transformer来直接运行InternLM2-Chat-1.8B模型，后面对比一下LMDeploy的使用感受。<br />现在，让我们点击左上角的图标，打开VSCode。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917334574-b33adfaf-6e19-4d9a-9281-79134a705347.png#averageHue=%231a1f1d&clientId=u723146a6-54e6-4&from=paste&height=74&id=u7b82192e&originHeight=74&originWidth=251&originalType=binary&ratio=1&rotation=0&showTitle=false&size=13401&status=done&style=none&taskId=ube31eafc-b277-48df-92d1-3340298634e&title=&width=251)<br />在左边栏**空白区域**单击鼠标右键，点击`Open in Intergrated Terminal`。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917348001-49c4f425-0d01-4e6a-87f7-52c0a1b8baf4.png#averageHue=%231a405b&clientId=u723146a6-54e6-4&from=paste&height=423&id=uf90a1224&originHeight=574&originWidth=422&originalType=binary&ratio=1&rotation=0&showTitle=false&size=83904&status=done&style=none&taskId=uc57e7923-a904-4cb0-a921-b54987b26a8&title=&width=311)<br />![](./imgs/2.3_3.jpg#id=GHARo&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />等待片刻，打开终端。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917354291-d8338f49-f28b-4849-a2a5-b394edd807eb.png#averageHue=%23242222&clientId=u723146a6-54e6-4&from=paste&height=294&id=ueee49dd4&originHeight=399&originWidth=812&originalType=binary&ratio=1&rotation=0&showTitle=false&size=69977&status=done&style=none&taskId=u81c78201-a658-483e-8b61-32b59e26641&title=&width=599)

在终端中输入如下指令，新建`pipeline_transformer.py`。
```shell
touch /root/pipeline_transformer.py
```

回车执行指令，可以看到侧边栏多出了`pipeline_transformer.py`文件，点击打开。后文中如果要创建其他新文件，也是采取类似的操作。

![](./imgs/2.3_5.jpg#id=EScwh&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917376866-c3ceb55e-bd4e-458b-8673-94bb29c53157.png#averageHue=%23b9260e&clientId=u723146a6-54e6-4&from=paste&height=427&id=ue12b761a&originHeight=427&originWidth=761&originalType=binary&ratio=1&rotation=0&showTitle=false&size=81455&status=done&style=none&taskId=u798c616b-00b7-46e6-8a42-840b6caafe3&title=&width=761)

将以下内容复制粘贴进入`pipeline_transformer.py`。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

inp = "hello"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=[])
print("[OUTPUT]", response)

inp = "please provide three suggestions about time management"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=history)
print("[OUTPUT]", response)
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917392847-c5342fbd-7b88-4773-8e6b-7ce77c44317a.png#averageHue=%233e372e&clientId=u723146a6-54e6-4&from=paste&height=426&id=u49e510f5&originHeight=426&originWidth=739&originalType=binary&ratio=1&rotation=0&showTitle=false&size=125677&status=done&style=none&taskId=u7e4be6fa-07d7-417f-a592-cd856483fc9&title=&width=739)<br />按`Ctrl+S`键保存（Mac用户按`Command+S`）。<br />回到终端，激活conda环境。

```shell
conda activate lmdeploy
```

运行python代码：
```shell
python /root/pipeline_transformer.py
```

得到输出：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917406296-a2e3f2c4-de68-469d-83a6-e26689865f70.png#averageHue=%23323232&clientId=u723146a6-54e6-4&from=paste&height=272&id=u38e84f9f&originHeight=272&originWidth=752&originalType=binary&ratio=1&rotation=0&showTitle=false&size=123487&status=done&style=none&taskId=u6ee11cf3-b8c9-4931-99e7-562287ce4fb&title=&width=752)

记住这种感觉，一会儿体验一下LMDeploy的推理速度，感受一下对比~（手动狗头）

<a name="5dce3fbf"></a>
## 2.4 使用LMDeploy与模型对话

这一小节我们来介绍如何应用LMDeploy直接与模型进行对话。<br />首先激活创建好的conda环境：
```shell
conda activate lmdeploy
```

使用LMDeploy与模型进行对话的通用命令格式为：
```shell
lmdeploy chat [HF格式模型路径/TurboMind格式模型路径]
```

例如，您可以执行如下命令运行下载的1.8B模型：
```shell
lmdeploy chat /root/internlm2-chat-1_8b
```

下面我们就可以与InternLM2-Chat-1.8B大模型对话了。比如输入“请给我讲一个小故事吧”，然后按两下回车键。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917432164-03f13752-3efa-4245-90e1-b8881e2bc7b4.png#averageHue=%231c1c1c&clientId=u723146a6-54e6-4&from=paste&height=337&id=ue6d667f0&originHeight=337&originWidth=757&originalType=binary&ratio=1&rotation=0&showTitle=false&size=268062&status=done&style=none&taskId=u6bd0667b-ec8f-4b96-8f3b-e4ef9bb2a87&title=&width=757)

速度是不是明显比原生Transformer快呢~当然，这种感受可能不太直观，感兴趣的佬可以查看拓展部分“6.3 定量比较LMDeploy与Transformer库的推理速度”。

输入“exit”并按两下回车，可以退出对话。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917441419-ef888963-9045-4a0e-8880-4f7e972ec774.png#averageHue=%231c1b18&clientId=u723146a6-54e6-4&from=paste&height=84&id=u7ebb9f64&originHeight=84&originWidth=512&originalType=binary&ratio=1&rotation=0&showTitle=false&size=48042&status=done&style=none&taskId=udbbd433a-8543-4236-905a-db1c956cff2&title=&width=512)

**拓展内容**：有关LMDeploy的chat功能的更多参数可通过-h命令查看。

```shell
lmdeploy chat -h
```

<a name="a25864b3"></a>
# 3.LMDeploy模型量化(lite)
本部分内容主要介绍如何对模型进行量化。主要包括 KV8量化和W4A16量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。<br />正式介绍 LMDeploy 量化方案前，需要先介绍两个概念：

- 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速度。
- 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。

常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。<br />那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用**KV8量化**和**W4A16**量化。KV8量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。W4A16 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

<a name="5071ba04"></a>
## 3.1 设置最大KV Cache缓存大小

KV Cache是一种缓存技术，通过存储键值对的形式来复用计算结果，以达到提高性能和降低内存消耗的目的。在大规模训练和推理中，KV Cache可以显著减少重复计算量，从而提升模型的推理速度。理想情况下，KV Cache全部存储于显存，以加快访存速度。当显存空间不足时，也可以将KV Cache放在内存，通过缓存管理器控制将当前需要使用的数据放入显存。<br />模型在运行时，占用的显存可大致分为三部分：模型参数本身占用的显存、KV Cache占用的显存，以及中间运算结果占用的显存。LMDeploy的KV Cache管理器可以通过设置`--cache-max-entry-count`参数，控制KV缓存**占用剩余显存**的最大比例。默认的比例为0.8。<br />下面通过几个例子，来看一下调整`--cache-max-entry-count`参数的效果。首先保持不加该参数（默认0.8），运行1.8B模型。

```shell
lmdeploy chat /root/internlm2-chat-1_8b
```

与模型对话，查看右上角资源监视器中的显存占用情况。

![](./imgs/3.1_2.jpg#id=hmk6h&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917514390-3964fcca-21a2-41ee-b06e-63d263bee79a.png#averageHue=%23232529&clientId=u723146a6-54e6-4&from=paste&height=45&id=uaa9914d2&originHeight=45&originWidth=354&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10049&status=done&style=none&taskId=uab081354-4392-45c4-a5a2-d7c4fa0d98c&title=&width=354)<br />此时显存占用为7856MB。下面，改变`--cache-max-entry-count`参数，设为0.5。
```shell
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.5
```

与模型对话，再次查看右上角资源监视器中的显存占用情况。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917531607-f8c5b175-72a0-4aa2-a65c-f39874f79559.png#averageHue=%23222428&clientId=u723146a6-54e6-4&from=paste&height=47&id=u95b1be3e&originHeight=47&originWidth=387&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10702&status=done&style=none&taskId=u535b1fd1-3668-4f6b-bbb8-fa35d83e86b&title=&width=387)<br />看到显存占用明显降低，变为6608M。<br />下面来一波“极限”，把`--cache-max-entry-count`参数设置为0.01，约等于禁止KV Cache占用显存。

```shell
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.01
```
然后与模型对话，可以看到，此时显存占用仅为4560MB，代价是会降低模型推理速度。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917543348-ddc0e9d2-71da-4ca0-a119-8a0c595ef155.png#averageHue=%23232629&clientId=u723146a6-54e6-4&from=paste&height=44&id=uc1a23d54&originHeight=44&originWidth=346&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10326&status=done&style=none&taskId=ub982dff1-f8cf-46bd-bb2a-2e7ad12c5e6&title=&width=346)<br />![](./imgs/3.1_4.jpg#id=T7hh8&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="38a850a8"></a>
## 3.2 使用W4A16量化
LMDeploy使用AWQ算法，实现模型4bit权重量化。推理引擎TurboMind提供了非常高效的4bit推理cuda kernel，性能是FP16的2.4倍以上。它支持以下NVIDIA显卡：

- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm90）：40 系列

运行前，首先安装一个依赖库。
```shell
pip install einops==0.7.0
```

仅需执行一条命令，就可以完成模型量化工作。
```shell
lmdeploy lite auto_awq \
   /root/internlm2-chat-1_8b \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir /root/internlm2-chat-1_8b-4bit
```

运行时间较长，请耐心等待。量化工作结束后，新的HF模型被保存到`internlm2-chat-1_8b-4bit`目录。下面使用Chat功能运行W4A16量化后的模型。
```shell
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq
```

为了更加明显体会到W4A16的作用，我们将KV Cache比例再次调为0.01，查看显存占用情况。
```shell
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.01
```
可以看到，显存占用变为2472MB，明显降低。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917582253-8b50763d-262a-449d-aa0e-092f5add3eda.png#averageHue=%23232629&clientId=u723146a6-54e6-4&from=paste&height=46&id=uc277ac3f&originHeight=46&originWidth=343&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10029&status=done&style=none&taskId=u438c1404-e235-4409-a416-089cd9c1690&title=&width=343)

**拓展内容**：有关LMDeploy的lite功能的更多参数可通过-h命令查看。

```shell
lmdeploy lite -h
```

<a name="9325d639"></a>
# 4.LMDeploy服务(serve)
在第二章和第三章，我们都是在本地直接推理大模型，这种方式成为本地部署。在生产环境下，我们有时会将大模型封装为API接口服务，供客户端访问。<br />我们来看下面一张架构图：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917599041-157ed4dd-42ad-49c1-901d-4f8151034e41.png#averageHue=%23f5f5f5&clientId=u723146a6-54e6-4&from=paste&height=522&id=u06bb053f&originHeight=522&originWidth=838&originalType=binary&ratio=1&rotation=0&showTitle=false&size=130281&status=done&style=none&taskId=ud2bc5b16-24e5-449b-86a1-60c57296b29&title=&width=838)<br />![](./imgs/4_1.jpg#id=Ty2N1&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

我们把从架构上把整个服务流程分成下面几个模块。

- 模型推理/服务。主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。
- API Server。中间协议层，把后端推理/服务通过HTTP，gRPC或其他形式的接口，供前端调用。
- Client。可以理解为前端，与用户交互的地方。通过通过网页端/命令行去调用API接口，获取模型推理/服务。

值得说明的是，以上的划分是一个相对完整的模型，但在实际中这并不是绝对的。比如可以把“模型推理”和“API Server”合并，有的甚至是三个流程打包在一起提供服务。

<a name="3fa2639d"></a>
## 4.1 启动API服务器

通过以下命令启动API服务器，推理`internlm2-chat-1_8b`模型：
```shell
lmdeploy serve api_server \
    /root/internlm2-chat-1_8b \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```

其中，model-format、quant-policy这些参数是与第三章中量化推理模型一致的；server-name和server-port表示API服务器的服务IP与服务端口；tp参数表示并行数量（GPU数量）。<br />通过运行以上指令，我们成功启动了API服务器，请勿关闭该窗口，后面我们要新建客户端连接该服务。<br />可以通过运行一下指令，查看更多参数及使用方法：
```shell
lmdeploy serve api_server -h
```

你也可以直接打开`http://{host}:23333`查看接口的具体使用说明，如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917623649-41a1ab86-339e-4879-955d-073e42780db3.png#averageHue=%23f9fcf4&clientId=u723146a6-54e6-4&from=paste&height=440&id=ue37d727a&originHeight=440&originWidth=845&originalType=binary&ratio=1&rotation=0&showTitle=false&size=94054&status=done&style=none&taskId=u1f2b3814-ed1f-4b30-b36e-832712e0ebb&title=&width=845)<br />注意，这一步由于Server在远程服务器上，所以本地需要做一下ssh转发才能直接访问。**在你本地打开一个cmd窗口**，输入命令如下：
```shell
ssh -CNg -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的ssh端口号
```
ssh 端口号就是下面图片里的 39864，请替换为你自己的。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917640538-91d27284-6d66-4985-9c66-94ac0fc59be0.png#averageHue=%23cdd1cf&clientId=u723146a6-54e6-4&from=paste&height=625&id=u43274a74&originHeight=625&originWidth=769&originalType=binary&ratio=1&rotation=0&showTitle=false&size=139064&status=done&style=none&taskId=u9ec70671-6eca-47c1-860d-1e5f0035e13&title=&width=769)

然后打开浏览器，访问`http://127.0.0.1:23333`。

<a name="6a5d8963"></a>
## 4.2 命令行客户端连接API服务器

在“4.1”中，我们在终端里新开了一个API服务器。<br />本节中，我们要新建一个命令行客户端去连接API服务器。首先通过VS Code新建一个终端：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917660246-c45fda0e-fdd9-4cd9-9b97-e54c720dcbb5.png#averageHue=%23202e31&clientId=u723146a6-54e6-4&from=paste&height=396&id=u6b64b4dc&originHeight=396&originWidth=857&originalType=binary&ratio=1&rotation=0&showTitle=false&size=214725&status=done&style=none&taskId=u454bc682-6de2-4070-b38f-f6c1358ebd4&title=&width=857)

激活conda环境。
```shell
conda activate lmdeploy
```
运行命令行客户端：
```shell
lmdeploy serve api_client http://localhost:23333
```

运行后，可以通过命令行窗口直接与模型对话：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917687107-793f92af-9d91-49e2-8922-a88868a329a2.png#averageHue=%232f2f2f&clientId=u723146a6-54e6-4&from=paste&height=231&id=uee50709e&originHeight=231&originWidth=752&originalType=binary&ratio=1&rotation=0&showTitle=false&size=165491&status=done&style=none&taskId=u68d2941e-8b39-4fe6-8ea7-ae9b8880e34&title=&width=752)<br />![](./imgs/4.2_3.jpg#id=Idgfk&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

现在你使用的架构是这样的：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917695483-c076e85e-6279-4f6d-a838-656f08a52828.png#averageHue=%23fcfaf6&clientId=u723146a6-54e6-4&from=paste&height=429&id=u7c7e9e0d&originHeight=429&originWidth=649&originalType=binary&ratio=1&rotation=0&showTitle=false&size=163305&status=done&style=none&taskId=u7cfd05b1-a695-498f-a87c-9d6938ed413&title=&width=649)<br />![](./imgs/4.2_4.jpg#id=aAh01&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="4851c2db"></a>
## 4.3 网页客户端连接API服务器

关闭刚刚的VSCode终端，但服务器端的终端不要关闭。<br />新建一个VSCode终端，激活conda环境。
```shell
conda activate lmdeploy
```

使用Gradio作为前端，启动网页客户端。
```shell
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```
运行命令后，网页客户端启动。在电脑本地新建一个cmd终端，新开一个转发端口：
```shell
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的ssh端口号>
```

打开浏览器，访问地址`http://127.0.0.1:6006`<br />然后就可以与模型进行对话了！<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917728461-3226afe8-c71e-48df-aee1-33e7df2a4088.png#averageHue=%23fafdfa&clientId=u723146a6-54e6-4&from=paste&height=536&id=ud8d2e37a&originHeight=536&originWidth=814&originalType=binary&ratio=1&rotation=0&showTitle=false&size=79889&status=done&style=none&taskId=u328c0021-d71a-4f25-b608-77f22fbbf8d&title=&width=814)<br />![](./imgs/4.3_2.jpg#id=ZKVkC&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

现在你使用的架构是这样的：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917742614-8dec7d29-3250-4aa6-a983-f7450522ba52.png#averageHue=%23fdfaf5&clientId=u723146a6-54e6-4&from=paste&height=628&id=ue0af7dd4&originHeight=628&originWidth=690&originalType=binary&ratio=1&rotation=0&showTitle=false&size=214405&status=done&style=none&taskId=u14367b9c-1eb8-4a21-9f6f-a5b9069c10e&title=&width=690)<br />![](./imgs/4.3_3.jpg#id=Ewmmu&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="7ca75289"></a>
# 5.Python代码集成

在开发项目时，有时我们需要将大模型推理集成到Python代码里面。

<a name="7a34825e"></a>
## 5.1 Python代码集成运行1.8B模型

首先激活conda环境。
```shell
conda activate lmdeploy
```

新建Python源代码文件`pipeline.py`。
```shell
touch /root/pipeline.py
```

打开`pipeline.py`，填入以下内容。
```python
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```

> **代码解读**：\
>  
> - 第1行，引入lmdeploy的pipeline模块 \
> - 第3行，从目录“./internlm2-chat-1_8b”加载HF模型 \
> - 第4行，运行pipeline，这里采用了批处理的方式，用一个列表包含两个输入，lmdeploy同时推理两个输入，产生两个输出结果，结果返回给response \
> - 第5行，输出response


保存后运行代码文件：

```shell
python /root/pipeline.py
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917764643-094bbe06-4f6e-4e1c-b8a4-2f801289fc1c.png#averageHue=%23414141&clientId=u723146a6-54e6-4&from=paste&height=273&id=ub08e78d2&originHeight=273&originWidth=746&originalType=binary&ratio=1&rotation=0&showTitle=false&size=300591&status=done&style=none&taskId=u92527c5f-9264-4432-b01f-8dd152b672b&title=&width=746)<br />![](./imgs/5.1_1.jpg#id=jGq7z&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="cc3424a5"></a>
## 5.2 向TurboMind后端传递参数

在第3章，我们通过向lmdeploy传递附加参数，实现模型的量化推理，及设置KV Cache最大占用比例。在Python代码中，可以通过创建TurbomindEngineConfig，向lmdeploy传递参数。<br />以设置KV Cache占用比例为例，新建python文件`pipeline_kv.py`。

```shell
touch /root/pipeline_kv.py
```

打开`pipeline_kv.py`，填入如下内容：
```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 调低 k/v cache内存占比调整为总显存的 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('/root/internlm2-chat-1_8b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```

保存后运行python代码：
```shell
python /root/pipeline_kv.py
```

得到输出结果：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917781984-2744812d-d587-4ee0-88a8-3e8fcec9d1fc.png#averageHue=%23464646&clientId=u723146a6-54e6-4&from=paste&height=212&id=uc58dd062&originHeight=212&originWidth=731&originalType=binary&ratio=1&rotation=0&showTitle=false&size=234502&status=done&style=none&taskId=ua20a1fd4-f3e0-4702-b94f-f22c7e92626&title=&width=731)<br />![](./imgs/5.2_1.jpg#id=IxjbQ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="2c82b19f"></a>
# 6.拓展部分
<a name="d0071967"></a>
## 6.1 使用LMDeploy运行视觉多模态大模型llava

最新版本的LMDeploy支持了llava多模态模型，下面演示使用pipeline推理`llava-v1.6-7b`。**注意，运行本pipeline最低需要30%的InternStudio开发机，请完成基础作业后向助教申请权限。**

首先激活conda环境。
```shell
conda activate lmdeploy
```

安装llava依赖库。
```shell
pip install git+https://github.com/haotian-liu/LLaVA.git@4e2277a060da264c4f21b364c867cc622c945874
```

新建一个python文件，比如`pipeline_llava.py`。
```shell
touch /root/pipeline_llava.py
```

打开`pipeline_llava.py`，填入内容如下：
```python
from lmdeploy.vl import load_image
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

> **代码解读**： \
>  
> - 第1行引入用于载入图片的load_image函数，第2行引入了lmdeploy的pipeline模块， \
> - 第5行创建了pipeline实例 \
> - 第7行从github下载了一张关于老虎的图片，如下： <br />![](./imgs/6.1_1.jpg#id=Bp1mh&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) ![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917858594-350b4e10-9427-4644-ad44-1f671b9612f4.png#averageHue=%23758b42&clientId=u723146a6-54e6-4&from=paste&height=281&id=u59eda9e8&originHeight=281&originWidth=429&originalType=binary&ratio=1&rotation=0&showTitle=false&size=268668&status=done&style=none&taskId=u3759b037-1114-47e2-be31-e3670fdbd56&title=&width=429)
> - 第8行运行pipeline，输入提示词“describe this image”，和图片，结果返回至response \
> - 第9行输出response


保存后运行pipeline。

```shell
python /root/pipeline_llava.py
```

得到输出结果：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917871290-197aaa0f-1016-4c0a-be6f-9e25104955c6.png#averageHue=%233f3f3f&clientId=u723146a6-54e6-4&from=paste&height=291&id=ub485f7d3&originHeight=291&originWidth=822&originalType=binary&ratio=1&rotation=0&showTitle=false&size=325428&status=done&style=none&taskId=u0cfa8622-681d-4516-9808-598d246aa33&title=&width=822)

> **大意（来自百度翻译）**：一只老虎躺在草地上。老虎面对镜头，头微微向一侧倾斜，给人一种好奇或专注的表情。老虎在较浅的背景上有一种独特的深色条纹图案，这是该物种的特征。皮毛是橙色和黑色的混合，深色的条纹垂直向下延伸，浅色的皮毛出现在胸部和腹部。老虎的眼睛睁开，警觉，耳朵竖起，这表明它对周围环境很关注。背景是模糊的绿色区域，表明照片是在户外拍摄的，可能是在自然栖息地或野生动物保护区。这张图片是特写，聚焦于老虎的头部和上身，突出了老虎的特征和皮毛的纹理。照片中没有可见的文字或其他物体，照片的风格是自然的野生动物拍摄，旨在捕捉环境中的动物。


由于官方的Llava模型对中文支持性不好，因此如果使用中文提示词，可能会得到出乎意料的结果，比如将提示词改为“请描述一下这张图片”，你可能会得到类似《印度鳄鱼》的回复。

![](./imgs/6.1_3.jpg#id=HiMXD&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917886012-2b381810-f7f1-41be-9713-f07821a511e7.png#averageHue=%23464646&clientId=u723146a6-54e6-4&from=paste&height=135&id=u0bd2be23&originHeight=135&originWidth=837&originalType=binary&ratio=1&rotation=0&showTitle=false&size=155984&status=done&style=none&taskId=u4b0d10aa-8905-4ece-9128-152da598fa8&title=&width=837)

我们也可以通过Gradio来运行llava模型。新建python文件`gradio_llava.py`。

```shell
touch /root/gradio_llava.py
```

打开文件，填入以下内容：
```python
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.launch()
```

运行python程序。
```shell
python /root/gradio_llava.py
```

通过ssh转发一下7860端口。
```shell
ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p <你的ssh端口>
```

通过浏览器访问`http://127.0.0.1:7860`。<br />然后就可以使用啦~

![](./imgs/6.1_4.jpg#id=g5l3c&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917906200-93d687f1-7868-4556-99e6-0e08226db33b.png#averageHue=%23b9bb8c&clientId=u723146a6-54e6-4&from=paste&height=417&id=u9e2c69af&originHeight=417&originWidth=830&originalType=binary&ratio=1&rotation=0&showTitle=false&size=159248&status=done&style=none&taskId=u26daccc3-dcb9-477c-8610-b4fc4c9c6df&title=&width=830)

<a name="3a708df8"></a>
## 6.2 使用LMDeploy运行第三方大模型

LMDeploy不仅支持运行InternLM系列大模型，还支持其他第三方大模型。支持的模型列表如下：

| Model | Size |
| --- | --- |
| Llama | 7B - 65B |
| Llama2 | 7B - 70B |
| InternLM | 7B - 20B |
| InternLM2 | 7B - 20B |
| InternLM-XComposer | 7B |
| QWen | 7B - 72B |
| QWen-VL | 7B |
| QWen1.5 | 0.5B - 72B |
| QWen1.5-MoE | A2.7B |
| Baichuan | 7B - 13B |
| Baichuan2 | 7B - 13B |
| Code Llama | 7B - 34B |
| ChatGLM2 | 6B |
| Falcon | 7B - 180B |
| YI | 6B - 34B |
| Mistral | 7B |
| DeepSeek-MoE | 16B |
| DeepSeek-VL | 7B |
| Mixtral | 8x7B |
| Gemma | 2B-7B |
| Dbrx | 132B |


可以从Modelscope，OpenXLab下载相应的HF模型，下载好HF模型，下面的步骤就和使用LMDeploy运行InternLM2一样啦~

<a name="3f1e8461"></a>
## 6.3 定量比较LMDeploy与Transformer库的推理速度差异

为了直观感受LMDeploy与Transformer库推理速度的差异，让我们来编写一个速度测试脚本。测试环境是30%的InternStudio开发机。<br />先来测试一波Transformer库推理Internlm2-chat-1.8b的速度，新建python文件，命名为`benchmark_transformer.py`，填入以下内容：

```python
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

# warmup
inp = "hello"
for i in range(5):
    print("Warm up...[{}/5]".format(i+1))
    response, history = model.chat(tokenizer, inp, history=[])

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response, history = model.chat(tokenizer, inp, history=history)
    total_words += len(response)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```

运行python脚本：
```shell
python benchmark_transformer.py
```

得到运行结果：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917928819-3af87da5-5a5d-4531-a127-4464508e7239.png#averageHue=%232c2c2c&clientId=u723146a6-54e6-4&from=paste&height=260&id=u2218cdb5&originHeight=260&originWidth=837&originalType=binary&ratio=1&rotation=0&showTitle=false&size=148815&status=done&style=none&taskId=ub8da9efa-5bdb-440b-9656-f079d7e0aac&title=&width=837)

可以看到，Transformer库的推理速度约为78.675 words/s，注意单位是words/s，不是token/s，word和token在数量上可以近似认为成线性关系。<br />下面来测试一下LMDeploy的推理速度，新建python文件`benchmark_lmdeploy.py`，填入以下内容：

```python
import datetime
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')

# warmup
inp = "hello"
for i in range(5):
    print("Warm up...[{}/5]".format(i+1))
    response = pipe([inp])

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response = pipe([inp])
    total_words += len(response[0].text)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```

运行脚本：
```shell
python benchmark_lmdeploy.py
```

得到运行结果：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1714917944738-b797a9a3-0582-437f-b353-a935383712dc.png#averageHue=%23232220&clientId=u723146a6-54e6-4&from=paste&height=173&id=u518bade7&originHeight=173&originWidth=623&originalType=binary&ratio=1&rotation=0&showTitle=false&size=21021&status=done&style=none&taskId=ub85cc1b3-431c-4509-a598-4845f5b02d5&title=&width=623)<br />![](./imgs/6.3_2.jpg#id=uJ8V7&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />可以看到，LMDeploy的推理速度约为473.690 words/s，是Transformer库的6倍。
<a name="22bf1248"></a>
# 课后作业

作业请查看[homework.md](./homework.md)。
