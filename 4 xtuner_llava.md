![](img4md/head.jpg#id=sC8cs&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

特别喜欢里面的一句话，给大语言模型加上眼睛。<br />感觉数据才是最关键的，数据才是王道；<br />这个文档里面，讲述的是关于一张图片制作的图片+复杂文本数据。如果是大量的图片数据呢？<br />是要微调更长的时间。

<a name="c098d547"></a>
# 1. XTuner多模态训练与测试

在本节课中，我们将学习使用XTuner微调多模态LLM的内容，本部分需要的GPU资源为24GB 30% 的 A100。<br />这是学完本节内容后的多模态LLM性能效果展示：<br />**Finetune前的多模态LLM(InternLM_Chat_1.8B_llava)：只会给图像打标题**<br />![](img4md/ft_before.png#id=dmJJJ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

**Finetune后的多模态LLM(InternLM_Chat_1.8B_llava)：会根据图像回答问题了**<br />![](img4md/ft_after.png#id=BeKzn&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
<a name="c08b462d"></a>
## 下面，让我们正式开始今天的课程吧~

<a name="7944ddb0"></a>
## 1.1. 给LLM装上电子眼：多模态LLM原理简介
<a name="cf95929c"></a>
### 1.1.1. 文本单模态

![](https://cdn.nlark.com/yuque/__mermaid_v3/50d0924d26470f5c2a56588dd47d274d.svg#lake_card_v2=eyJ0eXBlIjoibWVybWFpZCIsImNvZGUiOiJmbG93Y2hhcnQgTFJcbmFb6L6T5YWl5paH5pysXSAtLi0gQSjmlofmnKxFbWJlZGRpbmfmqKHlnospIC0uLT4gYlsv5paH5pys5ZCR6YePL11cbmIgLS0-IGMoKEwgTCBNKSlcbmMgLS0-IGRb6L6T5Ye65paH5pysXSIsInVybCI6Imh0dHBzOi8vY2RuLm5sYXJrLmNvbS95dXF1ZS9fX21lcm1haWRfdjMvNTBkMDkyNGQyNjQ3MGY1YzJhNTY1ODhkZDQ3ZDI3NGQuc3ZnIiwiaWQiOiIwYjhiOTFjMCIsImNhcmQiOiJkaWFncmFtIn0=)
<a name="3135ec23"></a>
### 1.1.2. 文本+图像多模态
![](https://cdn.nlark.com/yuque/__mermaid_v3/39fdbdaf577e06e725712c29d575da19.svg#lake_card_v2=eyJ0eXBlIjoibWVybWFpZCIsImNvZGUiOiJmbG93Y2hhcnQgTFJcbmFb6L6T5YWl5paH5pysXSAtLi0gQSjmlofmnKxFbWJlZGRpbmfmqKHlnospIC0uLT4gYlsv5paH5pys5ZCR6YePL11cbmIgLS0-IGMoKEwgTCBNKSlcbmMgLS0-IGRb6L6T5Ye65paH5pysXVxuc3ViZ3JhcGggXCIgXCJcbmZb6L6T5YWl5Zu-5YOPXSAtLi0gRihcIkltYWdlIFByb2plY3RvclwiKVxuRiAtLi0-IGdbL-WbvuWDj-WQkemHjy9dXG5lbmRcbmcgLS0-IGMiLCJ1cmwiOiJodHRwczovL2Nkbi5ubGFyay5jb20veXVxdWUvX19tZXJtYWlkX3YzLzM5ZmRiZGFmNTc3ZTA2ZTcyNTcxMmMyOWQ1NzVkYTE5LnN2ZyIsImlkIjoiOTQyZTJjNTEiLCJjYXJkIjoiZGlhZ3JhbSJ9)
<a name="7e17afed"></a>
## 1.2. 什么型号的电子眼：LLaVA方案简介
[Haotian Liu等](https://arxiv.org/abs/2304.08485)使用GPT-4V对图像数据生成描述，以此构建出大量`<question text><image> -- <answer text>`的数据对。利用这些数据对，配合文本单模态LLM，训练出一个Image Projector。<br />所使用的`文本单模型LLM`和训练出来的`Image Projector`，统称为`LLaVA模型`。

<a name="0c412f25"></a>
### 1.2.1. LLaVA训练阶段示意图

![](https://cdn.nlark.com/yuque/__mermaid_v3/9f53377284d01966c4e8dbfd718cec4a.svg#lake_card_v2=eyJ0eXBlIjoibWVybWFpZCIsImNvZGUiOiJmbG93Y2hhcnQgVEI7XG5zdWJncmFwaCDorq3nu4PpmLbmrrVcbmFbKFwi5paH5pysK-WbvuWDjzxicj7pl67nrZTlr7k8YnI-KOiuree7g-aVsOaNrilcIildIC0tPiBie-aYvuWNoX1cbmMoKOaWh-acrOWNleaooeaAgTxicj5MTE0pKSAtLT4gYlxuYiAtLT4gZChbSW1hZ2U8YnI-UHJvamVjdG9yXSlcbmVuZCIsInVybCI6Imh0dHBzOi8vY2RuLm5sYXJrLmNvbS95dXF1ZS9fX21lcm1haWRfdjMvOWY1MzM3NzI4NGQwMTk2NmM0ZThkYmZkNzE4Y2VjNGEuc3ZnIiwiaWQiOiJlYzA2NGExYyIsImNhcmQiOiJkaWFncmFtIn0=)
<a name="0afc1c1b"></a>
### 1.2.2. LLaVA测试阶段示意图

![](https://cdn.nlark.com/yuque/__mermaid_v3/5482c00df85b4657de126a617e860e90.svg#lake_card_v2=eyJ0eXBlIjoibWVybWFpZCIsImNvZGUiOiJmbG93Y2hhcnQgVEI7XG5zdWJncmFwaCDmtYvor5XpmLbmrrVcbmEoW0ltYWdlPGJyPlByb2plY3Rvcl0pIC0tPiBie-aYvuWNoX1cbmMoKOaWh-acrOWNleaooeaAgTxicj5MTE0pKSAtLT4gYlxuZVvovpPlhaXlm77lg49dIC0tPiBiXG5iIC0tPiBkW-i-k-WHuuaWh-acrF1cbmVuZCIsInVybCI6Imh0dHBzOi8vY2RuLm5sYXJrLmNvbS95dXF1ZS9fX21lcm1haWRfdjMvNTQ4MmMwMGRmODViNDY1N2RlMTI2YTYxN2U4NjBlOTAuc3ZnIiwiaWQiOiI5MGYyY2Y1YiIsImNhcmQiOiJkaWFncmFtIn0=)
> Image Projector的训练和测试，有点类似之前我们讲过的LoRA微调方案。

二者都是在已有LLM的基础上，用新的数据训练一个新的小文件。<br />只不过，LLM套上LoRA之后，有了新的灵魂（角色）；而LLM套上Image Projector之后，才有了眼睛。

<a name="b2f77ed6"></a>
## 1.3. 快速上手
<a name="f1f5f9ad"></a>
### 1.3.1. 环境准备
<a name="674b2dca"></a>
#### 1.3.1.1. 开发机准备
首先我们需要前往 [InternStudio](https://studio.intern-ai.org.cn/) 中创建一个开发机进行使用。然后在进入界面后首先选择开发机。<br />首先，打开 `Intern Studio` 界面，点击 创建开发机 配置开发机系统。<br />之后我们填写 `开发机名称` 后，点击 选择镜像 使用 `Cuda11.7-conda` 镜像，然后在资源配置中，使用 `30% A100 * 1` 的选项，然后立即创建开发机器。<br />点击 `进入开发机` 选项。<br />最后我们点击 `Terminal` 进入终端界面即可开始操作！
<a name="68121b5d"></a>
#### 1.3.1.2. XTuner安装
```bash
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 的环境：
# pytorch    2.0.1   py3.10_cuda11.7_cudnn8.5.0_0

cd ~ && studio-conda xtuner0.1.17
# 如果你是在其他平台：
# conda create --name xtuner0.1.17 python=3.10 -y

# 激活环境
conda activate xtuner0.1.17
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir -p /root/xtuner0117 && cd /root/xtuner0117

# 拉取 0.1.17 的版本源码
git clone -b v0.1.17  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.15 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd /root/xtuner0117/xtuner

# 从源码安装 XTuner
pip install -e '.[all]' && cd ~
```

> 假如速度太慢可以 `Ctrl + C` 退出后换成 `pip install -e '.[all]' -i https://mirrors.aliyun.com/pypi/simple/`


假如在这一过程中没有出现任何的报错的话，那也就意味着我们成功安装好支持 XTuner 所运行的环境啦。其实对于很多的初学者而言，安装好环境意味着成功了一大半！
<a name="a77b98e7"></a>
### 1.3.2. 概述
> 在本节中，我们将 **自己构造 **`**<question text><image>--<answer text>**`** 数据对，基于InternLM2_Chat_1.8B这个文本单模态模型，使用LLaVA方案，训练一个给InternLM2_Chat_1.8B使用的Image Projector文件。**


LLaVA方案中，给LLM增加视觉能力的过程，即是训练Image Projector文件的过程。<br />该过程分为2个阶段：Pretrain和Finetune。
![](https://cdn.nlark.com/yuque/__mermaid_v3/9664b3856e5061a7d2dce0810c4709af.svg#lake_card_v2=eyJ0eXBlIjoibWVybWFpZCIsImNvZGUiOiJmbG93Y2hhcnQgTFI7XG4gICAgc3ViZ3JhcGggUHJldHJhaW7pmLbmrrVcbiAgICBhWyhcIuWbvuWDjzxicj4rPGJyPuagh-mimCjnn63mlofmnKwpXCIpXSAtLT4gYnvmmL7ljaF9XG4gICAgYygoXCLmlofmnKzljZXmqKHmgIFMTE08YnI-KEludGVybkxNMl9DaGF0XzEuOEIpXCIpKSAtLT4gYlxuICAgIGIgLS0-IGQoKFByZXRyYWluZWQ8YnI-TExhVkEpKVxuICAgIGVuZFxuXG4gICAgc3ViZ3JhcGggRmluZXR1bmXpmLbmrrVcbiAgICBmWyhcIuWbvuWDjzxicj4rPGJyPuWkjeadguWvueivneaWh-acrFwiKV0gLS0-IGd75pi-5Y2hfVxuICAgIGQgLS0-IGdcbiAgICBnIC0tPiBpKChGaW5ldHVuZWQ8YnI-TExhVkEpKVxuICAgIGVuZCIsInVybCI6Imh0dHBzOi8vY2RuLm5sYXJrLmNvbS95dXF1ZS9fX21lcm1haWRfdjMvOTY2NGIzODU2ZTUwNjFhN2QyZGNlMDgxMGM0NzA5YWYuc3ZnIiwiaWQiOiIwMTQ0NzQxOSIsImNhcmQiOiJkaWFncmFtIn0=)
<a name="992b92fa"></a>
### 1.3.3. Pretrain阶段

在Pretrain阶段，我们会使用大量的`图片+简单文本（caption, 即图片标题）`数据对，使LLM理解图像中的**普遍特征**。即，对大量的图片进行**粗看**。

Pretrain阶段训练完成后，此时的模型已经有视觉能力了！但是由于训练数据中都是图片+图片标题，所以此时的模型虽然有视觉能力，但无论用户问它什么，它都只会回答输入图片的标题。即，**此时的模型只会给输入图像“写标题”**。

> Pretrain阶段相当于是开发LLM时预训练工作，对硬件要求非常高，有8卡的学有余力同学可以自行尝试。详见[XTuner-LLaVA](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset)和[LLaVA](https://llava-vl.github.io/)。
>  


```bash
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2

NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
```
 



在本次实战营中，我们已经为大家提供了Pretrain阶段的产物——`iter_2181.pth`文件。它就是幼稚园阶段的Image Projector！大家带着`iter_2181.pth`文件继续进入下一阶段进行Finetune即可。

<a name="112fd486"></a>
### 1.3.4. Finetune阶段

在Finetune阶段，我们会使用`图片+复杂文本`数据对，来对Pretrain得到的Image Projector即iter_2181.pth进行进一步的训练。

<a name="9ea008a1"></a>
#### 1.3.4.1. 训练数据构建
<a name="198d07fe"></a>
##### 1.3.4.1.1. 格式
```json
[
    {
        "id": "随便什么字符串",
        "image": "图片文件的相对位置。相对谁？相对你后面config文件里指定的image_folder参数的路径。",
        "conversation": [
            {
                "from": "human",
                "value": "<image>\n第1个问题。"
            },
            {
                "from": "gpt",
                "value": "第1个回答"
            },
            {
                "from": "human",
                "value": "第2个问题。"
            },
            {
                "from": "gpt",
                "value": "第2个回答"
            },
            # ......
            {
                "from": "human",
                "value": "第n个问题。"
            },
            {
                "from": "gpt",
                "value": "第n个回答"
            },
        ]
    },

    # 下面是第2组训练数据了。

    {
        "id": "随便什么字符串",
        "image": "图片文件的相对位置。相对谁？相对你后面config文件里指定的image_folder参数的路径。",
        "conversation": [
            {
                "from": "human",
                "value": "<image>\n第1个问题。"
            },
            # ......
            {
                "from": "gpt",
                "value": "第n个回答"
            }
        ]
    }
]
```

> 注意：每组训练数据的第1个来自human的问题前，要加上图片占位符，即`<image>`


示例 
```json
[
  {
    "id": "000000033471",
    "image": "coco/train2017/000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  },
  {
    "id": "000000052846",
    "image": "coco/train2017/000000052846.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhere is the cat positioned in the image?"
      },
      {
        "from": "gpt",
        "value": "The cat is positioned on top of the back of the couch in the living room."
      }
    ]
  }
 ]
```
 

<a name="a0de86d7"></a>
##### 1.3.4.1.2. 制作
我们可以效法LLaVA作者的做法，将自己的图片发送给GPT，要求其按照上述格式生成若干条问答对。
prompts ![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713253679728-eb6998be-4344-4e44-90f3-60fce7ebbdf1.png#averageHue=%23cfcfcc&clientId=ufaeaad2c-40e0-4&from=paste&height=224&id=u686f3112&originHeight=280&originWidth=567&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=184228&status=done&style=none&taskId=u3eac1b54-9b52-43cb-a208-5bee3f9d5eb&title=&width=453.6)<br />Create a dataset for me, following this format.
```json
[
  {
    "id": "<random_number_string>",
    "image": "test_img/oph.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe this image."
      },
      {
        "from": "gpt",
        "value": "<answer1>"
      },
      {
        "from": "human",
        "value": "<question2>"
      },
      {
        "from": "gpt",
        "value": "<answer2>"
      },
      {
        "from": "human",
        "value": "<question3>"
      },
      {
        "from": "gpt",
        "value": "<answer3>"
      }
    ]
  }
]
```
 <br />The questions and answers, please generate for me, based on the image I sent to you. Thes questions should be from the shallow to the deep, and the answers should be as detailed and correct as possible. The questions and answers should be stick to the contents in the image itself, like objects, peoples, equipment, environment, purpose, color, attitude, etc. 5 question and answer pairs.<br /> 

为了方便大家跟随课程，针对这张示例图片的问答对数据（repeat_data.json），大家按照下面的脚本运行就可以生成啦~（重复200次）

```bash
cd ~ && git clone https://github.com/InternLM/tutorial -b camp2 && conda activate xtuner0.1.17 && cd tutorial

python /root/tutorial/xtuner/llava/llava_data/repeat.py \
  -i /root/tutorial/xtuner/llava/llava_data/unique_data.json \
  -o /root/tutorial/xtuner/llava/llava_data/repeated_data.json \
  -n 200
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713254445790-2712bcba-0510-45eb-a672-ea69e5baea03.png#averageHue=%23faf9f9&clientId=ufaeaad2c-40e0-4&from=paste&height=351&id=uf2753a02&originHeight=439&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=41163&status=done&style=none&taskId=u2555fd2a-b3a3-4beb-bc52-d07303b4cb1&title=&width=688)
<a name="80f585a8"></a>
#### 1.3.4.2. 准备配置文件

> 如果你懒到不想自己改配置文件，或者怎么改都失败。我们准备了一个fool_config文件在仓库里。运行：


```python
cp /root/tutorial/xtuner/llava/llava_data/internlm2_chat_1_8b_llava_tutorial_fool_config.py /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py
```

<a name="783723f3"></a>
##### 1.3.4.2.1. 创建配置文件

```bash
# 查询xtuner内置配置文件
xtuner list-cfg -p llava_internlm2_chat_1_8b

# 拷贝配置文件到当前目录
xtuner copy-cfg \
  llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune \
  /root/tutorial/xtuner/llava
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713254668377-05692df8-4947-4ce2-b478-6d765f317bf4.png#averageHue=%23eeeeee&clientId=ufaeaad2c-40e0-4&from=paste&height=177&id=u507f91bb&originHeight=221&originWidth=730&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13346&status=done&style=none&taskId=ua5852679-0c40-4fcc-8e1e-28a2dd7b4aa&title=&width=584)

当前你的`/root/tutorial/xtuner/llava/`目录下的文件结构应该是这样：

```bash
|-- llava_data
|   |-- repeat.py
|   |-- repeated_data.json
|   |-- test_img
|   |   `-- oph.jpg
|   `-- unique_data.json
`-- llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py
```

<a name="c7ce6cab"></a>
##### 1.3.4.2.2. 修改配置文件

修改`llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py`文件中的：

- pretrained_pth
- llm_name_or_path
- visual_encoder_name_or_path
- data_root
- data_path
- image_folder

```diff
# Model
- llm_name_or_path = 'internlm/internlm2-chat-1_8b'
+ llm_name_or_path = '/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b'
- visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
+ visual_encoder_name_or_path = '/root/share/new_models/openai/clip-vit-large-patch14-336'

# Specify the pretrained pth
- pretrained_pth = './work_dirs/llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain/iter_2181.pth'  # noqa: E501
+ pretrained_pth = '/root/share/new_models/xtuner/iter_2181.pth'

# Data
- data_root = './data/llava_data/'
+ data_root = '/root/tutorial/xtuner/llava/llava_data/'
- data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
+ data_path = data_root + 'repeated_data.json'
- image_folder = data_root + 'llava_images'
+ image_folder = data_root

# Scheduler & Optimizer
- batch_size = 16  # per_device
+ batch_size = 1  # per_device


# evaluation_inputs
- evaluation_inputs = ['请描述一下这张图片','Please describe this picture']
+ evaluation_inputs = ['Please describe this picture','What is the equipment in the image?']
```

<a name="2e3566e2"></a>
#### 1.3.4.3. 开始Finetune

```bash
cd /root/tutorial/xtuner/llava/
xtuner train /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py --deepspeed deepspeed_zero2
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713255723221-05cc0ff6-8b31-4b99-9aad-1faa09009d09.png#averageHue=%23d8d8d8&clientId=ufaeaad2c-40e0-4&from=paste&height=314&id=u1149836a&originHeight=392&originWidth=1044&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15401&status=done&style=none&taskId=u58910721-47d0-4bed-8bf0-03d80e9e31c&title=&width=835.2)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713256541491-579bf430-9c4a-4f0a-b2e8-e1c63b00273b.png#averageHue=%23efefef&clientId=ud154694c-ca72-4&from=paste&height=400&id=ue64906d7&originHeight=500&originWidth=1041&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30066&status=done&style=none&taskId=u70dce81b-d929-4ff2-9c62-765737fab77&title=&width=832.8)
<a name="d5069506"></a>
### 1.3.5. 对比Finetune前后的性能差异

<a name="094c16ce"></a>
#### 1.3.5.1. Finetune前
> 即：**加载 1.8B 和 Pretrain阶段产物(iter_2181) 到显存。**

```bash
# 解决小bug
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# pth转huggingface
xtuner convert pth_to_hf \
  llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain \
  /root/share/new_models/xtuner/iter_2181.pth \
  /root/tutorial/xtuner/llava/llava_data/iter_2181_hf

# 启动！
xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
  --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
  --llava /root/tutorial/xtuner/llava/llava_data/iter_2181_hf \
  --prompt-template internlm2_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

> Q1: Describe this image.<br />Q2: What is the equipment in the image?


![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713258978231-353bbd9c-5446-4a65-8b1d-7a5179b60ad8.png#averageHue=%23f4f4f4&clientId=u74eb78e3-c3d5-4&from=paste&height=176&id=u5f03ab0e&originHeight=220&originWidth=875&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10220&status=done&style=none&taskId=u06ea828f-b407-4f69-9dcd-c488fff3413&title=&width=700)<br />啊这；

<a name="40a5d4e9"></a>
#### 1.3.5.2. Finetune后

> 即：**加载 1.8B 和 Fintune阶段产物 到显存。**


```bash
# 解决小bug
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# pth转huggingface
xtuner convert pth_to_hf \
  /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py \
  /root/tutorial/xtuner/llava/work_dirs/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy/iter_1200.pth \
  /root/tutorial/xtuner/llava/llava_data/iter_1200_hf

# 启动！
xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
  --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
  --llava /root/tutorial/xtuner/llava/llava_data/iter_1200_hf \
  --prompt-template internlm2_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

> Q1: Describe this image.<br />Q2: What is the equipment in the image?

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1713260416545-1a3c9205-5787-4965-8adb-81a43b646823.png#averageHue=%23efefef&clientId=u74eb78e3-c3d5-4&from=paste&height=221&id=u31f4c2b3&originHeight=276&originWidth=963&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16488&status=done&style=none&taskId=u26ed82c4-44ad-46e4-88d6-f7ef7d34ba6&title=&width=770.4)<br />Finetune前后效果对比：


