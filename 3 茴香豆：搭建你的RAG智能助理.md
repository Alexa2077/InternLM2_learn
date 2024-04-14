![](imgs/title.jpg#id=LeYh0&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />感觉RAG就相当于，构建一个外部知识库，外部数据库。 然后大模型在接受到提问之后，然后然后将检索到的文档与原始问题一起作为prompt输入到llm中，生成最终回答。


<a name="62febaff"></a>
## 0 RAG 概述

RAG（Retrieval Augmented Generation）技术，通过检索与用户输入相关的信息片段，并结合**_外部知识库_**来生成更准确、更丰富的回答。解决 LLMs 在处理知识密集型任务时可能遇到的挑战, 如幻觉、知识过时和缺乏透明、可追溯的推理过程等。提供更准确的回答、降低推理成本、实现外部记忆。

![](./imgs/RAG_overview.png#id=vc0XS&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712673342048-de28ad59-44a9-4fe8-a0a6-e1cf6e9cceb4.png#averageHue=%23224e9b&clientId=u3cbe65bf-c272-4&from=paste&height=400&id=u1c082be9&originHeight=525&originWidth=887&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=218644&status=done&style=none&taskId=u0a6e1488-773f-4288-93e5-feeb87f71ae&title=&width=676.6000366210938)

RAG 能够让基础模型实现非参数知识更新，**无需训练就可以掌握新领域的知识**。本次课程选用的[茴香豆](https://github.com/InternLM/HuixiangDou)应用，就应用了 RAG 技术，可以快速、高效的搭建自己的知识领域助手。

<a name="83493a78"></a>
### RAG 效果比对
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712673447236-21d6f302-f691-4f91-bc13-e94502962af7.png#averageHue=%23f1f2ec&clientId=u3cbe65bf-c272-4&from=paste&height=182&id=uf2badcbd&originHeight=227&originWidth=908&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=283058&status=done&style=none&taskId=uacea15b9-c6fe-4e5b-8f05-e72e498891c&title=&width=726.4)<br />如图所示，由于茴香豆是一款比较新的应用， `InternLM2-Chat-7B` 训练数据库中并没有收录到它的相关信息。左图中关于 huixiangdou 的 3 轮问答均未给出准确的答案。右图未对 `InternLM2-Chat-7B` 进行任何增训的情况下，通过 RAG 技术实现的新增知识问答。


<a name="104cd8de"></a>
## 1 环境配置
<a name="ef2f14f0"></a>
### 1.1 配置基础环境

这里以在 [Intern Studio](https://studio.intern-ai.org.cn/) 服务器上部署**茴香豆**为例。<br />首先，打开 `Intern Studio` 界面，点击 **_创建开发机_** 配置开发机系统。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712673911123-33a78b90-e7c8-444f-b091-7cde5d1e4d93.png#averageHue=%23dfe5dd&clientId=u3cbe65bf-c272-4&from=paste&height=263&id=ue1e0a936&originHeight=329&originWidth=731&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20437&status=done&style=none&taskId=u662d55d8-0fbd-4a75-afd1-758d5192a9f&title=&width=584.8)<br />![](../helloworld/images/img-1.png#id=NkXjg&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

填写 `开发机名称` 后，点击 选择镜像 使用 `Cuda11.7-conda` 镜像，然后在资源配置中，使用 `30% A100 * 1` 的选项，然后立即创建开发机器。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712673917700-a8b65133-adea-46a4-92e3-f2b11fde64a0.png#averageHue=%23f7f6f8&clientId=u3cbe65bf-c272-4&from=paste&height=235&id=u40c589b7&originHeight=294&originWidth=880&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43442&status=done&style=none&taskId=u77c1c3ab-1d7f-40bf-af4f-570684635e8&title=&width=704)<br />![](imgs/30gpu.png#id=D0ZeP&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

点击 `进入开发机` 选项。

![](../helloworld/images/img-3.png#id=RfDbW&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712673923013-891e2964-2e65-4a31-8c9e-eb7e88dded48.png#averageHue=%23e4dfd3&clientId=u3cbe65bf-c272-4&from=paste&height=205&id=ud938118b&originHeight=256&originWidth=852&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=32407&status=done&style=none&taskId=ud8274119-3083-408c-b644-104af2fcb23&title=&width=681.6)

进入开发机后，从官方环境复制运行 InternLM 的基础环境，命名为 `InternLM2_Huixiangdou`，在命令行模式下运行：
```bash
studio-conda -o internlm-base -t InternLM2_Huixiangdou
```

复制完成后，在本地查看环境。
```bash
conda env list
```
结果如下所示。
```bash
# conda environments:
#
base                  *  /root/.conda
InternLM2_Huixiangdou                 /root/.conda/envs/InternLM2_Huixiangdou
```
运行 **_conda_** 命令，激活 `InternLM2_Huixiangdou`  **_python_** 虚拟环境:
```bash
conda activate InternLM2_Huixiangdou
```
环境激活后，命令行左边会显示当前（也就是 `InternLM2_Huixiangdou`）的环境名称，如下图所示:<br />后续教程所有操作都需要在该环境下进行，重启开发机或打开新命令行后要重新激活环境。

<a name="fc4ccf3f"></a>
### 1.2 下载基础文件

复制茴香豆所需模型文件，为了减少下载和避免 **HuggingFace** 登录问题，所有作业和教程涉及的模型都已经存放在 `Intern Studio` 开发机共享文件中。本教程选用 **InternLM2-Chat-7B** 作为基础模型。
```bash
# 创建模型文件夹
cd /root && mkdir models

# 复制BCE模型
ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1

# 复制大模型参数（下面的模型，根据作业进度和任务进行**选择一个**就行）
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```

<a name="d86b113c"></a>
### 1.3 下载安装茴香豆
安装茴香豆运行所需依赖。
```bash
# 安装 python 依赖
# pip install -r requirements.txt

pip install protobuf==4.25.3 accelerate==0.28.0 aiohttp==3.9.3 auto-gptq==0.7.1 bcembedding==0.1.3 beautifulsoup4==4.8.2 einops==0.7.0 faiss-gpu==1.7.2 langchain==0.1.14 loguru==0.7.2 lxml_html_clean==0.1.0 openai==1.16.1 openpyxl==3.1.2 pandas==2.2.1 pydantic==2.6.4 pymupdf==1.24.1 python-docx==1.1.0 pytoml==0.1.21 readability-lxml==0.8.1 redis==5.0.3 requests==2.31.0 scikit-learn==1.4.1.post1 sentence_transformers==2.2.2 textract==1.6.5 tiktoken==0.6.0 transformers==4.39.3 transformers_stream_generator==0.0.5 unstructured==0.11.2

## 因为 Intern Studio 不支持对系统文件的永久修改，在 Intern Studio 安装部署的同学不建议安装 Word 依赖，后续的操作和作业不会涉及 Word 解析。
## 想要自己尝试解析 Word 文件的同学，uncomment 掉下面这行，安装解析 .doc .docx 必需的依赖
# apt update && apt -y install python-dev python libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
```

从茴香豆官方仓库下载茴香豆。
```bash
cd /root
# 下载 repo
git clone https://github.com/internlm/huixiangdou && cd huixiangdou
git checkout 447c6f7e68a1657fce1c4f7c740ea1700bde0440
```

茴香豆工具在 `Intern Studio` 开发机的安装工作结束。如果部署在自己的服务器上，参考上节课模型下载内容或本节 [3.4 配置文件解析](#34-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90) 部分内容下载模型文件。
<a name="cf80db40"></a>
## 2 使用茴香豆搭建 RAG 助手

<a name="84527af9"></a>
### 2.1 修改配置文件
用已下载模型的路径替换 `/root/huixiangdou/config.ini` 文件中的默认模型，需要修改 3 处模型地址，分别是:<br />命令行输入下面的命令，修改用于向量数据库和词嵌入的模型

```bash
sed -i '6s#.*#embedding_model_path = "/root/models/bce-embedding-base_v1"#' /root/huixiangdou/config.ini
```

用于检索的重排序模型
```bash
sed -i '7s#.*#reranker_model_path = "/root/models/bce-reranker-base_v1"#' /root/huixiangdou/config.ini
```

和本次选用的大模型
```bash
sed -i '29s#.*#local_llm_path = "/root/models/internlm2-chat-7b"#' /root/huixiangdou/config.ini
```

修改好的配置文件应该如下图所示：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674079900-d699f25d-6dd0-4119-9628-287064e3b70b.png#averageHue=%23fefdfc&clientId=u3cbe65bf-c272-4&from=paste&height=292&id=u7ceee729&originHeight=365&originWidth=908&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=78331&status=done&style=none&taskId=uedbb4a29-96cc-498b-bca2-b42be8ba933&title=&width=726.4)<br />![](imgs/model_path.png#id=jVB9M&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />配置文件具体含义和更多细节参考 [3.4 配置文件解析](#34-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90)。

<a name="afec1f2e"></a>
### 2.2 创建知识库

本示例中，使用 **InternLM** 的 **Huixiangdou** 文档作为新增知识数据检索来源，在不重新训练的情况下，打造一个 **Huixiangdou** 技术问答助手。<br />首先，下载 **Huixiangdou** 语料：
```bash
cd /root/huixiangdou && mkdir repodir

git clone https://github.com/internlm/huixiangdou --depth=1 repodir/huixiangdou
```

提取知识库特征，创建向量数据库。数据库向量化的过程应用到了 **LangChain** 的相关模块，默认嵌入和重排序模型调用的网易 **BCE 双语模型**，如果没有在 `config.ini` 文件中指定本地模型路径，茴香豆将自动从 **HuggingFace**  拉取默认模型。

除了语料知识的向量数据库，茴香豆建立接受和拒答两个向量数据库，用来在检索的过程中更加精确的判断提问的相关性，这两个数据库的来源分别是：

- 接受问题列表，希望茴香豆助手回答的示例问题 
   - 存储在 `huixiangdou/resource/good_questions.json` 中
- 拒绝问题列表，希望茴香豆助手拒答的示例问题 
   - 存储在 `huixiangdou/resource/bad_questions.json` 中
   - 其中多为技术无关的主题或闲聊
   - 如："nihui 是谁", "具体在哪些位置进行修改？", "你是谁？", "1+1"

运行下面的命令，增加茴香豆相关的问题到接受问题示例中：
```bash
cd /root/huixiangdou
mv resource/good_questions.json resource/good_questions_bk.json

echo '[
    "mmpose中怎么调用mmyolo接口",
    "mmpose实现姿态估计后怎么实现行为识别",
    "mmpose执行提取关键点命令不是分为两步吗，一步是目标检测，另一步是关键点提取，我现在目标检测这部分的代码是demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth   现在我想把这个mmdet的checkpoints换位yolo的，那么应该怎么操作",
    "在mmdetection中，如何同时加载两个数据集，两个dataloader",
    "如何将mmdetection2.28.2的retinanet配置文件改为单尺度的呢？",
    "1.MMPose_Tutorial.ipynb、inferencer_demo.py、image_demo.py、bottomup_demo.py、body3d_pose_lifter_demo.py这几个文件和topdown_demo_with_mmdet.py的区别是什么，\n2.我如果要使用mmdet是不是就只能使用topdown_demo_with_mmdet.py文件，",
    "mmpose 测试 map 一直是 0 怎么办？",
    "如何使用mmpose检测人体关键点？",
    "我使用的数据集是labelme标注的，我想知道mmpose的数据集都是什么样式的，全都是单目标的数据集标注，还是里边也有多目标然后进行标注",
    "如何生成openmmpose的c++推理脚本",
    "mmpose",
    "mmpose的目标检测阶段调用的模型，一定要是demo文件夹下的文件吗，有没有其他路径下的文件",
    "mmpose可以实现行为识别吗，如果要实现的话应该怎么做",
    "我在mmyolo的v0.6.0 (15/8/2023)更新日志里看到了他新增了支持基于 MMPose 的 YOLOX-Pose，我现在是不是只需要在mmpose/project/yolox-Pose内做出一些设置就可以，换掉demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py 改用mmyolo来进行目标检测了",
    "mac m1从源码安装的mmpose是x86_64的",
    "想请教一下mmpose有没有提供可以读取外接摄像头，做3d姿态并达到实时的项目呀？",
    "huixiangdou 是什么？",
    "使用科研仪器需要注意什么？",
    "huixiangdou 是什么？",
    "茴香豆 是什么？",
    "茴香豆 能部署到微信吗？",
    "茴香豆 怎么应用到飞书",
    "茴香豆 能部署到微信群吗？",
    "茴香豆 怎么应用到飞书群",
    "huixiangdou 能部署到微信吗？",
    "huixiangdou 怎么应用到飞书",
    "huixiangdou 能部署到微信群吗？",
    "huixiangdou 怎么应用到飞书群",
    "huixiangdou",
    "茴香豆",
    "茴香豆 有哪些应用场景",
    "huixiangdou 有什么用",
    "huixiangdou 的优势有哪些？",
    "茴香豆 已经应用的场景",
    "huixiangdou 已经应用的场景",
    "huixiangdou 怎么安装",
    "茴香豆 怎么安装",
    "茴香豆 最新版本是什么",
    "茴香豆 支持哪些大模型",
    "茴香豆 支持哪些通讯软件",
    "config.ini 文件怎么配置",
    "remote_llm_model 可以填哪些模型?"
]' > /root/huixiangdou/resource/good_questions.json
```

再创建一个测试用的问询列表，用来测试拒答流程是否起效：
```bash
cd /root/huixiangdou

echo '[
"huixiangdou 是什么？",
"你好，介绍下自己"
]' > ./test_queries.json
```

在确定好语料来源后，运行下面的命令，创建 RAG 检索过程中使用的向量数据库：
```bash
# 创建向量数据库存储目录
cd /root/huixiangdou && mkdir workdir 

# 分别向量化知识语料、接受问题和拒绝问题中后保存到 workdir
python3 -m huixiangdou.service.feature_store --sample ./test_queries.json
```

向量数据库的创建需要等待一小段时间，过程约占用 1.6G 显存。<br />完成后，**Huixiangdou** 相关的新增知识就以向量数据库的形式存储在 `workdir` 文件夹下。<br />检索过程中，**茴香豆会将输入问题与两个列表中的问题在向量空间进行相似性比较，判断该问题是否应该回答**，避免群聊过程中的问答泛滥。确定的回答的问题会利用基础模型提取关键词，在知识库中检索 `top K` 相似的 `chunk`，综合问题和检索到的 `chunk` 生成答案。

<a name="e3806c18"></a>
### 2.3 运行茴香豆知识助手

我们已经提取了知识库特征，并创建了对应的向量数据库。现在，让我们来测试一下效果：<br />命令行运行：
```bash
# 填入问题
sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py

# 运行茴香豆
cd /root/huixiangdou/
python3 -m huixiangdou.main --standalone
```

RAG 技术的优势就是**非参数化的模型调优**，这里使用的仍然是基础模型 `InternLM2-Chat-7B`， 没有任何额外数据的训练。面对同样的问题，我们的**茴香豆技术助理**能够根据我们提供的数据库生成准确的答案：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674251346-f9ed5fa1-e8be-4895-849d-6163e74cbf46.png#averageHue=%23d0d0d0&clientId=u3cbe65bf-c272-4&from=paste&height=179&id=ufa9f18a2&originHeight=224&originWidth=903&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=153867&status=done&style=none&taskId=ub92317a8-8ed2-4125-86d3-b653acd6edc&title=&width=722.4)<br />![](./imgs/huixiangdou.png#id=RIG3e&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674260468-1b4b84c6-eb4b-4df5-8931-51f415860763.png#averageHue=%23e0e0e0&clientId=u3cbe65bf-c272-4&from=paste&height=374&id=u97383330&originHeight=468&originWidth=901&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=210180&status=done&style=none&taskId=uea226efa-813c-49fa-ad9b-ab36adfe994&title=&width=720.8)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674273623-2d45ff66-d1ee-4932-96be-aa9272fb2a93.png#averageHue=%23dedede&clientId=u3cbe65bf-c272-4&from=paste&height=350&id=ue5fff7ce&originHeight=438&originWidth=899&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=201630&status=done&style=none&taskId=u72a6b99e-577d-4525-9b54-7caee29ea76&title=&width=719.2)<br />![](./imgs/install.png#id=qEQVw&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

`InternLM2-Chat-7B` 的关于 `huixiangdou` 问题的原始输出：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674304955-e8dab1e3-705d-4f7d-b07f-2e0ec16acb0b.png#averageHue=%23dcdcdc&clientId=u3cbe65bf-c272-4&from=paste&height=281&id=u7b576bb7&originHeight=351&originWidth=900&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=227520&status=done&style=none&taskId=u49633f7c-a03c-4a58-a7ff-a746d6de1b4&title=&width=720)<br />![](./imgs/internlm27b.png#id=Yd8vX&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

到此我们就完成了一个 茴香豆知识助手 的服务器端部署（基础作业）的全部内容。<br />后面可以根据自己的实际需求，学习茴香豆的进阶应用或者[阅读茴香豆文档](https://github.com/InternLM/HuixiangDou/tree/main/docs)将茴香豆链接到即时通讯软件或[打造自己的茴香豆 Web 版](https://github.com/InternLM/HuixiangDou/tree/main/web)。

<a name="90632d2f"></a>
## 3 茴香豆进阶（选做）

   ![](imgs/overall.png#id=aL2rI&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) ![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674334056-8890d775-30f1-4a58-96d2-321686e2ddf4.png#averageHue=%232c982b&clientId=u3cbe65bf-c272-4&from=paste&height=217&id=u2a06701b&originHeight=271&originWidth=156&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=22587&status=done&style=none&taskId=ub09b070f-6016-4b62-8976-0ee74612a79&title=&width=124.8)

茴香豆并非单纯的 RAG 功能实现，而是一个专门针对群聊优化的知识助手，下面介绍一些茴香豆的进阶用法。详情请阅读[技术报告](https://arxiv.org/abs/2401.08772)或观看本节课理论视频。

<a name="6e738709"></a>
### 3.1 加入网络搜索
茴香豆除了可以从本地向量数据库中检索内容进行回答，也可以加入网络的搜索结果，生成回答。<br />开启网络搜索功能需要用到 **Serper** 提供的 API：

1. 登录 [Serper](https://serper.dev/) ，注册：

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674363446-a8699a94-05c2-4c76-be7c-06347391e606.png#averageHue=%231b1e28&clientId=u3cbe65bf-c272-4&from=paste&height=276&id=ub935e11a&originHeight=345&originWidth=875&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=68016&status=done&style=none&taskId=uf59bfd84-e987-4883-b03c-6c721b532e3&title=&width=700)<br />![](imgs/serper.png#id=yLgL1&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

2. 进入 [Serper API](https://serper.dev/api-key) 界面，复制自己的 API-key：

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674370290-f92d3b9c-5bbd-478d-a7aa-6d9c62ed8eb8.png#averageHue=%231b1e29&clientId=u3cbe65bf-c272-4&from=paste&height=272&id=u5afddec8&originHeight=340&originWidth=818&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=38032&status=done&style=none&taskId=uc30ffb3d-66e4-4b88-964b-d6e95a97395&title=&width=654.4)<br />![](imgs/serper_api.png#id=JRaja&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

1. 替换 `/huixiangdou/config.ini` 中的 **_${YOUR-API-KEY}_** 为自己的API-key：

```
[web_search]
# check https://serper.dev/api-key to get a free API key
x_api_key = "${YOUR-API-KEY}"
domain_partial_order = ["openai.com", "pytorch.org", "readthedocs.io", "nvidia.com", "stackoverflow.com", "juejin.cn", "zhuanlan.zhihu.com", "www.cnblogs.com"]
save_dir = "logs/web_search_result"
```

其中 `domain_partial_order` 可以设置网络搜索的范围。

![](imgs/serper_api_key.png#id=SxNUY&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674384992-530c896c-e148-45d5-91d6-ccbf6a64762b.png#averageHue=%23fefdfc&clientId=u3cbe65bf-c272-4&from=paste&height=82&id=ufc5c9ec2&originHeight=103&originWidth=896&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=33158&status=done&style=none&taskId=u72673525-c6d7-45eb-9a8c-7200aedad2f&title=&width=716.8)

<a name="a4ad3000"></a>
### 3.2 使用远程模型

茴香豆除了可以使用本地大模型，还可以轻松的调用云端模型 API。

目前，茴香豆已经支持 `Kimi`，`GPT-4`，`Deepseek` 和 `GLM` 等常见大模型API。

想要使用远端大模型，首先修改 `/huixiangdou/config.ini` 文件中

```
enable_local = 0 # 关闭本地模型
enable_remote = 1 # 启用云端模型
```

接着，如下图所示，修改 `remote_` 相关配置，填写 API key、模型类型等参数。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674408181-aa1fb496-6a9a-45ea-85bd-8f79ee9acac6.png#averageHue=%23fefefe&clientId=u3cbe65bf-c272-4&from=paste&height=336&id=u39fc1462&originHeight=420&originWidth=887&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=108714&status=done&style=none&taskId=u5476e87e-0602-484f-bb59-68645aa141e&title=&width=709.6)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674414873-c28c9292-90d5-4756-8e0b-1d3c52f6cd1b.png#averageHue=%23fefbfa&clientId=u3cbe65bf-c272-4&from=paste&height=219&id=u901cd429&originHeight=274&originWidth=902&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=97329&status=done&style=none&taskId=u28fd9599-a61c-4ec2-9643-a54f94d7818&title=&width=721.6)<br />![](imgs/remote.png#id=YleVZ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

| 远端模型配置选项 | GPT | Kimi | Deepseek | ChatGLM | xi-api | alles-apin |
| --- | --- | --- | --- | --- | --- | --- |
| `remote_type` | gpt | kimi | deepseek | zhipuai | xi-api | alles-apin |
| `remote_llm_max_text_length`<br /> 最大值 | 192000 | 128000 | 16000 | 128000 | 192000 | - |
| `remote_llm_model` | "gpt-4-0613" | "moonshot-v1-128k" | "deepseek-chat" | "glm-4" | "gpt-4-0613" | - |


启用远程模型可以大大降低GPU显存需求，根据测试，采用远程模型的茴香豆应用，最小只需要2G内存即可。<br />需要注意的是，这里启用的远程模型，只用在问答分析和问题生成，依然需要本地嵌入、重排序模型进行特征提取。<br />也可以尝试同时开启 local 和 remote 模型，茴香豆将采用混合模型的方案，详见 [技术报告](https://arxiv.org/abs/2401.08772)，效果更好。<br />[茴香豆 Web 版](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web) 在 **OpenXLab** 上部署了混合模型的 Demo，可上传自己的语料库测试效果。

<a name="69a8664c"></a>
### 3.3 利用 Gradio 搭建网页 Demo

让我们用 **Gradio** 搭建一个自己的网页对话 Demo，来看看效果。

1. 首先，安装 **Gradio** 依赖组件：
```bash
pip install gradio==4.25.0 redis==5.0.3 flask==3.0.2 lark_oapi==1.2.4
```

2. 运行脚本，启动茴香豆对话 Demo 服务：
```bash
cd /root/huixiangdou
python3 -m tests.test_query_gradio
```

此时服务器端接口已开启。如果在本地服务器使用，直接在浏览器中输入 [127.0.0.1:7860](http://127.0.0.1:7860/) ，即可进入茴香豆对话 Demo 界面。<br />针对远程服务器，如我们的 `Intern Studio` 开发机，我们需要设置端口映射，转发端口到本地浏览器：

1. 查询开发机端口和密码（图中端口示例为 38374）：

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674460118-79804c1e-6a9f-4433-a393-46f7d2a97b6b.png#averageHue=%23969993&clientId=u3cbe65bf-c272-4&from=paste&height=298&id=ucc61fe53&originHeight=373&originWidth=771&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=38272&status=done&style=none&taskId=u26da2414-3431-4661-87b5-feeb2777901&title=&width=616.8)<br />![](imgs/check_port.png#id=nWfbG&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

2. 在本地打开命令行工具：
- Windows 使用快捷键组合 `Windows + R`（Windows 即开始菜单键）打开指令界面，并输入命令 `Powershell`，按下回车键
- Mac 用户直接找到并打开`终端`
- Ubuntu 用户使用快捷键组合 `ctrl + alt + t`

在命令行中输入如下命令，命令行会提示输入密码：
```
ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p <你的端口号>
```

3. 复制开发机密码到命令行中，按回车，建立开发机到本地到端口映射。

![](imgs/port_psw.png#id=uBzCA&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674482818-bb90cfec-6aaf-483d-baf9-ae55b13fd6ec.png#averageHue=%23111111&clientId=u3cbe65bf-c272-4&from=paste&height=96&id=ueffb5878&originHeight=120&originWidth=661&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17276&status=done&style=none&taskId=u453198bf-3fe9-4f32-a449-8a96be4c49d&title=&width=528.8)

4. 在本地浏览器中输入 [127.0.0.1:7860](http://127.0.0.1:7860/) 进入 **Gradio** 对话 Demo 界面，开始对话。

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712674490885-94fc4be1-15f6-47bb-8b83-1837bae3bcb3.png#averageHue=%232c3542&clientId=u3cbe65bf-c272-4&from=paste&height=362&id=ud69ecab1&originHeight=453&originWidth=875&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=200359&status=done&style=none&taskId=u0bcf229d-e624-4819-94f3-4633e57b3e8&title=&width=700)<br />![](imgs/gradio.png#id=h5uTB&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

如果需要更换检索的知识领域，只需要用新的语料知识重复步骤 [2.2 创建知识库](#22-%E5%88%9B%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93) 提取特征到新的向量数据库，更改 `huixiangdou/config.ini` 文件中 `work_dir = "新向量数据库路径"`；<br />或者运行：

```
python3 -m tests.test_query_gradi --work_dir <新向量数据库路径>
```

无需重新训练或微调模型，就可以轻松的让基础模型学会新领域知识，搭建一个新的问答助手。
<a name="8a42ea89"></a>
### 3.4 配置文件解析

茴香豆的配置文件位于代码主目录下，采用 `Toml` 形式，有着丰富的功能，下面将解析配置文件中重要的常用参数。
```
[feature_store]
...
reject_throttle = 0.22742061846268935
...
embedding_model_path = "/root/models/bce-embedding-base_v1"
reranker_model_path = "/root/models/bce-reranker-base_v1"
...
work_dir = "workdir"
```

`reject_throttle`: 拒答阈值，0-1，数值越大，回答的问题相关性越高。拒答分数在检索过程中通过与示例问题的相似性检索得出，高质量的问题得分高，无关、低质量的问题得分低。只有得分数大于拒答阈值的才会被视为相关问题，用于回答的生成。当闲聊或无关问题较多的环境可以适当调高。<br />`embedding_model_path` 和 `reranker_model_path`: 嵌入和重排用到的模型路径。不设置本地模型路径情况下，默认自动通过 **_Huggingface_** 下载。开始自动下载前，需要使用下列命令登录 **_Huggingface_** 账户获取权限：

```bash
huggingface-cli login
```

`work_dir`: 向量数据库路径。茴香豆安装后，可以通过切换向量数据库路径，来回答不同知识领域的问答。
```
[llm.server]
...
local_llm_path = "/root/models/internlm2-chat-1_8b"
local_llm_max_text_length = 3000
...
```

`local_llm_path`: 本地模型文件夹路径或模型名称。现支持 **书生·浦语** 和 **通义千问** 模型类型，调用 `transformers` 的 `AutoModels` 模块，除了模型路径，输入 **_Huggingface_** 上的模型名称，如_"internlm/internlm2-chat-7b"_、_"qwen/qwen-7b-chat-int8"_、_"internlm/internlm2-chat-20b"_，也可自动拉取模型文件。<br />`local_llm_max_text_length`: 模型可接受最大文本长度。

远端模型支持参考上一小节。
```
[worker]
# enable search enhancement or not
enable_sg_search = 0
save_path = "logs/work.txt"
...
```

`[worker]`: 增强搜索功能，配合 `[sg_search]` 使用。增强搜索利用知识领域的源文件建立图数据库，当模型判断问题为无关问题或回答失败时，增强搜索功能将利用 LLM 提取的关键词在该图数据库中搜索，并尝试用搜索到的内容重新生成答案。在 `config.ini` 中查看 `[sg_search]` 具体配置示例。
```
[worker.time]
start = "00:00:00"
end = "23:59:59"
has_weekday = 1
```

`[worker.time]`: 可以设置茴香豆每天的工作时间，通过 `start` 和 `end` 设定应答的起始和结束时间。<br />`has_weekday`: `= 1` 的时候，周末不应答😂（豆哥拒绝 996）。
```
[frontend]
...
```

`[fronted]`:  前端交互设置。[茴香豆代码仓库](https://github.com/InternLM/HuixiangDou/tree/main/docs) 查看具体教程。

<a name="4a9e08fb"></a>
### 3.5 文件结构

通过了解主要文件的位置和作用，可以更好的理解茴香豆的工作原理。
```bash
.
├── LICENSE
├── README.md
├── README_zh.md
├── android
├── app.py
├── config-2G.ini
├── config-advanced.ini
├── config-experience.ini
├── config.ini # 配置文件
├── docs # 教学文档
├── huixiangdou # 存放茴香豆主要代码，重点学习
├── huixiangdou-inside.md
├── logs
├── repodir # 默认存放个人数据库原始文件，用户建立
├── requirements-lark-group.txt
├── requirements.txt
├── resource
├── setup.py
├── tests # 单元测试
├── web # 存放茴香豆 Web 版代码
└── web.log
└── workdir # 默认存放茴香豆本地向量数据库，用户建立
```

```bash
./huixiangdou
├── __init__.py
├── frontend # 存放茴香豆前端与用户端和通讯软件交互代码
│   ├── __init__.py
│   ├── lark.py
│   └── lark_group.py
├── main.py # 运行主贷
├── service # 存放茴香豆后端工作流代码
│   ├── __init__.py
│   ├── config.py #
│   ├── feature_store.py # 数据嵌入、特征提取代码
│   ├── file_operation.py
│   ├── helper.py
│   ├── llm_client.py
│   ├── llm_server_hybrid.py # 混合模型代码
│   ├── retriever.py # 检索模块代码
│   ├── sg_search.py # 增强搜索，图检索代码
│   ├── web_search.py # 网页搜索代码
│   └── worker.py # 主流程代码
└── version.py
```

茴香豆工作流中用到的 **Prompt** 位于 `huixiangdou/service/worker.py` 中。可以根据业务需求尝试调整 **Prompt**，打造你独有的茴香豆知识助手。
```python
...
        # Switch languages according to the scenario.
        if self.language == 'zh':
            self.TOPIC_TEMPLATE = '告诉我这句话的主题，直接说主题不要解释：“{}”'
            self.SCORING_QUESTION_TEMPLTE = '“{}”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'  # noqa E501
            self.SCORING_RELAVANCE_TEMPLATE = '问题：“{}”\n材料：“{}”\n请仔细阅读以上内容，判断问题和材料的关联度，用0～10表示。判断标准：非常相关得 10 分；完全没关联得 0 分。直接提供得分不要解释。\n'  # noqa E501
            self.KEYWORDS_TEMPLATE = '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。搜索参数类型 string， 内容是短语或关键字，以空格分隔。\n你现在是{}交流群里的技术助手，用户问“{}”，你打算通过谷歌搜索查询相关资料，请提供用于搜索的关键字或短语，不要解释直接给出关键字或短语。'  # noqa E501
            self.SECURITY_TEMAPLTE = '判断以下句子是否涉及政治、辱骂、色情、恐暴、宗教、网络暴力、种族歧视等违禁内容，结果用 0～10 表示，不要解释直接给出得分。判断标准：涉其中任一问题直接得 10 分；完全不涉及得 0 分。直接给得分不要解释：“{}”'  # noqa E501
            self.PERPLESITY_TEMPLATE = '“question:{} answer:{}”\n阅读以上对话，answer 是否在表达自己不知道，回答越全面得分越少，用0～10表示，不要解释直接给出得分。\n判断标准：准确回答问题得 0 分；答案详尽得 1 分；知道部分答案但有不确定信息得 8 分；知道小部分答案但推荐求助其他人得 9 分；不知道任何答案直接推荐求助别人得 10 分。直接打分不要解释。'  # noqa E501
            self.SUMMARIZE_TEMPLATE = '{} \n 仔细阅读以上内容，总结得简短有力点'  # noqa E501
            # self.GENERATE_TEMPLATE = '材料：“{}”\n 问题：“{}” \n 请仔细阅读参考材料回答问题，材料可能和问题无关。如果材料和问题无关，尝试用你自己的理解来回答问题。如果无法确定答案，直接回答不知道。'  # noqa E501
            self.GENERATE_TEMPLATE = '材料：“{}”\n 问题：“{}” \n 请仔细阅读参考材料回答问题。'  # noqa E501
...
```

<a name="116e0772"></a>
## 作业

查看 [homework.md](./homework.md) 查看本节作业。
