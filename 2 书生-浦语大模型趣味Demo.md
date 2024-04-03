![](images/logo.jpg#id=gWAUI&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
<a name="bbd3102e"></a>
## 1 **趣味 Demo 任务列表**
本节课可以让同学们实践 4 个主要内容，分别是：

- **部署 **`**InternLM2-Chat-1.8B**`** 模型进行智能对话**
- **部署实战营优秀作品 **`**八戒-Chat-1.8B**`** 模型**
- **通过 **`**InternLM2-Chat-7B**`** 运行 **`**Lagent**`** 智能体 **`**Demo**`
- **实践部署 **`**浦语·灵笔2**`** 模型**

另：这个平台是我用过最好用的平台，各方面都非常的丝滑。
<a name="e9e01057"></a>
## 2 **部署 **`**InternLM2-Chat-1.8B**`** 模型进行智能对话**
<a name="1ccd2e18"></a>
### **2.1 配置基础环境**
首先，打开 `Intern Studio` 界面，点击 创建开发机 配置开发机系统。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144047319-9f6ba8b8-8816-4eb2-a03c-136d1e663497.png#averageHue=%23e9eae2&clientId=u8e479706-f3e9-4&from=paste&height=334&id=ucf20a2b5&originHeight=418&originWidth=1108&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=39743&status=done&style=none&taskId=ufeb6a920-2aff-4872-bb7f-391ef1eefc3&title=&width=886.4)<br />填写 `开发机名称` 后，点击 选择镜像 使用 `Cuda11.7-conda` 镜像，然后在资源配置中，使用 `10% A100 * 1` 的选项，然后立即创建开发机器。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144065024-965d2519-82f1-4185-9051-7bd5e38bab37.png#averageHue=%23d2d4cc&clientId=u8e479706-f3e9-4&from=paste&height=459&id=u1c6ae6e3&originHeight=574&originWidth=1251&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=96437&status=done&style=none&taskId=u122f6769-b73c-4baa-8b3c-7a47e5d4962&title=&width=1000.8)

点击 `进入开发机` 选项。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144080717-dca7c482-1827-4259-8a1b-c94927104569.png#averageHue=%23e2ddd1&clientId=u8e479706-f3e9-4&from=paste&height=353&id=u86eb7778&originHeight=441&originWidth=1244&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=65193&status=done&style=none&taskId=u5ae58368-59e6-4987-920e-ad2f02aa077&title=&width=995.2)

**进入开发机后，在 **`**terminal**`** 中输入环境配置命令 (配置环境时间较长，需耐心等待)：**

```bash
studio-conda -o internlm-base -t demo
# 与 studio-conda 等效的配置方案
# conda create -n demo python==3.10 -y
# conda activate demo
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144093266-301c9faa-da8b-4b98-87e3-562b081b3bb9.png#averageHue=%23f9f8f7&clientId=u8e479706-f3e9-4&from=paste&height=459&id=u77216d71&originHeight=574&originWidth=1132&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=97191&status=done&style=none&taskId=u4b106013-4004-4811-9354-02741160f70&title=&width=905.6)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144111464-8ba8e213-c7cb-4be8-8d7a-c0d1f14bf2da.png#averageHue=%23faf6f6&clientId=u8e479706-f3e9-4&from=paste&height=323&id=u2e0e958f&originHeight=404&originWidth=1230&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=54335&status=done&style=none&taskId=u6773592c-081a-479d-b68f-aa12d3b0556&title=&width=984)<br />配置完成后，进入到新创建的 `conda` 环境之中：
```bash
conda activate demo
```

输入以下命令，完成环境包的安装：
```bash
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```

<a name="a06ef58f"></a>
### **2.2 下载 **`**InternLM2-Chat-1.8B**`** 模型**
按路径创建文件夹，并进入到对应文件目录中：
```bash
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```

通过左侧文件夹栏目，双击进入 `demo` 文件夹。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144137210-2a27f23d-811a-4536-9143-be14d1c1bc78.png#averageHue=%23faf8f8&clientId=u8e479706-f3e9-4&from=paste&height=256&id=uefc857d4&originHeight=320&originWidth=753&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21151&status=done&style=none&taskId=ub7438c5c-b32a-4dbb-a0e3-b9c38e3316c&title=&width=602.4)<br />双击打开 `/root/demo/download_mini.py` 文件，复制以下代码：
```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')
```

执行命令，下载模型参数文件：

```bash
python /root/demo/download_mini.py
```

<a name="0623fd7f"></a>
### **2.3 运行 cli_demo**
双击打开 `/root/demo/cli_demo.py` 文件，复制以下代码：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
```

输入命令，执行 Demo 程序：
```bash
conda activate demo
python /root/demo/cli_demo.py
```
等待模型加载完成，键入内容示例：
```
请创作一个 300 字的小故事
```

效果如下：<br />![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711874284359-df8cecbd-eb8b-4991-8d0a-6f46259f8dbb.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0#averageHue=%23e9e9e9&from=url&id=NIphO&originHeight=250&originWidth=937&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](images/img-5.png#id=a2lw2&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)当然，公主和王子的关系也可以调换，就是有些别扭。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711874401763-94a189a9-bca0-4e65-aab5-db1d5d7874fa.png#averageHue=%23eaeaea&clientId=u882cb77d-be0b-4&from=paste&height=330&id=ud03d3167&originHeight=660&originWidth=2190&originalType=binary&ratio=2&rotation=0&showTitle=false&size=247069&status=done&style=none&taskId=u4d3cc7d8-5a1e-485e-ba42-6d3d29bb1d3&title=&width=1095)

<a name="0ecb3d1c"></a>
## 3 **实战：部署实战营优秀作品 **`**八戒-Chat-1.8B**`** 模型**
<a name="b79c952f"></a>
### 3.1 **简单介绍 **`**八戒-Chat-1.8B**`**、**`**Chat-嬛嬛-1.8B**`**、**`**Mini-Horo-巧耳**`**（实战营优秀作品）**
`八戒-Chat-1.8B`、`Chat-嬛嬛-1.8B`、`Mini-Horo-巧耳` 均是在第一期实战营中运用 `InternLM2-Chat-1.8B` 模型进行微调训练的优秀成果。其中，`八戒-Chat-1.8B` 是利用《西游记》剧本中所有关于猪八戒的台词和语句以及 LLM API 生成的相关数据结果，进行全量微调得到的猪八戒聊天模型。作为 `Roleplay-with-XiYou` 子项目之一，`八戒-Chat-1.8B` 能够以较低的训练成本达到不错的角色模仿能力，同时低部署条件能够为后续工作降低算力门槛。

![](images/img-6.png#id=lyk6X&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144234823-b35a2da5-eae1-464e-b9bd-a62ffa6eca5c.png#averageHue=%23d5b89d&clientId=u4bd60cff-8c67-4&from=paste&height=236&id=ue93078ca&originHeight=295&originWidth=427&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=216008&status=done&style=none&taskId=u49292244-78ea-41be-9476-011d52b780f&title=&width=341.6)

当然，同学们也可以参考其他优秀的实战营项目，具体模型链接如下：

- **八戒-Chat-1.8B：**[**https://www.modelscope.cn/models/JimmyMa99/BaJie-Chat-mini/summary**](https://www.modelscope.cn/models/JimmyMa99/BaJie-Chat-mini/summary)
- **Chat-嬛嬛-1.8B：**[**https://openxlab.org.cn/models/detail/BYCJS/huanhuan-chat-internlm2-1_8b**](https://openxlab.org.cn/models/detail/BYCJS/huanhuan-chat-internlm2-1_8b)
- **Mini-Horo-巧耳：**[**https://openxlab.org.cn/models/detail/SaaRaaS/Horowag_Mini**](https://openxlab.org.cn/models/detail/SaaRaaS/Horowag_Mini)

🍏那么，开始实验！！！
<a name="f9f5e94a"></a>
### 3.2 **配置基础环境**
运行环境命令：
```bash
conda activate demo
```

使用 `git` 命令来获得仓库内的 Demo 文件：
```bash
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
# git clone https://github.com/InternLM/Tutorial -b camp2
cd /root/Tutorial
```
<a name="29919190"></a>
### 3.3 **下载运行 Chat-八戒 Demo**
在 `Web IDE` 中执行 `bajie_download.py`：
```bash
python /root/Tutorial/helloworld/bajie_download.py
```

待程序下载完成后，输入运行命令：
```bash
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```

待程序运行的同时，对端口环境配置本地 `PowerShell` 。使用快捷键组合 `Windows + R`（Windows 即开始菜单键）打开指令界面，并输入命令，按下回车键。（Mac 用户打开终端即可）<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144267910-b035027a-dd4e-4860-934b-58b226fc1292.png#averageHue=%23faf6f5&clientId=u4bd60cff-8c67-4&from=paste&height=425&id=u5499e765&originHeight=531&originWidth=786&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=316299&status=done&style=none&taskId=u22d7b5b6-f763-48f0-82f5-0b4941b7b1a&title=&width=628.8)

打开 PowerShell 后，先查询端口，再根据端口键入命令 （例如图中端口示例为 38374）：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144275817-9bb5daa5-3551-45c2-bcf9-d3283c044efc.png#averageHue=%239b9e98&clientId=u4bd60cff-8c67-4&from=paste&height=385&id=u1cf1f546&originHeight=481&originWidth=1064&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=64873&status=done&style=none&taskId=u6f20d964-a3dd-4a64-903d-e7fcd75eeb5&title=&width=851.2)<br />![](images/img-A.png#id=Tf5OT&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
```bash
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```

再复制下方的密码，输入到 `password` 中，直接回车：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144288660-8406747c-72d3-4315-8413-0f0b61725ac3.png#averageHue=%23a7a9a2&clientId=u4bd60cff-8c67-4&from=paste&height=404&id=u46f078bb&originHeight=505&originWidth=881&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=56360&status=done&style=none&taskId=uce09eff4-0a29-4bbb-8d44-724a3e48d5f&title=&width=704.8)<br />![](images/img-B.png#id=okAiA&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

最终保持在如下效果即可：![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711988190990-b8b09aa5-0b78-4eaa-bedf-16c123f7bff0.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_775%2Climit_0#averageHue=%23f3f3f3&from=url&id=AUCuO&originHeight=338&originWidth=775&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)

打开 [http://127.0.0.1:6006](http://127.0.0.1:6006) 后，等待加载完成即可进行对话，键入内容示例如下：

```
你好，请自我介绍
```

效果图如下：<br />![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711988312907-1db2f23c-7e0b-4da4-8b8a-8b5c1a15a550.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_860%2Climit_0%2Fresize%2Cw_860%2Climit_0#averageHue=%23f4f4f5&from=url&id=iSH0Y&originHeight=411&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](images/img-D.png#id=qnvYa&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="d6ab1b41"></a>
## 4 **实战：使用 **`**Lagent**`** 运行 **`**InternLM2-Chat-7B**`** 模型（开启 30% A100 权限后才可开启此章节）**

<a name="11378749"></a>
### 4.1 **初步介绍 Lagent 相关知识**
Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144411119-5de1f90c-6d2b-4565-9c48-448e00bb5443.png#averageHue=%23ececec&clientId=u0013df6d-b807-4&from=paste&height=441&id=u18cc2a5c&originHeight=551&originWidth=1110&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=86284&status=done&style=none&taskId=u03b3230e-fbc6-48e6-89c5-7c93a8a5b02&title=&width=888)<br />![](images/Lagent-1.png#id=My1bN&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

Lagent 的特性总结如下：

- 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。
- 接口统一，设计全面升级，提升拓展性，包括： 
   - Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
   - Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
   - Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。

<a name="8ebafb7d"></a>
### 4.2 **配置基础环境（开启 30% A100 权限后才可开启此章节）**

打开 `Intern Studio` 界面，调节配置（必须在开发机关闭的条件下进行）：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144425467-ff30695d-54ee-4403-9734-6c9703b9d7e5.png#averageHue=%23beb8a8&clientId=u0013df6d-b807-4&from=paste&height=415&id=uc0fe3e51&originHeight=519&originWidth=981&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=72310&status=done&style=none&taskId=uf2297ca9-c588-4940-8fb8-7489497339d&title=&width=784.8)

重新开启开发机，输入命令，开启 conda 环境：

```bash
conda activate demo
```

打开文件子路径
```bash
cd /root/demo
```

使用 git 命令下载 Lagent 相关的代码库：
```bash
git clone https://gitee.com/internlm/lagent.git
# git clone https://github.com/internlm/lagent.git
cd /root/demo/lagent
git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
pip install -e . # 源码安装
```

<a name="9edc0e82"></a>
### 4.3 **使用 **`**Lagent**`** 运行 **`**InternLM2-Chat-7B**`** 模型为内核的智能体**

`Intern Studio` 在 share 文件中预留了实践章节所需要的所有基础模型，包括 `InternLM2-Chat-7b` 、`InternLM2-Chat-1.8b` 等等。我们可以在后期任务中使用 `share` 文档中包含的资源，但是在本章节，为了能让大家了解各类平台使用方法，还是推荐同学们按照提示步骤进行实验。<br />打开 lagent 路径：
```bash
cd /root/demo/lagent
```

在 terminal 中输入指令，构造软链接快捷访问方式：

```bash
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```

打开 `lagent` 路径下 `examples/internlm2_agent_web_demo_hf.py` 文件，并修改对应位置 (71行左右) 代码：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144468857-e38e92e7-58d2-406e-978a-2cae7ebdecdb.png#averageHue=%23fbf9f8&clientId=u0013df6d-b807-4&from=paste&height=312&id=u8c85382f&originHeight=390&originWidth=867&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=104679&status=done&style=none&taskId=u32c7fc66-034d-4e02-8859-e79565a9dd9&title=&width=693.6)

```bash
# 其他代码...
value='/root/models/internlm2-chat-7b'
# 其他代码...
```

输入运行命令 - **点开 6006 链接后，大约需要 5 分钟完成模型加载：**

```bash
streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006
```

待程序运行的同时，对本地端口环境配置本地 `PowerShell` 。使用快捷键组合 `Windows + R`（Windows 即开始菜单键）打开指令界面，并输入命令，按下回车键。（Mac 用户打开终端即可）<br />打开 PowerShell 后，先查询端口，再根据端口键入命令 （例如图中端口示例为 38374）：
```bash
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```

再复制下方的密码，输入到 `password` 中，直接回车：

打开 [http://127.0.0.1:6006](http://127.0.0.1:6006) 后，（会有较长的加载时间）勾上数据分析，其他的选项不要选择，进行计算方面的 Demo 对话，即完成本章节实战。键入内容示例：
```
请解方程 2*X=1360 之中 X 的结果
```
![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711989797013-af377080-dd03-47e0-8a68-b96326bd3868.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0#averageHue=%23e4c064&from=url&id=YUkxw&originHeight=434&originWidth=937&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](images/img-I.png#id=tSUe9&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="961b5f07"></a>
## 5 **实战：实践部署 **`**浦语·灵笔2**`** 模型（开启 50% A100 权限后才可开启此章节）**
<a name="d4455c1e"></a>
### 5.1 **初步介绍 **`**XComposer2**`** 相关知识**
`浦语·灵笔2` 是基于 `书生·浦语2` 大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色，总结起来其具有：

- 自由指令输入的图文写作能力： `浦语·灵笔2` 可以理解自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。
- 准确的图文问题解答能力：`浦语·灵笔2` 具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。
- 杰出的综合能力： `浦语·灵笔2-7B` 基于 `书生·浦语2-7B` 模型，在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 `GPT-4V` 和 `Gemini Pro`。

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712144562274-4be0f6a4-e027-4c1e-92aa-08201c079a79.png#averageHue=%23f4f1f0&clientId=u12f8736b-2a8c-4&from=paste&height=344&id=u355c65a7&originHeight=589&originWidth=1120&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=352039&status=done&style=none&taskId=u408934a6-8815-42f5-9070-fafe647839c&title=&width=654)<br />![](images/Benchmark_radar_CN.png#id=ZIgVQ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="2729e7f6"></a>
### 5.2 **配置基础环境（开启 50% A100 权限后才可开启此章节）**
选用 `50% A100` 进行开发：<br />进入开发机，启动 `conda` 环境：
```bash
conda activate demo
# 补充环境包
pip install timm==0.4.12 sentencepiece==0.1.99 markdown2==2.4.10 xlsxwriter==3.1.2 gradio==4.13.0 modelscope==1.9.5
```
下载 **InternLM-XComposer 仓库** 相关的代码资源：
```bash
cd /root/demo
git clone https://gitee.com/internlm/InternLM-XComposer.git
# git clone https://github.com/internlm/InternLM-XComposer.git
cd /root/demo/InternLM-XComposer
git checkout f31220eddca2cf6246ee2ddf8e375a40457ff626
```
在 `terminal` 中输入指令，构造软链接快捷访问方式：
```bash
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b /root/models/internlm-xcomposer2-7b
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b /root/models/internlm-xcomposer2-vl-7b
```

<a name="544330c9"></a>
### 5.3 **图文写作实战（开启 50% A100 权限后才可开启此章节）**
继续输入指令，用于启动 `InternLM-XComposer`：
```bash
cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
--code_path /root/models/internlm-xcomposer2-7b \
--private \
--num_gpus 1 \
--port 6006
```

待程序运行的同时，参考章节 3.3 部分对端口环境配置本地 `PowerShell` 。使用快捷键组合 `Windows + R`（Windows 即开始菜单键）打开指令界面，（Mac 用户打开终端即可）并输入命令，按下回车键：<br />打开 PowerShell 后，先查询端口，再根据端口键入命令 （例如图中端口示例为 38374）：
```bash
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```

再复制下方的密码，输入到 `password` 中，直接回车：<br />最终保持在如下效果即可：<br />打开 [http://127.0.0.1:6006](http://127.0.0.1:6006) 实践效果如下图所示：<br />![](images/img-9.png#id=KyaEF&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712134954837-676baadb-24dc-44ee-a1f0-b51c928cd354.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0#averageHue=%231a212e&from=url&height=252&id=QZu2P&originHeight=424&originWidth=937&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=&width=558)<br />![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712135100397-43c4c629-da7c-42c3-bbcb-37ee1e0cf19a.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0#averageHue=%231b202c&from=url&height=290&id=ynPoX&originHeight=484&originWidth=937&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=&width=562)<br />![](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712135120264-a77e68d4-be7e-406d-b618-447750bea289.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0#averageHue=%232c303a&from=url&height=389&id=AI1X8&originHeight=646&originWidth=937&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=&width=564)
<a name="3dc764c7"></a>
### 5.4 **图片理解实战（开启 50% A100 权限后才可开启此章节）**

根据附录 6.4 的方法，关闭并重新启动一个新的 `terminal`，继续输入指令，启动 `InternLM-XComposer2-vl`：
```bash
conda activate demo

cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  \
--code_path /root/models/internlm-xcomposer2-vl-7b \
--private \
--num_gpus 1 \
--port 6006
```

打开 [http://127.0.0.1:6006](http://127.0.0.1:6006) (上传图片后) 键入内容示例如下：

```
请分析一下图中内容
```

实践效果如下图所示：<br />多个大模型的话，就需要在不同的卡进行部署。 而且，相同的模型，要部署在很多卡上。这样的话，就不怕被多线程调用+每次推理都需要比较长的时间了。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712136086597-a1872993-8a9f-467f-8849-5e68b8f2b9ec.png#averageHue=%232f3847&clientId=u80b38ab1-d1fe-4&from=paste&height=420&id=u5d8872a4&originHeight=628&originWidth=960&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=186004&status=done&style=none&taskId=ue86460fb-a516-4663-985b-0d44dc7966c&title=&width=642)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712136005609-735e2243-b5df-400a-b6b9-a46828b01fa8.png#averageHue=%231e2634&clientId=u80b38ab1-d1fe-4&from=paste&height=295&id=u8c915110&originHeight=730&originWidth=1577&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=101207&status=done&style=none&taskId=u5ecf7ee3-f439-498e-860d-0cf249466b3&title=&width=637)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1712136170659-499d29eb-ca47-414d-9b5e-7073874a2d67.png#averageHue=%23232c3b&clientId=u80b38ab1-d1fe-4&from=paste&height=256&id=u332c9bc7&originHeight=389&originWidth=926&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18985&status=done&style=none&taskId=u2645d09d-f884-47a7-af1a-03bd49309d4&title=&width=609.7999877929688)<br />![](images/img-7.png#id=FA4jf&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)


