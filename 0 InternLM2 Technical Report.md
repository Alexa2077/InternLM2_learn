

# 介绍：

大型语言模型（LLM）的发展包含几个主要阶段：预训练、监督微调（SFT）和人类反馈强化学习（RLHF）（Ouyang et al., 2022）
InternLM2 广泛详细介绍了它如何**为预训练准备文本**、**代码和长上下文数据**。
如何有效地扩展LLM的上下文长度是目前的一个研究热点，因为许多下游应用，例如检索增强生成（RAG）（Gao et al., 2023）和 agent（Xi et al., 2023），都依赖于在长上下文中。 
在长上下文预训练之后，我们利用监督微调（SFT）和来自人类反馈的强化学习（RLHF）来确保模型很好地遵循人类指令并与人类价值观保持一致。



# 2 Infrastructure:
## InternEvo
InternEvo:  我们利用 InternEvo 这一高效、轻量级的预训练框架来进行模型训练

- 减少通信开销
- 长序列训练
- 通信与计算重叠
- 容错能力
- 互动培训
## Model Structure
 **LLaMA (Touvron et al., 2023a) 基于 Transformer 架构**，用 RMSNorm (Zhang & Sennrich, 2019) 替换 LayerNorm (Ba et al., 2016) 并将激活函数设置为 SwiGLU (Shazeer, 2020)，从而改进了训练效率和性能。

自 LLaMA 揭幕以来（Touvron 等人，2023a），社区一直积极致力于增强围绕 LLaMA 架构构建的生态系统。这包括高效推理（lla，2023）和算子优化（Dao，2023）等方面的进步。
为了确保我们的模型 InternLM2 与这个完善的生态系统以及其他著名的llm，如 Falcon (Almazrouei et al., 2023)、Qwen (Bai et al., 2023a)、Baichuan (Yang et al., 2023a)、Baichuan (Yang et al., 2023) 无缝结合， 2023)、Mistral (Jiang et al., 2023)，我们选择遵循** LLaMA 的结构设计原则**。

预训练的时候，特殊设计了kqv的乘法；

# 3 Pre-train
## 3.1 预训练数据
llm的预训练受到数据的严格影响，它包括处理敏感数据、涵盖全面的知识以及平衡效率和质量。

### Text Data：
预训练数据集中的文本数据可以按来源分类为**网页、论文、专利和书籍**。为了将这些源转换为预训练数据集，我们首先将所有数据标准化为指定格式，按类型和语言对它们进行分类，并以 **JSON Lines (jsonl) 格式存储**。然后，对于所有数据，我们应用多个处理步骤，包括**基于规则的过滤、重复数据删除、安全过滤和质量过滤**。这会产生丰富、安全且高质量的文本数据集。

**数据来源**：来自网页的中英文数据分别占占总数的86.46%，为主要来源。 
**数据处理：**
**数据格式化 **：我们的网页数据主要来自**Common Crawl1**。首先，我们需要解压原始 Warc 格式文件，并使用 Trafilatura (Barbaresi, 2021) 进行 HTML 解析和主要文本提取。然后，我们使用 pycld22 库对主要文本进行语言检测和分类。最后，我们给数据分配一个唯一的标识符，并以jsonl（JSON行）格式存储，得到Format数据。
**基于规则的阶段**:从互联网上随机提取的网页数据通常包含大量低质量数据，例如解析错误、格式错误和非自然语言文本。常见的做法是设计基于规则的正则化和过滤方法来修改和过滤数据.
**重复数据删除 :**互联网上存在大量重复文本，这会对模型训练产生负面影响。因此，我们采用**基于局部敏感哈希（LSH）的方法**对数据进行模糊去重。
**安全过滤**:互联网上充斥着有毒和色情内容，使用这些内容进行模型训练可能会对性能产生负面影响，并增加生成不安全内容的可能性。
**质量过滤:**与书籍、论文和专利等来源相比，互联网来源的数据包含大量低质量内容,导致提取的文本难以阅读且缺乏逻辑连贯性。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711631860922-69104a96-8a4e-41c1-93a4-4e60e90c6751.png#averageHue=%23c9eef3&clientId=u798b757c-8123-4&from=paste&height=267&id=u96824f40&originHeight=334&originWidth=816&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=40692&status=done&style=none&taskId=u68d95d42-abf6-4ab3-a4a0-cbbf6775d92&title=&width=652.8)


### Code Data:


**数据来源分布:** 我们从各种来源收集数据，包括直接从 GitHub 爬取、公共数据集以及与编码和编程相关的在线资源，如问答论坛、教程站点和 API 文档，统计数据如图 4 所示.高质量的数据将具有更高的采样权重，并且可以在预训练阶段进行多次训练迭代。中等质量的数据具有正常的采样权重，通常训练一次。低质量数据被排除在外
**格式清理**:所有数据转换为统一的markdown格式。
**重复数据删除: **重复代码数据删除与处理自然语言类似，只是标记化会影响超参数的选择。例如，Python 示例使用两个空格、四个空格或制表符来表示缩进。
**质量过滤: **质量是llm研究中预训练的一个关键但模糊的方面，主要是因为很难量化其对规模方面模型性能的影响。我们采用了混合、多阶段的过滤过程，包括基于规则和模型的评分器。基于规则的评分器是启发式的且多种多样，尽管我们发现代码风格不是可靠的质量指标，并且可能会将太多代码错误地分类为低质量。对于基于模型的评分，我们评估了几个骨干模型，并使用大约 50,000 个标记样本对其进行训练。然而，我们观察到，不同语言的评分者评估和人类判断之间的相关性有所不同，并且扩大训练集并没有显着提高评分者的准确性。因此，我们只对模型预测与人工注释验证集上的人工评估非常吻合的语言采用基于模型的评分。
**依赖性排序 **：InternLM2 的训练上下文窗口已扩展到 32,000 个令牌，允许利用代码存储库的整个上下文。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711632355783-c76fc67c-e91b-4c0f-a9f4-d0ab8129e988.png#averageHue=%23fefdfd&clientId=u798b757c-8123-4&from=paste&height=305&id=u102b3122&originHeight=408&originWidth=869&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=37934&status=done&style=none&taskId=ud9fd6580-d3f9-4865-a878-3b36a6ee617&title=&width=649.2000122070312)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711632542219-11d33ef1-2eb1-4636-b11f-36781fe312d4.png#averageHue=%23fdfcfc&clientId=u798b757c-8123-4&from=paste&height=289&id=ue34e3d4c&originHeight=361&originWidth=729&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43946&status=done&style=none&taskId=udd95c9f7-836c-4e81-9f40-db300e239d5&title=&width=583.2)

### Long Context Data

处理超长上下文（> 32K tokens）的能力是llm研究中越来越受欢迎的主题，它扩大和促进了应用程序，例如书籍摘要、支持长期对话以及支持处理涉及复杂推理步骤的任务。

**数据过滤管道**我们的数据处理管道旨在过滤掉低质量的长文本数据。它包括三个阶段： a) 长度选择，一个基于规则的过滤器，选择超过 32K 字节的数据样本； b) 统计过滤器，利用统计特征来识别和删除异常数据； c) 困惑度过滤器，利用困惑度的差异来评估文本片段之间的连贯性，过滤掉具有分散注意力的上下文的样本。请注意，为长上下文训练选择的所有数据都是标准预训练语料库的子集，这意味着在预训练期间将至少学习两次长上下文数据。
**统计过滤器**我们采用各种词汇和语言特征来构建我们的统计过滤器。不符合既定规则的数据样本将被排除在预训练语料库之外。
**Perplexity 过滤器** Perplexity 通常被视为文本序列 P(X) 概率的估计器；
**阈值选择**选择合适的阈值是数据过滤过程中具有挑战性但又必不可少的部分；

## 3.2 预训练：

Tokenization：
我们选择使用 GPT-4 的标记化方法，因为它在压缩各种文本内容方面具有卓越的效率。我们的主要参考是 cl100k 词汇 4，它主要包含英语和编程语言标记，总共 100,256 个条目，少量包含不到 3,000 个中文标记。

## 3.3 Pre-training Phases
用于预训练1.8B、7B和20B模型的token总数从2.0T到2.6T不等，预训练过程由三个不同的阶段组成。
第一阶段，我们使用长度不超过4k的预训练语料库。
在第二阶段，我们包含了50%的长度不超过32k的预训练语料库。
在第三阶段，我们利用了特定于能力的增强数据。在每个阶段，我们混合了英文、中文和代码的数据。



# 4 Alignment

预培训阶段使llm具备解决各种任务所需的基础能力和知识。我们进一步对llm进行微调，充分激发他们的能力，引导llm成为有用且无害的人工智能助手。

这个阶段通常称为“**对齐”**，通常包含两个阶段：**监督微调（SFT）和来自人类反馈的强化学习（RLHF），**然后我们提出了 CONditionalOnLine RLHF，它应用了一种新颖的条件奖励模型，可以协调不同类型的人类偏好（例如，多步推理准确性、有用性、无害性），并进行三轮在线 RLHF 以减少奖励hacking行为.


## 4.1 SFT:

在监督微调（SFT）阶段，我们使用包含 1000 万条指令数据实例的数据集，这些数据实例经过筛选以确保它们的有用性和无害性。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711633149253-81640649-5975-486f-983d-e4ad383fbdec.png#averageHue=%23fdfcfb&clientId=u798b757c-8123-4&from=paste&height=271&id=uaf088228&originHeight=339&originWidth=581&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43887&status=done&style=none&taskId=uf93b01f9-6553-4ca8-9236-f38a337ccf3&title=&width=464.8)


## 4.2 COOL Reinforcement Learning from Human Feedback

**人类反馈强化学习（RLHF）**（Christiano et al., 2017；Ouyang et al., 2022）是大型语言模型领域的一种创新方法。通过结合人类反馈，RLHF 创建了奖励模型，作为人类偏好的代理，从而为法学硕士提供奖励信号，以通过使用近端策略优化 (PPO) 进行学习（Schulman 等人，2017）。这种方法使模型能够更好地理解和执行通过传统方法难以定义的任务。

- **问题**：尽管RLHF取得了一定的成就，但其实际应用中仍存在一些问题。首先是偏好冲突。例如，在开发对话系统时，我们期望它提供有用的信息（有帮助），同时不产生有害或不适当的内容（无害）。然而，这两种偏好在实践中往往无法同时满足，**因为在某些情况下提供有用的信息可能涉及敏感或高风险内容. （看是对好人有用，还是对坏人有用。）**
- **问题：**现有的 RLHF 方法（Touvron et al., 2023b; Dai et al., 2023; Wu et al., 2023）通常依赖于多个偏好模型进行评分，这也在训练管道中引入了更多模型，从而增加了计算成本并减慢了速度训练速度。其次，RLHF 面临奖励黑客攻击的问题，特别是当策略随着规模的增加而变得更加强大时（Manheim & Garrabrant，2018；Gao et al.，2022），模型可能会**学会通过捷径“欺骗”奖励系统为了获得高分**，而不是真正学习预期的行为。这导致模型以意想不到的方式最大化奖励，显着影响法学硕士的有效性和可靠性。

**解决方法：**为了解决这些问题，我们提出了 Conditional OnLine RLHF (COOL RLHF)。 COOL RLHF首先引入了**条件奖励机制来协调不同的偏好**，该机制允许奖励模型根据特定条件动态地将注意力分配给各种偏好，从而优化地整合多种偏好。此外，COOL RLHF采用多轮Online RLHF策略，使LLM能够及时适应新的人类反馈，减少奖励黑客的发生。
**(就跟孩子一样，想要奖励，需要讲条件，不能只说好坏，需要看在什么条件下是好是坏，再给糖果。)**
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711633597721-90ad6975-ef1d-46a9-9ab4-fa5a9f102a37.png#averageHue=%23f4f3f0&clientId=u798b757c-8123-4&from=paste&height=441&id=u6de55ffb&originHeight=551&originWidth=866&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=81672&status=done&style=none&taskId=uad94a10f-8d6d-4d7f-b72c-f414698a970&title=&width=692.8)

### online RLHF
获得条件奖励模型后，我们进行近端策略优化（PPO），以使llm与奖励模型 Ouyang 等人建模的人类偏好保持一致。 
两种路径：

### PPO Training Details

在RL对齐阶段，我们采用了标准的PPO（邻近策略优化）算法并对其进行了多次调整，以确保训练过程更加稳定。该框架涉及四个模型：参与者模型、评论家模型、参考模型和奖励模型。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711633743413-1fbc1e98-1f22-419f-927e-f345f308bafd.png#averageHue=%23f8f8f6&clientId=u798b757c-8123-4&from=paste&height=539&id=uf3500a8c&originHeight=674&originWidth=870&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=77383&status=done&style=none&taskId=ua7cbdba0-16cb-4259-9154-68ea21bf4c6&title=&width=696)

## 4.3 Long-Context Finetuning

为了在微调后保留LLM的长上下文能力，我们在SFT和RLHF中继续使用长上下文预训练数据，受到之前在SFT中采用长上下文预训练语料库的工作的启发（Xiong等人，2023）。具体来说，我们利用两种类型的数据：一种包括来自书籍的长上下文数据，另一种是从 GitHub 存储库获取并通过特定范例连接的长上下文数据.
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711633809558-763ea100-ce4f-4abb-a566-84a7b4993275.png#averageHue=%23f7f5f2&clientId=u798b757c-8123-4&from=paste&height=369&id=u5dbd48a7&originHeight=461&originWidth=791&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=60813&status=done&style=none&taskId=uba88dc83-3e65-435c-a163-38bdc3d6703&title=&width=632.8)


# 5 Evaluation and Analysis

在本节中，我们对语言模型在各个领域和任务中的性能进行全面的评估和分析。评估分为两个主要类别：(a) 下游任务和 (b) 一致性。对于每个类别，我们进一步将评估分解为具体的子任务，以详细了解模型的优点和缺点。最后，我们讨论了语言模型中数据污染的潜在问题及其对模型性能和可靠性的影响。除非另有明确指定，所有评估均使用 **OpenCompass**


## 5.2 Performance on Downstream Tasks
六个任务：
(1) comprehensive examinations, 
(2) language and knowledge, 
(3) reasoning and mathematics,
(4) multiple programming language coding, 
(5) long-context modeling, 
(6) tool utilization.

### 1 Comprehensive Examination
数据集：
MMLU (Hendrycks et al., 2020)：一个多项选择题数据集，包含 57 个子任务，涵盖人文、社会科学、STEM 等主题。我们报告 5 次射击的结果。 
CMMLU (Li et al., 2023a)：针对中国的多项选择题数据集，包含 67 个子任务。除了人文、社会科学、STEM等，它还包括许多中国特有的任务。我们报告 5 次射击的结果。
 C-Eval (Huang et al., 2023)：一个多项选择题数据集，包含 52 个子任务和 4 个难度级别，涵盖人文、社会科学、STEM 等主题。我们报告 5 次射击的结果。 
AGIEval（Zhong 等人，2023）：以人为本的基准，包括多项选择题和开放式问题。这些问题来自 20 场官方、公开、高标准的入学和资格考试，面向一般人类考生，并报告 0 次结果。
 GAOKAO-Bench（Zhang et al., 2023）：包含 2010 年至 2022 年中国高考（高考）的数据集，包括主观题和客观题。我们仅评估客观问题的数据集并报告 0-shot 结果。


### 2 Language and Knowledge
数据集:
TriviaQA（Joshi 等人，2017）：包含阅读理解和开放域 QA 的数据集。平均而言，每个问题有 6 个可能的答案。我们利用数据的开放域 QA 子集并报告 0 次结果。
 NaturalQuestions (Kwiatkowski et al., 2019)：QA 数据集，其中问题来自用户，答案由专家验证。我们报告 0 次结果。 
C3 (Sun et al., 2020)：自由形式多项选择中文机器阅读理解数据集。我们报告 0 次结果。
 RACE (Lai et al., 2017)：一个阅读理解数据集，其中包括中国 12 至 18 岁中学生和高中生的英语阅读理解考试题。我们使用高中生的子集并报告 0-shot 结果。 
FLORES（Team et al., 2022）：从维基百科提取的翻译数据集，涵盖 101 种语言。我们评估了从英语到其他 100 种语言的翻译结果，反之亦然。对于每对翻译任务，我们选择 100 个样本并使用 BLEU 进行评估。我们报告 8 次射击的结果。


### 3 Reasoning and Mathematics
Reasoning Datasets:
 WinoGrande（Sakaguchi 等人，2020）：一个常识推理数据集，包含 44,000 个多项选择题，每个题有两个选项。它要求模型根据场景为描述性文本中的代词选择合适的实体词。 
HellaSwag（Zellers 等人，2019）：用于评估常识自然语言推理的具有挑战性的数据集，由 70,000 个多项选择题组成。每个问题都提出一个场景和四种可能的结果，要求选择最合理的结论。 
BigBench Hard (BBH)（Suzgun 等人，2023）：大型语言模型的测试集合，BBH 从 BIG-Bench 中提取了 23 个具有挑战性的任务，其中当代语言模型当时尚未超越人类的表现。

Mathematics Datasets:
• GSM8K-Test（Cobbe 等人，2021）：包含大约 1,300 个初级情景数学问题的数据集。这些问题的解答涉及2到8个步骤，主要利用基本算术运算（加、减、乘、除）进行一系列基本计算，得出最终答案。 
MATH（Hendrycks 等人，2021）：包含 12,500 个具有挑战性的高中水平竞赛数学问题的数据集，涵盖从代数到微积分的多个领域。每个问题都包含完整的逐步解决方案。 
TheoremQA（Chen 等人，2023a）：一个 STEM 定理驱动的问答数据集，包含 800 个 QA 对，涵盖数学、EE&CS、物理和金融领域的 350 多个定理。它测试了大型语言模型在应用定理解决具有挑战性的大学水平问题方面的局限性。
MathBench（匿名，2024b）：MathBench 包含 3709 个问题，其中包含多个阶段的逐步增加的挑战。每个阶段均包含双语理论题和应用题，每个题都精准标注了三级标签，表明其细粒度的知识点。

### 4 Coding

Python 编码任务 
HumanEval HumanEval（Chen 等人，2021）是一个广泛认可的数据集，可作为评估代码生成模型性能的基准。它由 164 个精心设计的编程任务组成，每个任务都由一个 Python 函数和一个随附的文档字符串组成，以提供上下文和规范。该数据集以人类编写的代码为特色，在生成或完成程序时评估大型语言模型 (LLM) 的能力方面发挥着关键作用。 
MBPP MBPP（Austin 等人，2021）由入门级程序员可以解决的 974 个编程任务组成。这些任务的范围从简单的数字操作到需要外部知识的更复杂的问题，例如定义特定的整数序列。我们使用 MBPP 的 Santinized 版本，它仅包含经过作者手工验证的数据子集。


多种编程语言编码任务 :
HumanEval-X HumanEval-X（Zheng 等人，2023b）数据集是原始 HumanEval 基准的多语言扩展，旨在评估跨多种编程语言的代码生成模型的功能。它由 164 个手工编程问题组成，每个问题都翻译成五种主要语言：C++、Java、JavaScript、Go 和 Python。

### 5 Long-context Modeling
L-评估。 L-Eval 是一个长上下文基准，由 18 个子任务8组成，包括来自法律、经济和技术等各个领域的文本。 L-Eval由411个文档和2000多个测试用例组成，平均文档长度为7217字。该数据集中的子任务可分为两大类：5 个封闭式任务和 13 个开放式任务。封闭式任务使用基于精确匹配的准确度进行评估，而开放式任务则采用 Rouge 分数作为指标。长凳。 
LongBench 是一个长上下文基准测试，由 21 个子任务组成，总共 4750 个测试用例。它是首个双语长上下文基准测试，平均英文文本长度为 6711 个单词，平均中文文本长度为 13386 个字符。 21个子任务分为6类，可以更全面地评估模型的各方面能力。
Evaluation Results。我们在表 15 中报告了 InternLM2 在长上下文基准上的评估结果。InternLM2 的所有变体都在两个基准上展示了强大的长上下文建模性能。 InternLM2-Chat-20B-SFT 在 L-Eval 上实现了最佳性能，并大幅优于同类产品。在 LongBench 上，InternLM2-Chat-7B-SFT 在 4 个输出方面优于其他 ≤ 7B 模型

