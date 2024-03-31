
internlm2 的体系：

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711864585454-21cbd328-1d2e-4ad1-baba-fe3353e4ed46.png#averageHue=%23294693&clientId=uefd9b574-443e-4&from=paste&height=331&id=ue0cb7cd8&originHeight=662&originWidth=1228&originalType=binary&ratio=2&rotation=0&showTitle=false&size=506759&status=done&style=none&taskId=u2fd0298e-fe80-4a3f-b8bd-ade7872822f&title=&width=614)

一个大模型从训练到应用需要那些步骤？
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711864778640-01f02039-81fb-471d-b373-20e4a1c8b768.png#averageHue=%23223f8e&clientId=uefd9b574-443e-4&from=paste&height=341&id=u50ae365d&originHeight=1302&originWidth=2400&originalType=binary&ratio=2&rotation=0&showTitle=false&size=706657&status=done&style=none&taskId=u8e991322-84eb-47ec-a1d7-ed12ff449be&title=&width=628)

如下图所示：从模型到应用典型流程：

1. 模型选型
2. 算力资源-微调或者续训
3. 是否需要与环境交互
4. 构建智能体
5. 模型评测
6. 模型部署

![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711864792376-f6ce8676-1d37-483f-9105-6036b3ef9782.png#averageHue=%23244190&clientId=uefd9b574-443e-4&from=paste&height=316&id=u1302e3f7&originHeight=632&originWidth=1254&originalType=binary&ratio=2&rotation=0&showTitle=false&size=332269&status=done&style=none&taskId=u7b811a71-db8f-46dc-b2d3-f72e1b7fd39&title=&width=627)

从模型到应用到典型流程：书生-浦语都提供了框架或者工具箱；
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711864881068-4b05572b-6e33-43f5-bd5c-1066425d4247.png#averageHue=%233e589c&clientId=uefd9b574-443e-4&from=paste&height=328&id=u99ca6552&originHeight=656&originWidth=1330&originalType=binary&ratio=2&rotation=0&showTitle=false&size=558293&status=done&style=none&taskId=u177cd042-9eed-4c8e-8335-80ef832e319&title=&width=665)
1，数据：语料数据集
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711864918444-383ec783-7889-4a68-9101-3e4c1fe23894.png#averageHue=%23314e97&clientId=uefd9b574-443e-4&from=paste&height=344&id=u4506006b&originHeight=668&originWidth=1286&originalType=binary&ratio=2&rotation=0&showTitle=false&size=736795&status=done&style=none&taskId=ubb6b4ca2-6176-4115-8fbd-77412eda051&title=&width=663)

2，预训练
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711866133889-8f43df00-d49e-4cf2-bb73-10bff63488e1.png#averageHue=%23415a9c&clientId=uefd9b574-443e-4&from=paste&height=335&id=uac5ddc55&originHeight=1314&originWidth=2590&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1694863&status=done&style=none&taskId=u3963284d-9fae-4691-b685-318b5f07195&title=&width=660)


3，微调
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711864985011-b4a3a8c4-b722-4973-a1cf-16019cd8a340.png#averageHue=%232a4590&clientId=uefd9b574-443e-4&from=paste&height=333&id=u3796220c&originHeight=666&originWidth=1274&originalType=binary&ratio=2&rotation=0&showTitle=false&size=474769&status=done&style=none&taskId=u314fc1e5-054f-47fc-8c6a-72d0d22ff37&title=&width=637)

高效微调框架：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865194453-47600839-3856-47b2-b184-d29f3dd3c676.png#averageHue=%236074ab&clientId=uefd9b574-443e-4&from=paste&height=331&id=u8b8bc7e3&originHeight=1322&originWidth=2556&originalType=binary&ratio=2&rotation=0&showTitle=false&size=2069226&status=done&style=none&taskId=ub6374753-8c0b-4c27-8fcf-98a05598056&title=&width=640)

4，评测：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865252482-1a60b834-7713-4f95-a522-b6b02b3ff6ec.png#averageHue=%23294895&clientId=uefd9b574-443e-4&from=paste&height=269&id=u9109421d&originHeight=488&originWidth=1230&originalType=binary&ratio=2&rotation=0&showTitle=false&size=415997&status=done&style=none&taskId=uecdeacaa-5682-4147-a1b4-7bb9113cde1&title=&width=679)

榜单：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865296538-dc7683b0-44d7-437b-ad51-ad8326de4f4c.png#averageHue=%23fafcfa&clientId=uefd9b574-443e-4&from=paste&height=367&id=u1dbf4ca0&originHeight=686&originWidth=1210&originalType=binary&ratio=2&rotation=0&showTitle=false&size=398078&status=done&style=none&taskId=ue75863d0-f3ff-4cde-862a-09c74f57d8d&title=&width=648)

数据集评测：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865364110-64d13029-6a2e-4977-b6e5-b4d207cb0150.png#averageHue=%23304e99&clientId=uefd9b574-443e-4&from=paste&height=333&id=u57362474&originHeight=658&originWidth=1244&originalType=binary&ratio=2&rotation=0&showTitle=false&size=580304&status=done&style=none&taskId=u43da7e57-2e85-436d-a55c-b518600ef83&title=&width=629)

评测集：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865384766-dfb0063f-a72f-4cc1-9292-7d3606ed4457.png#averageHue=%23d8e4d9&clientId=uefd9b574-443e-4&from=paste&height=340&id=u4fb890e5&originHeight=652&originWidth=1200&originalType=binary&ratio=2&rotation=0&showTitle=false&size=489623&status=done&style=none&taskId=u4e944448-a1a1-47ce-84ce-010a22f476a&title=&width=625)


5，模型部署：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865524412-b0beaee9-58a2-4213-b4b8-2d6b0ee41639.png#averageHue=%23314d97&clientId=uefd9b574-443e-4&from=paste&height=320&id=ue194885e&originHeight=640&originWidth=1218&originalType=binary&ratio=2&rotation=0&showTitle=false&size=567638&status=done&style=none&taskId=u6e7d4d38-24fd-4331-8c5d-30700a6e843&title=&width=609)

6，智能体框架：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865610267-070c2ce3-a26d-4c02-b76b-180fa96bbf85.png#averageHue=%234862a2&clientId=uefd9b574-443e-4&from=paste&height=321&id=u96f7b71f&originHeight=642&originWidth=1206&originalType=binary&ratio=2&rotation=0&showTitle=false&size=550434&status=done&style=none&taskId=ub6a6cb11-840d-44a5-8cb0-6b723a72e67&title=&width=603) 
智能体工具箱：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38497976/1711865690163-f7b48c4a-2225-4a20-a025-00efc38b205e.png#averageHue=%2399a8c7&clientId=uefd9b574-443e-4&from=paste&height=327&id=u30dad5f0&originHeight=1354&originWidth=2554&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1680048&status=done&style=none&taskId=ub29fffe3-d508-480a-91c6-7ac96179582&title=&width=617)
