# 第二次作业报告

## 方法：

### 1.预处理
（1）名字预处理：统一转为小写；统一使用“_”进行姓和名的分隔；考虑到数据中存在姓和名顺序不同的情况，统一对字符串按字母顺序排列  
（2）文本预处理：统一文本格式  
（3）机构预处理：简写转换 + 统一文本格式  

### 2.按照强规则进行第一步区分:
对每个待消歧名字下的所有论文：  
（1）如果除待消歧作者外重合的coauthors数≥threshold1，则划归为同一簇；  
（2）如果除待消歧作者外重合的coauthors数<threshold1，但≥threshold2，且该待消歧名字对应的org相同，则也划归为同一簇；  
最后将按强规则无法找到同簇文章的整合成一个列表存储在结果列表的最后，后续按标题、摘要等的语义相似度进行划分  

### 3.按照语义相似度进行第二步区分：
Step 1：利用所有文章的title, keywords, abstract, organization, venue训练word2vec模型；    
Step 2：利用word2vec生成的词向量（映射的数据包括title, keywords, org, venue, year），对第一部区分中的离群文章进行DBSCAN聚类，将这些离群文章划分为多簇；  
Step 3：对每个由离群文章构成的簇，计算其与原来由强规则得到的簇之间的cosine distance。若最短距离小于阈值，则将该簇划归到与其距离最短的簇中；否则，则单独作为一簇。

### 4.评估：
计算每个名字下的pairwise f1 score，取平均作为评估指标，从而调节模型中的阈值等参数


## 结果：
第一步区分过程的耗时过长，还未得到结果


## 参考思路与代码：
（1）https://www.biendata.xyz/models/category/3968/L_notebook/  
（2）https://www.biendata.xyz/models/category/3000/L_notebook/  
（3）附件：同名消歧比赛的一些参考思路.pdf

  
    
#### （比赛昵称：diiiiiik；排名：无）
