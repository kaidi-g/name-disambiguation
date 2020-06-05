# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%

import re
import json
import numpy as np
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


#%%

dirpath = 'C:/Users/kaidi/Documents/PhD/高级机器学习训练营/HW2-同名消歧/data'

with open(dirpath + '/train/train_author.json','r', encoding='UTF-8') as f:
     dict_train_author = json.load(f)

with open(dirpath + '/train/train_pub.json','r', encoding='UTF-8') as f:
     dict_train_pub = json.load(f)    
     
with open(dirpath + '/sna_data/sna_valid_author_raw.json','r', encoding='UTF-8') as f:
     dict_valid_raw = json.load(f)

with open(dirpath + '/sna_data/sna_valid_pub.json','r', encoding='UTF-8') as f:
     dict_valid_pub = json.load(f) 

with open(dirpath + '/sna_data/sna_valid_example_evaluation_scratch.json','r', encoding='UTF-8') as f:
     dict_valid_example = json.load(f) 
 

#%%  

# 数据预处理; 参考 https://biendata.com/models/category/3000/L_notebook/

# 预处理名字
def preprocessname(name):   
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name)
    name = '_'.join(sorted(name.split('_')))#将名字统一按字母顺序排序
    return name

#预处理文本
def preprocesstext(content):
    content = re.sub('[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+', " ", content)
    content = content.strip()
    content = content.lower()
    content = re.sub(r'\s{2,}', ' ', content).strip()
    return(content)
    
# 预处理机构
def preprocessorg(org):
    if org != "":
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        org = org.split(';')[0]  # 多个机构只取第一个
        org = preprocesstext(org)
         
    return org


#%%

#先基于强规则进行预区分:
##对每个待消歧名字下的所有论文：
###(1)如果重合的coauthors数≥threshold1，则划归为同一簇；
###(2)如果重合的coauthors数<threshold1，但≥threshold2，且该待消歧名字对应的org相同，则也划归为同一簇；
###(3)如果重合的coauthors数<threshold2，则划归为不同簇；
#思路参考所给附件《同名消歧比赛的一些参考思路》中的最后一条
#代码参考https://www.biendata.xyz/models/category/3000/L_notebook/

#获取待消歧作者的org
def get_org(authors_info, author_name):
    author_name = preprocessname(author_name)
    for au in authors_info:
        name = preprocessname(au['name'])
        if (name == author_name) and 'org' in au:
            return preprocessname(au['org'])
    return ''

#按重合的authors数量和org进行区分
def disambiguate_by_coauthor(raw_data, pub_data, threshold1, threshold2):
    
    result_dict = {}
    
    for name in raw_data:
        print(name)
        res = []
        pids = raw_data[name]
        paper_dict = {}
        
        for pid in pids:
            print('pid: ', pid )
            d = {}
            org = get_org(pub_data[pid]['authors'], name)                
            authors = [preprocessname(au['name']) for au in pub_data[pid]['authors']]
            if name in authors:
                authors.remove(name)
            
            d['authors'] = authors
            d['org'] = org    
            
            if len(res) == 0:
                res.append([pid])
                print('start')
            else:
                index = -1
                for i, cluster in enumerate(res):
                    for pid_c in cluster:
                        n_coauthor = len(set(authors) & set(paper_dict[pid_c]['authors']))
                        if n_coauthor >= threshold1:
                            index = i
                            continue
                        elif n_coauthor >= threshold2:
                            if org:
                                if org == paper_dict[pid_c]['org']:
                                    index = i 
                                    continue
                
                if index != -1:
                    res[i].append(pid)
                    print('belong to cluster %d' %i)
                else:
                    res.append([pid])
                    print('belong to new cluster')
                    
            paper_dict[pid] = d
        
        #将按强规则无法找到同簇文章的整合成一个列表存储在结果列表的最后，后续再继续按弱规则进行划分
        residual_list = []
        for l in res:
            if len(l) == 1:
                residual_list.append(l[0])
                res.remove(l)
        res.append(residual_list)
    
        result_dict[name] = res
        print(result_dict)
    
    json.dump(result_dict, open(dirpath+'/result/disambiguate_by_coauthor.json', 'w', encoding='utf-8'), indent=4)

disambiguate_by_coauthor(dict_valid_raw, dict_valid_pub, 2, 1)            


#%%
    
#提取所有文本到txt文件，用于后续训练word2vec模型    
f = open(dirpath + '/all_text.txt', 'w', encoding = 'utf-8')

#训练集中的文本
for pid in dict_train_pub:  
    
    pub = dict_train_pub[pid] 
    
    for author in pub['authors']:
        if 'org' in author:
            org = preprocesstext(preprocessorg(author['org']))
            f.write(org + '\n')
    
    title = preprocesstext(pub['title'])  
    f.write(title + '\n')  
      
    venue = preprocesstext(pub['venue'])
    f.write(venue + '\n')
    
    if "keywords" in pub and type(pub["keywords"]) is str:
        keywords = preprocesstext(" ".join(pub['keywords']))
        f.write(" ".join(keywords)  + '\n')
        
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = preprocesstext(pub['abstract'])
        f.write(abstract  + '\n')

#验证集中的文本    
for pid in dict_valid_pub:
    
    pub = dict_valid_pub[pid]  
    
    title = preprocesstext(pub['title'])  
    f.write(title + '\n')  
      
    venue = preprocesstext(pub['venue'])
    f.write(venue + '\n')
    
    if "keywords" in pub and type(pub["keywords"]) is str:
        keywords = preprocesstext(" ".join(pub['keywords']))
        f.write(" ".join(keywords)  + '\n')
        
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = preprocesstext(pub['abstract'])
        f.write(abstract  + '\n')

f.close()


#%%   

#训练word2vec模型

sentences = word2vec.Text8Corpus(dirpath + '/all_text.txt')
model = word2vec.Word2Vec(sentences, size=100, negative =5, min_count=2, window=5)
model.save(dirpath + '/word2vec.model')


#%%

#生成验证集每篇文章的词嵌入向量：word2vec(title, keywords, org, venue, year)

mopdelfilePath = dirpath + '/word2vec/word2vec.model'
model = word2vec.Word2Vec.load(mopdelfilePath)

wv_dict = {}

for pid in dict_valid_example:
    
    pub = dict_train_pub[pid] 
        
    title = preprocesstext(pub['title'])
    
    if "keywords" in pub and type(pub["keywords"]) is str:
        keywords = preprocesstext(" ".join(pub['keywords']))
    else:
        keywords = ''
    
    for author in pub['authors']:
        if 'org' in author:
            org = preprocesstext(preprocessorg(author['org']))
      
    venue = preprocesstext(pub['venue'])
    
    year = str(pub['year'])
    
    pstr = ' '.join([title, keywords, org, venue, year])
    
    wv_dict[pid] = model.wv(pstr)
    
json.dump(wv_dict, open(dirpath+'/word2vec/wv_valid.json', 'w', encoding='utf-8'), indent=4)


#%%

#基于语义相似度的弱规则

with open(dirpath + '/word2vec/wv_valid.json','r', encoding='UTF-8') as f1:
    wv_dict= json.load(f1)

with open(dirpath+'result/disambiguate_by_coauthor.json', 'r', encoding='utf-8') as f2:
    results = json.load(f2)
    
dis_threshold = 0.2

for name in results.keys():    
    #对离群的文章进行聚类
    outliers = results[name][-1]
    results.pop()
    out_wv = [wv_dict[pid] for pid in outliers]
    clustering = DBSCAN(eps=0.2, min_samples=1, metric='cosine').fit_predict(out_wv)
    labels = sorted(list(set(clustering)))
    out_dict = {}
    for label in labels:
        out_dict[label] = [i for i in range(len(clustering)) if clustering[i] == label]
    
    #计算新得到的簇和原来由强规则得到的簇之间的距离 
    for label in labels:
        X = out_wv[out_dict[label]]
        
        min_distance = 9999
        index = -1
        for k in range(len(results[name])):
            pre_pids = results[name][k]
            Y = [wv_dict[pid] for pid in pre_pids]
            distance = pairwise_distances(X, Y, metric='cosine')
            if distance < min_distance:
                min_distance = distance
                index = k
        
        #若和已有簇的最小距离小于阈值，则划为同一类；否则则划为新的一类
        if min_distance <= dis_threshold:
            results[name][index].extend(outliers[out_dict[label]])
        else:
            results[name].append(outliers[out_dict[label]])

json.dump(results, open(dirpath+'/final_results.json', 'w', encoding='utf-8'), indent=4)   
 

#%%

#结果评估，参考https://www.biendata.xyz/models/category/3968/L_notebook/

with open(dirpath+'result/final_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

def pairwise_evaluate(correct_labels,pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1

f1_scores = []
for name in dict_valid_example.keys():
    pairwise_f1 = pairwise_evaluate(dict_valid_example['name'], results['name'])[2]
    f1_scores.append(pairwise_f1)
print('avg_f1: %.4f', np.mean(f1_scores))
    
    
    
    