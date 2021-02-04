#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict, OrderedDict
import json
import os
path = os.getcwd()


# In[2]:


fr = open(path+'/Train_Caption.txt', 'r')
lines = len(fr.readlines())
print(lines)
fr.close()


# spilt 
# 5%are selected to be val set
train_num = round(lines*0.95)


fr = open(path+'/Train_Caption.txt', 'r')
i = 1
imgid = 0
sentid = 0
sentid2 = 0
sentences = []
images = []

for n, line in enumerate(fr):
    v = line.strip().split('| ')
    try:
        v_ = v[2].replace('.', '')
    except:
        if(i%5 == 0):
            imgid += 1
            sentid2 += 5
        i += 1
        sentid += 1
        continue
    v_ = v_.lower()
    token = v_.split()
    sentence = {
        "token": token,
        "raw": v[2],
        "imgid": imgid,
        "sentid": sentid}
    sentences.append(sentence)
    # for train 
    if(i%5 == 0 and n<=train_num):
        image = {
            "sentids": [sentid2, sentid2+1, sentid2+2, sentid2+3, sentid2+4],
            "imgid": imgid,
            "sentences": sentences,
            "split": "train",
            "filename": v[0]}
        imgid += 1
        sentid2 += 5
        images.append(image)
        sentences = []
    # for val 
    if(i%5 == 0 and n>train_num):
        image = {
            "sentids": [sentid2, sentid2+1, sentid2+2, sentid2+3, sentid2+4],
            "imgid": imgid,
            "sentences": sentences,
            "split": "val",
            "filename": v[0]}
        imgid += 1
        sentid2 += 5
        images.append(image)
        sentences = []
    
    sentid += 1
    i += 1
fr.close()

test_dict = {
    'images': images,
    'dataset': 'flickr30k'}

json_str = json.dumps(test_dict)
m=0
with open('m_dataset_flickr30k.json', 'w') as json_file:
    json_file.write(json_str)
    m=m+1


# In[ ]:




