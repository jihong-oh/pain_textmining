#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:17:15 2017

@author: Chang-Eop
"""

import os
os.chdir(r'C:\Users\JH\Dropbox\TextMining\PainNetwork')

import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import toCytoscape as cy
import codecs
import csv
import re
import time
import seaborn as sns; sns.set(color_codes=True)

    ##########동의어 목록을 뽑아놓은 상태에서는, 필요 없는 부분 FROM HERE#########
data = pd.read_excel('NIFSTD.xlsx')
data_1 = data.ix[:,['Preferred Label','Synonyms', 'has_obo_namespace']]

#data_1.ix[data_1.ix[:,0] == 'amygdala',:]

data_2 = data_1[data_1.ix[:,2] == 'uberon']
data_2.index = range(data_2.shape[0])


#NIFSTD의 동의어 정리
terms = [] #nested list 형식으로 동의어들 묶어넣을것
for i in range(data_2.shape[0]):
    term_prefer = data_2.ix[i,0]
    try: #동의어 없는 경우(nan) 대비.
        term_synonym = data_2.ix[i,1].split("|")
    except AttributeError:
        term_synonym = []
    
    temp = [term_prefer]
    temp.extend(term_synonym) #제일 앞이 prefer, 이어서 동의어들.
    terms.append(temp)


# pain: terms = 11239개      
    ########## UNTIL HERE #########    

########## 기존에 만들어놓은 동의어목록으로 terms 변수 설정. 목록 있다면 여기에서부터 ##########

# query_terms.csv 파일 불러와서 list로 만드는 코드
with open('query_terms.csv','r') as query_terms: 
    reader = csv.reader(query_terms)
    query_terms = list(reader)
    
query_terms = [[subelt for subelt in elt if subelt !=''] for elt in query_terms]  # nested list에서 element 중 '' 를 제거

terms = query_terms
 


   
candi_terms = pd.read_excel('ROI_Brede.xlsx')#'170525_RPO_Brede(except_Broadmann_Economo_Monkey)_1107.xlsx')
#오지홍 정리 


#candidate terms 모으기    (nan 빼고, 그냥 다 리스트로 만든것)
candi_total = []
for i in range(candi_terms.shape[0]):
    for j in range(candi_terms.shape[1]):
        if type(candi_terms.ix[i,j]) == str: 
            candi_total.append([candi_terms.ix[i,j]])
        else:
            pass
        
        
# candidate terms 중에 NIFSTD 동의어 목록에서 찾아지는 수
count = 0        
for candi in candi_total:
    
    for t in range(len(terms)):
        if candi[0].lower() in terms[t]:
            count += 1

#결과 : count = 233 -> 동의어목록 정리 후 count :217   
            

#text_total = pd.read_excel('pain_papers.txt')

f = codecs.open('180129_chronic_pain.txt','r',"utf-8")
texts = f.readlines() #초록 하나가 리스트의 요소 하나로 들어감.

#전체 소문자로 변환
for i,line in enumerate(texts):
    texts[i] = line.lower()


### 106 - 179까지 실행해야 함.
#candidate terms의 동의어들로 쿼리 구성하고, 문서 전체에서의 등장 빈도 카운트
freq_of_candis = {}

#syn_list = [] #query 목록 만들 때, 동의어를 nested list로 만들기 위해 필요했던 코드(line 126에설명)
n = 0
for candi in candi_total:
    counts = 0
    n += 1
    
    #단어 앞뒤 공백 제거
    candi = [candi[0].strip()]
    #동의어 포함한 쿼리 구성
    for t in range(len(terms)):
        if candi[0].lower() in terms[t]: #동의어 목록에 있으면 
            query = terms[t] #동의어 리스트를 쿼리로
            #syn_list.append(query) #쿼리를 syn_list에 append해서 query 목록 만들 때만 사용 
            break #찾아으면 다음 루프로
        else: #동의어 목록에 없으면 
            query = [candi[0].lower()] #그냥 그 자체만 리스트 형태로

                      #단어 앞뒤 공백 제거


#synonym의 약어로 인해, 잘못 counting 되는 단어들이 발생하는 문제 있었음
#예를 들면 thalamus에 th라든지 hypophysis의 HP
#query_syn.csv 파일로 저장해서, manually 수정하여 'query_syn_sum.csv' 파일 생성해놓음
#위의 n=0 부터 query까지 돌려서 syn_list에 동의어 저장한 뒤에, csv파일로 저장.
# 수정한 이후 다시 csv파일을 list로 읽어온 뒤에 ''은 제거하여 query_terms 만든 것

   ### 동의어목록 있는 syn_list를 'query_syn.csv' 파일로 저장하는 코드

#syn = open('nn_query_syn.csv','w') 
#writer =csv.writer(syn)
#writer.writerows(syn_list)
#syn.close()

### 동의어목록 manually 수정한 'query_terms.csv' 파일 불러오는 코드.

# with open('query_terms.csv','r') as query_terms:
#    reader = csv.reader(query_terms)
#    query_terms = list(reader)
#    print(query_terms)
#query_terms = [[subelt for subelt in elt if subelt !=''] for elt in query_terms] #nested list의 요소들 중 '' 포함된 것 삭제

                           
    
    #전체 텍스트에서 쿼리(동의어 포함) 등장 빈도 카운트(문서당 최대 1회 카운트)
    texts_found = [] #동의어들 사이에 같은 초록을 중복 카운팅하지 않기 위해 초록 발견될때마다 등록해놓는곳. 
    for qs in query:
         #동의어 검색된 초록 리스트(인덱스)
        #print('\n')
        for i, text in enumerate(texts):
            #print(i)
            #text = text[1]
            if qs in text: #해당 용어 발견했을때
                
                pattern_0 = re.compile(qs) #용어 존재    
                pattern_1 = re.compile('[a-z]'+qs)#용어 앞에 다른 문자 연결되어있거나
                pattern_2 = re.compile(qs+'[a-z]')#용어 뒤에 다른 문자 연결된 경우라면
                
                iters = pattern_0.finditer(text) #용어 존재하는 경우를  iterable 객체로 반환
                
                for it in iters: #용어 존재 경우 하나씩 돌아가면서
                    start = it.span()[0] #텍스트 상에서의 시작위치
                    end = it.span()[1] #텍스트 상에서의 용어 종료 위치
                    test_phrase = text[start-1:end+1] #용어 앞뒤로 한칸 추가하여 따냄
                    if not (pattern_1.search(test_phrase)) and not (pattern_2.search(test_phrase)) and not (i in texts_found):
                        # 앞뒤로 알파벳 있지 않은 경우만 카운트
                        counts += 1
                        texts_found.append(i) #동의어에서 한번이라도 카운팅된 초록은 등록(다시 카운팅 되지 않도록)
                        
                        break #하나라도 찾아서 카운트했으면 다음 텍스트로 넘어감       
                
    freq_of_candis[candi[0]] = counts
    print(counts, ", ", n, " of ", len(candi_total))
        


#결과 csv 포맷으로 저장
f2 = open('FreqCandis_ROI_Brede.csv', 'w')
w = csv.writer(f2)

for r_i, row in enumerate(freq_of_candis.items()):
    if r_i == 0:
        w.writerow(('Node', 'Freq')) #pandas 대비하여 컬럼 명 추가
    else:
        w.writerow(row)
f2.close()



         

        
#이하 작업중 시험 코드 
#pain과 다른 brain diseases 비교시에, node를 pain 기준으로 할 거라면 안 돌려도 되는 부분
nodes = pd.read_csv('FreqCandis_ROI_Brede.csv') #f2에서 csv로 저장한 파일명과 동일해야 함
nodes = nodes.sort_index(by = 'Freq', ascending=False)

nodes = nodes[nodes.Node != 'Motor cortex'] # motor area 동의어
nodes = nodes[nodes.Node != 'Primary motor cortex'] # motor area 동의어
nodes = nodes[nodes.Node != 'Hippocampal formation']  # hippocampus 동의어
nodes = nodes[nodes.Node != 'Cerebrospinal fluid'] # 적절치 않다고 판단
nodes = nodes[nodes.Node != 'Archicortex'] #동의어에 hippocampus가 있어 빈도 높게 나오는것으로 보임.
#nodes = nodes[nodes.Node != 'Lateral occipito-temporal gyrus'] #Fusiform gyrus 동의어
nodes = nodes[nodes.Node != 'Central nuclear group'] #Central amygdaloid nucleus 동의어
nodes = nodes[nodes.Node != 'Medial frontal gyrus'] #Middle frontal gyrus 동의어
nodes = nodes[nodes.Node != 'Reticular formation'] #Medullary reticular formation 인데, reticular formation도 preferred에 있어서 고려할 필요
nodes = nodes[nodes.Node != 'Midbrain tegmentum'] #Cerebral peduncle
nodes = nodes[nodes.Node != 'White matter'] # 추가. white matter 적당하지 않다고 생각해서 제외
nodes = nodes[nodes.Node != 'Sulcus'] # 추가. sulcus가 지칭하는 region 모호해서 삭제
nodes = nodes[nodes.Node != 'Dorsal striatum'] # striatum 동의어
nodes = nodes[nodes.Node != 'Corpus Striatum'] # striatum 동의어

thre = 100 # 총등장빈도 frequency 설정. pain에선 100 이상으로 설정했음/ chronic pain은 10
nodes = nodes[nodes.ix[:,1] >= thre] # 일정 빈도 이상에 해당하는 node만 
nodes.to_csv('FreqCandis_ROI_Brede_nodes.csv')   #nodes 를 csv 파일로 저장      
     
#list(data_2[data_2.ix[:,0] == 'fusiform gyrus']['Synonyms']) # node frequency 같은 brain region에 대해 동의어 겹치는지 확인해보는 코드. 겹치면 preferred 남기고 나머지 용어 지움.          
            
#### 시간 오래걸리는데, 처리 안해도 최종 결과에 영향 업는것으로 (일단) 판단되어 비활성화.
## texts에서 \n 제거 (시간 오래 걸림)
#ns = texts.count('\n')
#for i in range(ns): #'\n' in texts:
#    texts.remove('\n')
#    print(i+1, " of ", ns)
# 
## 이유 알수 없지만 '.\n'이 리스트에 섞여 있음.    
#for i in range(texts.count('.\n')):
#    texts.remove('.\n')
    
    



#texts를 text_lists로 정리.
#texts는 abstract가 쪼개어져 있는 경우가 있으므로 이를 다시 정리함. 
#요소 하나는 하나의 paper 정보 포함하도록 (year, abstract)가 되도록.
#시간 오래 걸림!!!!!!!

text_list = [] #엘리먼트 개별  (임시적으로 assign)
text_lists = [] 
for i, text in enumerate(texts):
    print(i+1, len(texts))
    try:
        year = int(text)
        text_lists.append(text_list)
        text_list = [year]
    except ValueError:
        text_list.append(text)
        
text_lists.append(text_list) #맨 마지막은 따로 추가해야 함
text_lists=text_lists[1:] #맨 앞에 '\n' 반복되는 꾸러미 하나 있어서 제거
### #text_lists.remove([]) #비활성화한 부분(223-232) 안 돌리면 필요 없음 .제일 첫 엘리먼트가 공리스트라 제거.
#pain : text_lists = 141536개 / text_list = 229개   
                      


            
           
#### hypothesis driven/data driven 구별하기 위해 manually curating할 초록 list 뽑기######

# 우선 동의어 포함한 query가 포함된 초록 list를 pain_text에 담고
# 이후 랜덤하게 일정 개수의 초록만 골라 저장.

################(from here) ####################

pain_text =[] #동의어목록 terms에 있는 영역이 등장한 초록의 출판년도와 초록을 담을 list
for i, year_abstract in enumerate(text_lists):
    print(i+1, " of ", len(text_lists))
    
    #            
    for j, node in enumerate(nodes.Node):
        node = node.lower()
        
        #노도의 동의어 목록으로 쿼리 구성
        for t in range(len(terms)):
            if node in terms[t]: #동의어 목록에 있으면 
                query = terms[t] #동의어 리스트를 쿼리로
                break #찾았으면 다음 루프로
            else: #동의어 목록에 없으면 
                query = [node]
               
        for qs in query:
            if qs in year_abstract[1]:
                pain_text.append([year_abstract[0],year_abstract[1]]) #연도와 초록으로 구성된 list를 append.
                
import random
rand_abs  = random.sample(pain_text,50) #랜덤하게 50개의 abs 추출
rand_abs.sort(key=lambda x:x[0]) # 연도순으로 정리
#text file로 저장
with open('random_abs.txt','w',encoding = "utf-8") as f:
   for rand_list in rand_abs:
       for string in rand_list:
           f.write(str(string))
           
################(until here) ####################
        
                
t_start = time.time()  
#occur_matrix 생성
#abstract별로 np.array (row: abstracts, col: year, freq. of nodes) 생성
occur_matrix = np.zeros((len(text_lists), len(nodes)+1))
for i, year_abstract in enumerate(text_lists):
    print(i+1, " of ", len(text_lists))
    occur_matrix[i,0] = year_abstract[0] #year
    
    #            
    for j, node in enumerate(nodes.Node):
        node = node.lower()
        
        #노도의 동의어 목록으로 쿼리 구성
        for t in range(len(terms)):
            if node in terms[t]: #동의어 목록에 있으면 
                query = terms[t] #동의어 리스트를 쿼리로
                break #찾아으면 다음 루프로
            else: #동의어 목록에 없으면 
                query = [node] #그냥 그 자체만 리스트 형태로
                                             
                
                
        #초록에 쿼리(동의어 포함) 등장 여부 결정(카운트 아닌 여부(1/0))
        val = [] #동의어 각각의 등장 여부를 1/0 리스트로 만들것.
        for qs in query:
            
            if qs in year_abstract[1]: #해당 용어 발견했을때
                
                pattern_0 = re.compile(qs) #용어 존재    
                pattern_1 = re.compile('[a-z]'+qs)#용어 앞에 다른 문자 연결되어있거나
                pattern_2 = re.compile(qs+'[a-z]')#용어 뒤에 다른 문자 연결된 경우라면
                
                iters = pattern_0.finditer(year_abstract[1]) #용어 존재하는 경우를  iterable 객체로 반환
                
                for it in iters: #용어 존재 경우 하나씩 돌아가면서
                    start = it.span()[0] #텍스트 상에서의 시작위치
                    end = it.span()[1] #텍스트 상에서의 용어 종료 위치
                    test_phrase = year_abstract[1][start-1:end+1] #용어 앞뒤로 한칸 추가하여 따냄
                    if not (pattern_1.search(test_phrase)) and not (pattern_2.search(test_phrase)):
                        # 앞뒤로 알파벳 있지 않은 경우만 카운트
                        val.append(1)
                        break #하나라도 찾아서 카운트했으면 다음 텍스트로 넘어감
        occur_matrix[i,j+1] = int(1 in val)                
                
    
#plt.matshow(occur_matrix[:500,1:])

t_end = time.time()          
        
print(t_end - t_start)         
                    
                

#>>> (t_end - t_start)/3600   
#9.949767874413066  
 
#year에 따라 sorting                
occur_matrix  = occur_matrix[np.argsort(occur_matrix[:,0])]                 
          
#연도별 총 초록 수
y_min = int(min(occur_matrix[:,0]))
y_max = int(max(occur_matrix[:,0]))
y_hist = np.zeros((y_max - y_min +1, 1))
for y_i, y in enumerate(range(y_min, y_max+1)):
    y_hist[y_i] = sum(occur_matrix[:,0] == y)

#첫번째 칼럼에 연도 배치    
y_hist = np.array([np.arange(y_min, y_max+1), y_hist[:,0]]).T

plt.figure()
plt.plot(y_hist[:,0], y_hist[:,1])

#y_hist_2: 최초-1975년까지 초록수 합하여 1975년으로 할당
y_hist_2 = y_hist[y_hist[:,0] == 1975]
y_hist_2[:,1] = np.sum(y_hist[y_hist[:,0] < 1976,1])
y_hist_2 = np.concatenate((y_hist_2,y_hist[y_hist[:,0] > 1975]))


#occur_matrix_2: 최초 - 1975년 초록까지 모두 1975년으로 처리
occur_matrix_2 = occur_matrix[:]
occur_matrix_2[occur_matrix_2[:,0] < 1976,0] = 1975

plt.figure()
plt.plot(y_hist_2[:,0], y_hist_2[:,1])


#총 frequency

total_freq = np.zeros((len(y_hist_2)-2, len(nodes))) #2015년까지
for y_i, y in enumerate(range(int(y_hist_2[0,0]), int(y_hist_2[-2,0]))):
    print(y_i, y)
    freq_sum_year = occur_matrix[occur_matrix[:,0] == y].sum(axis = 0)
    freq_sum_year = freq_sum_year[1:] # 첫번쨰 열은 년도의 합이기 때문에 제외
    total_freq[y_i,:] = freq_sum_year

total_freq_sum= total_freq.sum(axis=1)
# 그래프로 나타내기 (총frequency와, 각 그룹 대표 node의 추세를 보여주는 그래프)
#2016년 MeSH term 라벨링 제대로 안되어 있는 것 같아서 2015년까지 보여줄 것.
plt.figure(figsize=(15,10))
plt.plot(y_hist_2[0:41,0], total_freq_sum[0:41], 'black', label = 'Total') # 총frequency(node개수로 노멀라이즈)
#plt.plot(y_hist_2[0:42,0], y_hist_2[0:42,1]/len(nodes), 'black', label = 'Total') # 총frequency(node개수로 노멀라이즈)
plt.plot(y_hist_2[0:41,0], total_freq[:41,2],'g',label ='Brain stem') 
plt.plot(y_hist_2[0:41,0], total_freq[:41,7],'b', label = 'Medulla oblongata')
#plt.plot(y_hist_2[0:42,0], total_freq[:,10],'b', label = 'Hypophysis')
#plt.plot(y_hist_2[0:42,0], total_freq[:,14],'b', label = 'Midbrain')
#plt.plot(y_hist_2[0:42,0], total_freq[:,12],'b', label = 'Hypothalamus')
plt.plot(y_hist_2[0:41,0], total_freq[:41,1], 'r', label = 'ACC')
#plt.plot(y_hist_2[0:42,0], total_freq[:,4], 'r', label = 'insula') 
#plt.plot(y_hist_2[0:42,0], total_freq[:,13], 'grey',label = 'Hippocampus')
plt.plot(y_hist_2[0:41,0], total_freq[:41,18], 'grey',label = 'LC')
plt.legend(loc='upper left', fontsize = 'x-large')
plt.yticks(Fontsize=30)
plt.xticks(Fontsize=30)
plt.savefig('Total frequency.png', dpi=150)

# abstract 별로, 등장한 node개수를 counting하여 bar graph 생성
freq_per_abs = np.zeros((occur_matrix.shape[0], 2))
for i in range(occur_matrix.shape[0]):
    sum_per_abs = occur_matrix[i,1:].sum(axis=0)
    freq_per_abs[i,1] = sum_per_abs

freq_per_abs[:,0] = occur_matrix[:,0]
freq_per_abs = freq_per_abs[freq_per_abs[:,1] > 0]
freq_per_abs = freq_per_abs.astype('int')

freq_per_abs_1970s = freq_per_abs[(freq_per_abs[:,0]<1980)]
freq_per_abs_1980s = freq_per_abs[(freq_per_abs[:,0]>=1980) & (freq_per_abs[:,0]<1990)]
freq_per_abs_1990s = freq_per_abs[(freq_per_abs[:,0]>=1990) & (freq_per_abs[:,0]<2000)]
freq_per_abs_2000s = freq_per_abs[(freq_per_abs[:,0]>=2000) & (freq_per_abs[:,0]<2010)]
freq_per_abs_2010s = freq_per_abs[(freq_per_abs[:,0]>=2010) & (freq_per_abs[:,0]<2016)] #2015까지

sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
val = plt.hist(freq_per_abs_1970s[:,1],bins =range(1,11), log =True, color='r', edgecolor='black', linewidth=1)
#ax.set_ylim(1,10**3)
#plt.plot(val[1],val[0],'r')
plt.title('before 1980', Fontsize=30)
plt.xlabel('node frequencies per abstract', fontsize=30)
plt.ylabel('number of abstracts', fontsize=30)
plt.yticks(Fontsize=30)
plt.xticks(Fontsize=30)
plt.savefig('freq_per_abs_before1980.png',dpi=150)

plt.figure(figsize=(10,10))
val = plt.hist(freq_per_abs_1980s[:,1], bins =range(1,11), log = True,color='r', edgecolor='black', linewidth=1)
plt.title('1980s', Fontsize=30)
plt.xlabel('node frequencies per abstract', fontsize=30)
plt.ylabel('number of abstracts', fontsize=30)
plt.yticks(Fontsize=30)
plt.xticks(Fontsize=30)
plt.savefig('freq_per_abs_1980s.png',dpi=150)

plt.figure(figsize=(10,10))
plt.hist(freq_per_abs_1990s[:,1],bins =range(1,11), log=True,color='r', edgecolor='black', linewidth=1)
plt.title('1990s', Fontsize=30)
plt.xlabel('node frequencies per abstract', fontsize=30)
plt.ylabel('number of abstracts', fontsize=30)
plt.yticks(Fontsize=30)
plt.xticks(Fontsize=30)
plt.savefig('freq_per_abs_1990s.png',dpi=150)

plt.figure(figsize=(10,10))
plt.hist(freq_per_abs_2000s[:,1],bins =range(1,11), log=True,color='r', edgecolor='black', linewidth=1)
plt.title('2000s', Fontsize=30)
plt.xlabel('node frequencies per abstract', fontsize=30)
plt.ylabel('number of abstracts',fontsize=30)
plt.yticks(Fontsize=30)
plt.xticks(Fontsize=30)
plt.savefig('freq_per_abs_2000s.png',dpi=150)

plt.figure(figsize=(10,10))
plt.hist(freq_per_abs_2010s[:,1],bins =range(1,11), log=True,color='r', edgecolor='black', linewidth=1)
plt.title('2010s', Fontsize=30)
plt.xlabel('node frequencies per abstract', fontsize=30)
plt.ylabel('number of abstracts', fontsize=30)
plt.yticks(Fontsize=30)
plt.xticks(Fontsize=30)
plt.savefig('freq_per_abs_2010s.png',dpi=150)

#plt.bar(range(freq_per_abs.shape[0]), freq_per_abs[:], color='b')

##### 기존방식으로 rel_freq구하기
# relative frequency(slidinw window 적용 안한 채로 연도별 상대빈도)
rel_freq = np.zeros((len(y_hist_2)-2, len(nodes))) #2015년까지
for y_i, y in enumerate(range(int(y_hist_2[0,0]), int(y_hist_2[-2,0]))):
    print(y_i, y)
    rel_freq_year = occur_matrix[occur_matrix[:,0] == y].sum(axis = 0)/y_hist_2[y_i,1]
    rel_freq_year = rel_freq_year[1:]
    rel_freq[y_i,:] = rel_freq_year
    
##relative frequency clustermap 생성(sliding window 안하고 연도별로)

rel_freq_df = pd.DataFrame(rel_freq[:41,:]) #rel_frq를 dataframe 형태로 전환. 2015년까지
nodes_list = nodes.ix[:,0].tolist() #node_list 만들어서 column index 만들기 위함
nodes_list[27] = 'Reticular formation' # medullary reticular formation 동의어에 reticular formation 있으므로 전환
nodes_list[1] = 'ACC' # anterior cinculate gyrus 약어 변환
nodes_list[3]= 'PAG' # periaqueductal gray 약어
nodes_list[9] = 'PFC' #Prefrontal cortex 약어
nodes_list[18] = 'LC' # locus coeruleus 약어
nodes_list[20] = 'NAc' # Nucleus accumbens 약어
nodes_list[24] = 'DLPFC' # Dorsolateral prefrontal cortex 약어
nodes_list[25] = 'CeA' #Central amygdaloid nucleus 약어
                       
rel_freq_df.columns = nodes_list #rel_freq_df의 column 이름을 노드명으로
rel_freq_df.index = list(range(0,2016))[1975:2016] #row index 는 년도로. 2015년까지


#가로이미지(node가 행, years열) 로 저장할 때
data_clus = rel_freq_df.T #node가 행, years가 열
sns.set_style("whitegrid", {'axes.grid' : False}) #seaborn import하면 grid line생기므로, 지워줘야함
g = sns.clustermap(data_clus, method='average', row_cluster =True, col_cluster=False, figsize=(30,18),cmap='Reds')
g.ax_row_dendrogram.set_visible(False) #dendrogram  안보이게
plt.setp(g.ax_heatmap.yaxis.tick_left())
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(),rotation=0, Fontsize=32)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(),rotation=90, Fontsize=32)
plt.savefig('relative_clustermap_horizontal.png', dpi=100)

#세로이미지(years 행, nodes 열)로 저장할 때
sns.set_style("whitegrid", {'axes.grid' : False}) #seaborn import하면 grid line생기므로, 지워줘야함
g = sns.clustermap(rel_freq_df, method='average', row_cluster =False, col_cluster=True, figsize=(15,20), cmap='Reds')
g.ax_col_dendrogram.set_visible(False)
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(),rotation=0, Fontsize=20)
plt.setp(g.ax_heatmap.yaxis.tick_left())
plt.setp(g.ax_heatmap.xaxis.tick_top())
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(),rotation=90, fontsize=20)
plt.savefig('relative_clustermap.png', dpi=100)


##relative frequency matrix 만들고 저장하는 코드.였는데, 빈도순으로 나타났음. 밑의 clustermap으로 대신함

#plt.figure(figsize=(20,10))
#plt.set_cmap('Reds')
#plt.imshow(rel_freq[:,:])
##plt.matshow(rel_freq[:,:])
#plt.title('relative frequency') 
#plt.yticks(range(len(y_hist_2)), y_hist_2[:,0].astype(int),)
#plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90)
#plt.savefig('relative_frequency'+'.png',dpi=100) #relative frequency 파일 저장

## relative frequency clustermap 생성(sliding window를 먼저 적용)

WS=10 #window size. 
filt = np.ones(WS)/WS
y_hist_3 = np.zeros((len(y_hist_2)-WS-1, 2)) #sliding windown 적용했을 때 연도별 초록수 총합
y_hist_3[:,0]= range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)) 

for i in range(len(y_hist_2)-WS-1):
    WS_conv_hist = np.convolve(y_hist_2[:41,1],filt,'valid') 
    y_hist_3[:,1] = WS_conv_hist

freq_sum =  np.zeros((len(y_hist_2)-2, len(nodes))) #2015년까지
for y_i, y in enumerate(range(int(y_hist_2[0,0]),int(y_hist_2[-2,0]))):
    print(y_i, y)
    freq_sum_year = occur_matrix[occur_matrix[:,0]==y].sum(axis=0)
    freq_sum_year = freq_sum_year[1:]
    freq_sum[y_i,:]=freq_sum_year

freq_sum_slide = np.zeros((len(y_hist_2)-WS-1, len(nodes)))
for j in range(len(nodes)):
    WS_conv_freq= np.convolve(freq_sum[:,j],filt,'valid')
    freq_sum_slide[:,j]= WS_conv_freq

rel_freq_slide = np.zeros((len(y_hist_2)-WS-1, len(nodes)))
for y in range(len(y_hist_2)-WS-1):
    rel_freq_slide[y,:]= freq_sum_slide[y,:]/y_hist_3[y,1]

rel_freq_slide_df = pd.DataFrame(rel_freq_slide[:,:]) #rel_frq를 dataframe 형태로 전환. 2015년까지
nodes_list = nodes.ix[:,0].tolist() #node_list 만들어서 column index 만들기 위함
nodes_list[27] = 'Reticular formation' # medullary reticular formation 동의어에 reticular formation 있으므로 전환
nodes_list[1] = 'ACC' # anterior cinculate gyrus 약어 변환
nodes_list[3]= 'PAG' # periaqueductal gray 약어
nodes_list[9] = 'PFC' #Prefrontal cortex 약어
nodes_list[18] = 'LC' # locus coeruleus 약어
nodes_list[20] = 'NAc' # Nucleus accumbens 약어
nodes_list[24] = 'DLPFC' # Dorsolateral prefrontal cortex 약어
nodes_list[25] = 'CeA' #Central amygdaloid nucleus 약어
                       
rel_freq_slide_df.columns = nodes_list #rel_freq_df의 column 이름을 노드명으로
rel_freq_slide_df.index = list(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0])+1)) #row index 는 년도로. 2015년까지

#clustering matrix 만들고 저장
data_clus = rel_freq_slide_df.T #node가 행, years가 열
sns.set_style("whitegrid", {'axes.grid' : False}) #seaborn import하면 grid line생기므로, 지워줘야함
g = sns.clustermap(data_clus, method='weighted', row_cluster =True, col_cluster=False, figsize=(30,18),cmap='Reds')
g.ax_row_dendrogram.set_visible(False) #dendrogram  안보이게
plt.setp(g.ax_heatmap.yaxis.tick_left())
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(),rotation=0, Fontsize=32)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(),rotation=90, Fontsize=32)
plt.savefig('relative_clustermap_WS_weighted.png', dpi=100)


#바뀐 방식 사용하여 
#sliding window 적용한 rel_freq 이용한 그래프 그리기
plt.figure(figsize=(15,10))
plt.xticks(Fontsize=25)
plt.yticks(Fontsize=25)
# consistent high rel_freq group
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)),rel_freq_slide[:,2],'g') #rel_freq_slied[:,2] 가 brain stem
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)),rel_freq_slide[:,0],'g') #Thalamus
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)),rel_freq_slide[:,3],'g', label = 'consistent high') #PAG

# consistent low rel_freq group
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,15:27], 'grey') 
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,13], 'grey',label = 'consistent low') #hippocampus

#  rising rel_freq group
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,1], 'r') #ACC
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,4:7], 'r') #insula, somatosensory, postcentral gyrus
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,8:10], 'r') #amygdala, PFC
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,11], 'r',label = 'rising') # motor area

# falling rel_freq group
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,7], 'b') #medulla oblongata
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,10], 'b') #hypophysis
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,12], 'b') # hypothalamus
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,14], 'b') # midbrain
plt.plot(range(int(y_hist_3[0,0]), int(y_hist_3[-1,0]+1)), rel_freq_slide[:,27], 'b', label = 'falling') # reticular formation

plt.legend(loc='upper right',fontsize= 'x-large' )
plt.savefig('graph_ws_10.png', dpi=150)

#기존에는 rel_freq 연도별로 구해놓고 나서 나중에 sliding window 적용해서 그래프 그렸는데
#바뀐 코드에서는 rel_freq을 처음부터 sliding window 적용했으므로, 이 방식대로 분석할 것이라면 불필요한 코드


####    (from here) ##########

# slide window 사용하여 그래프를 좀 더 smooth하게 그릴 것. WS개의 연도의 상대빈도 합의 평균으로 구성된 매트릭스를 구성하고자 함. 
#WS=10 #window size. 
#filt = np.ones(WS)/WS
#rel_freq_slide = np.zeros((len(y_hist_2)-WS-1, len(nodes)))

#for i in range(len(nodes)):
#    WS_conv = np.convolve(rel_freq[:,i],filt,'valid') 
#    rel_freq_slide[:,i] = WS_conv

#print(rel_freq_slide)

# sliding window 적용하여 각 노드의 rel_freq 변화 그래프 그리기
#plt.figure(figsize=(15,10))
#plt.xticks(Fontsize=25)
#plt.yticks(Fontsize=25)
# consistent high rel_freq group
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1),rel_freq_slide[:,2],'g') #rel_freq_slied[:,2] 가 brain stem
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,0],'g') #Thalamus
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,3],'g', label = 'consistent high') #PAG

# consistent low rel_freq group
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,15:27], 'grey') 
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,13], 'grey',label = 'consistent low') #hippocampus

#  rising rel_freq group
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,1], 'r') #ACC
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,4:7], 'r') #insula, somatosensory, postcentral gyrus
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,8:10], 'r') #amygdala, PFC
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,11], 'r',label = 'rising') # motor area

# falling rel_freq group
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,7], 'b') #medulla oblongata
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,10], 'b') #hypophysis
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,12], 'b') # hypothalamus
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,14], 'b') # midbrain
#plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,27], 'b', label = 'falling') # reticular formation

#plt.legend(loc='upper right',fontsize= 'x-large' )
#plt.savefig('graph_ws_10.png', dpi=150)
    
                      
# legend 다는 방법 아직 못찾아서 위의 코드로 대신함
#A = [2,0,3] #지속적으로 높은 상대빈도
#B = [13,15,16,17,18,19,20,21,22,23,24,25,26] # 지속적으로 낮은 상대빈도
#C = [1,4,5,6,8,9,11] # 지날수록 높아지는 상대빈도
#D = [7,10,12,14,27] # 지날수록 낮아지는 상대빈도

#fig, ax = plt.subplots()
#plt.figure(figsize=(15,10))
#plt.xticks(Fontsize=25)
#plt.yticks(Fontsize=25)

# consistent high rel_freq group
#for i in range(len(A)):
#    plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,A[i]], 'g')
#    plt.hold(True)
# consistent low rel_freq group
#for i in range(len(B)):
#    plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,B[i]], 'grey')
#    plt.hold(True)
# rising rel_freq group
#for i in range(len(C)):
#    plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,C[i]], 'r')
#    plt.hold(True)
# falling rel_freq group   
#for i in range(len(D)):
#    plt.plot(range(int(y_hist_2[0,0]+WS/2), int(y_hist_2[-1,0]-WS/2)+1), rel_freq_slide[:,D[i]], 'b')
#    plt.hold(True)
    
#plt.legend(loc='upper right',fontsize= 'x-large' )
#plt.savefig('graph_ws_10.png', dpi=150)
    
#########    (till here) ##########


#occur_matrix_3: 노드 포함 안하는 초록 제거, 연도 제거
occur_matrix_3 = occur_matrix_2[np.sum(occur_matrix_2[:,1:], axis = 1) > 0,1:]
    
#전체 초록에 대한 co-occurence matrix  만들기
corr_total = np.dot(occur_matrix_3.T, occur_matrix_3)

#np.save('corr_total', corr_total) #corr_total 변수저장

#figure 약간 조정하고 이미지파일로 저장
plt.figure(figsize=(10,10))
plt.set_cmap('Reds')
plt.imshow(corr_total)
plt.title('corr_total')
plt.yticks(range(len(nodes.Node)),nodes_list)
plt.xticks(range(len(nodes.Node)),nodes_list, rotation = 90)
plt.savefig('corr_total'+'.png',dpi=100)

#기존 교수님이 만드신 코드
#plt.figure()
#plt.matshow(corr_total)
#plt.yticks(range(len(nodes.Node)),nodes.Node, Fontsize = 6)
#plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90, Fontsize = 6)
            
#연도별로 co-occurence matrix 구성(row, col: node 수, depth: 연도 수)



# rel_freq 100이상인 node들로 추린 것이 new_corr_total_100
corr_years = np.zeros((corr_total.shape[0], corr_total.shape[1],len(y_hist_2)))
for y_i, y in enumerate(y_hist_2[:,0]):
    occur_matrix_4 = occur_matrix_2[occur_matrix_2[:,0] == y,1:]
    corr_years[:,:,y_i] = np.dot(occur_matrix_4.T, occur_matrix_4)
    
    
#관심 연도(year of interest) 설정하여 co-occurence matrix 그리기

#yoi = 2013
#yoi_i = np.where(y_hist_2[:,0] == yoi)[0][0]

#plt.figure()
#plt.matshow(corr_years[:,:,yoi_i])
#plt.title(yoi)
#plt.yticks(range(len(nodes.Node)),nodes.Node, Fontsize = 6)
#plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90, Fontsize = 6)      

#node 순서를 빈도순이 아닌, 유사한 영역끼리 묶어서 나타내기 위해 순서 바꿈.
corr_years = corr_years[[22,17,19,9,24,5,6,11,23,0,12,10,13,8,25,1,4,16,26,21,20,2,14,3,7,18,27,15],:,:] 
corr_years = corr_years[:,[22,17,19,9,24,5,6,11,23,0,12,10,13,8,25,1,4,16,26,21,20,2,14,3,7,18,27,15],:]

#뇌영역 각 분류(cortex,diencephalon,limbic,bg,brain stem,cerebell) 간의 co-occur 보기 위해서
group = np.zeros((6,2)) #그룹수X2(start,stop) 행렬 만들고.
group[:,0] = [0,9,12,19,21,27] #시작하는 node 순서 설정
group[:,1] = [9,12,19,21,27,28] #종료하는 node 순서+1 로 stop 설정
group = pd.DataFrame(group) #DataFrame 형태로 바꿔서 indexing해서 사용할 예정
group.columns = ['start','stop'] #column 이름 설정
group.index = ['CTX','DIEN','Limbic','BG','BS','Cb'] #row index 설정

copy_corr_years = np.copy(corr_years) #copy하지 않으면 diagonal 0으로 채웠을 때, 원래 corr_year 변수도 0으로 채워지므로 카피
copy_corr_years = copy_corr_years[:,:,0:41] #2015년도까지

#그룹 간 co-occur 보기 위함. sliding window 사용하여 graph를 smooth하게 표현
WS=10 #window size 설정
filt = np.ones(WS)/WS
corr_years_slide = np.zeros((len(nodes), len(nodes), len(y_hist_2)-WS-1)) #2015년까지 할 거라 len(y_hist_2)-2 -WS+1

for i in range(len(nodes)):
    for j in range(len(nodes)):
        WS_conv = np.convolve(copy_corr_years[i,j,:],filt,'valid') 
        corr_years_slide[i,j,:] = WS_conv

#sliding window적용한 corr_years_slide에서 뇌영역 그룹 간의 변화 보기 위함. 영역마다 node개수 다르기 때문에
#각각 그룹마다 co-occur수 총합을 node개수로 나누어 노멀라이즈
#cerebellum은 해당 그룹에 하나 뿐이기에, 0처리
#같은 집단 내의 연결 중에서 self연결(diagonal부분에 해당) 0으로 만들고 계산해야 함.
#다른 영역집단 간은 총 노드 개수로 나누기만 하면 됨
corr_years_group = np.zeros((len(group.index),len(group.index),len(y_hist_2)-WS-1)) #그룹x그룹x년도 3차원 array
for k in range(len(y_hist_2)-WS-1):
    X = corr_years_slide[:,:,k] 
    for i in range(len(group.index)):
        for j in range(len(group.index)):
            r_start = group.start[i]
            r_stop = group.stop[i]
            c_start = group.start[j]
            c_stop = group.stop[j]
            Z = X[int(r_start):int(r_stop),int(c_start):int(c_stop)]
            Z_copy = np.copy(Z) #밑에서 대각선 0으로 만들 때 기존 Z에 영향주는 것 피하기 위해 카피
            if i == j == 5: #같은 분류에 속하는 뇌영역의 mean 계산, cbl인 경우
                corr_years_group[i,j,k] = 0
                print('a') #for문 제대로 돌아가는지 확인하기 위한 용도
            elif i==j:
                np.fill_diagonal(Z_copy,0) # 대각선 0으로 만듦
                U = np.sum(Z_copy)/(Z_copy.shape[0]*(Z_copy.shape[0]-1)) 
                corr_years_group[i,j,k] = U
                print('b')
            else : # 다른 분류에 속하는 뇌영역의 mean 계산
                U = np.sum(Z)/(Z.shape[0]*Z.shape[1])
                corr_years_group[i,j,k] = U
                print('c')
                
#각 년도의 co-occur합/가능한 모든 cooccur수로 나눈 값을 num_norm 변수에 할당하여 노멀라이즈에 사용. 
#normalize하지 않을 경우 년도 지날수록 논문 많기 때문에 co-occur도 증가하는 경향 나타날 것
for h in range(len(y_hist_2)-WS-1):
    num_norm = np.sum(corr_years_group[:,:,h])/(len(nodes)*(len(nodes)-1)) 
    corr_years_group[:,:,h] = corr_years_group[:,:,h]/num_norm 


#corr_years_group은 대각선 기준으로 대칭. 매트릭스의 upper triangle 숫자만 뽑아서 그래프 그릴 것
tri_upper_diag = corr_years_group[np.triu_indices(len(group.index))] # 대각선 포함 upper triangle 부분. indices(n)에서 n 은 정방행렬의 행(열) 길이

start_stop = pd.read_csv('start_stop.csv') #CTX,DIEN,Limbic,BG,BS,Cb그룹 시작과 끝 나누고 이름 달아놓은 파일
start_stop = np.array(start_stop) #데이터프레임을 array구조로 변환

#뇌 개별영역이 아닌, 영역그룹간 cooccurrence의 시간변화 그래프
plt.figure(figsize=(15,10))
col_list = [0,2,6,7,9, 11,14,18] # 색 다르게 할 line indexing. CTX-CTX, CTX-L, D-D, D-L, D,BS, L-L, L-Cb,BS-BS
num_plots = len(col_list)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0,0.9,num_plots)])

for i in range(tri_upper_diag.shape[0]-1):
    plt.yticks(Fontsize=28)
    plt.xticks(Fontsize=28)
    transformed_y = np.log(tri_upper_diag[i,:])
    transformed_y[np.isinf(transformed_y)] = 0 #0인 경우 log 취하면 -inf값 나오므로 0으로 설정
    if i in col_list:
        plt.plot(y_hist_2[5:37,0],transformed_y,label = start_stop[i][0]+'-'+start_stop[i][1]) #상위 몇 개만 색 다르게
    else:    
        plt.plot(y_hist_2[5:37,0],transformed_y,'grey',label = start_stop[i][0]+'-'+start_stop[i][1],linewidth=0.5) #나머진 회색
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
    plt.hold(True)
plt.savefig('corr_group.png', dpi=150)



# nodes_list는 비슷한 영역끼리 그룹 지어서, 약어로 정리한 파일. 그래프에서 node_name 순서대로 indexing 하기 위해 불러옴.
node_name =  pd.read_csv('nodes_list.csv')   

#musun 도움. co-occur matrix 1975-2016 연속으로 뽑는 코드
#2015까지 할거라면 range(len(y_hist_2)-2)를 입력
for i in range(len(y_hist_2)-2):
    yoi = i
    plt.figure(figsize=(30,30))
    plt.set_cmap('Reds')
    im = plt.imshow(corr_years[:,:,i])
    plt.colorbar(im,fraction=0.02,pad=0.01)
    plt.title(str(yoi+1975), Fontsize=30)
    plt.yticks(range(len(nodes.Node)),node_name.Node, Fontsize=28)
    plt.xticks(range(len(nodes.Node)),node_name.Node, rotation = 90, Fontsize=28)
    plt.savefig(str(+yoi+1975) +'.png', dpi=150) 

    
# 연대별 adjacency matrix
    
y_slice = [0,5,15,25,35,42] #1975,1980,1990,2000,2010년대(2015년까지)로 나누기 위함.
corr_years_1970s = np.sum(corr_years[:,:,y_slice[0]:y_slice[1]],axis=2) #1975-1979 (실제로는 1975가 75년 이전것들 합쳐놓은 것이므로 80년 이전 초록들)
corr_years_1980s = np.sum(corr_years[:,:,y_slice[1]:y_slice[2]],axis=2)  #1980년대
corr_years_1990s = np.sum(corr_years[:,:,y_slice[2]:y_slice[3]],axis=2) #1990년대
corr_years_2000s = np.sum(corr_years[:,:,y_slice[3]:y_slice[4]],axis=2) #2000년대
corr_years_2010s = np.sum(corr_years[:,:,y_slice[4]:y_slice[5]],axis=2)  #2010년대/ 2015ㅕ년까지

years_mat = [corr_years_1970s, corr_years_1980s, corr_years_1990s, corr_years_2000s, corr_years_2010s]
years_inc = np.arange(1970,2020,10)

for i in range(len(years_mat)):
    plt.figure(figsize=(20,20))
    plt.set_cmap('Reds')
    im = plt.imshow(years_mat[i])
    if i == 0:
        plt.title('before 1980s', Fontsize=30)
    else:
        plt.title(str(years_inc[i])+'s', Fontsize = 30)
    plt.colorbar(im,fraction=0.047,pad=0.01)
    plt.yticks(range(len(nodes.Node)),node_name.Node, Fontsize = 30)
    plt.xticks(range(len(nodes.Node)),node_name.Node, rotation = 90, Fontsize=30)
    plt.savefig(str(years_inc[i])+'s'+'.png', dpi=150)
    # 글씨 커서 잘려서 저장되므로 그냥 따로 이미지 저장했음
    



#edge density : (cor_number-diag_number)/2{전체노드*(노드-1)/2}
#thre = 50 #threshold 설정
#thre_year = (corr_years_2010s>thre)*1 #보고자하는 연대 입력해야함. threshold 이상에 해당하는 항목 1이 되는 array
#diag = np.sum(thre_year.diagonal())
#corr = np.sum(thre_year)
#den = (corr-diag)/len(nodes)/(len(nodes)-1) #edge density
#print(den) 

# 1970년대(1975년 이전) : thre = 0  (den : 0.06878)
# 1980년대 : thre = 3(den : 0.0714)
# 1990년대 :threshol = 9 (den : 0.0714)
# 2000년대 : threshold 34 (den : 0.07407) / 35(den: 0.0661)
# 2010년대 : threshold 35 (den :0.0714)

###wide range edge density 위한 코드(수정중)###
for k in range(len(y_hist_2)):
    K = corr_years_group[:,:,k]

years_interest = corr_years_1970s #연대 설정
X = np.zeros((51,len(nodes)+2))
X[:,0]=range(0,51)

for i in range(0,51):
    for j in range(0,len(nodes)):
        thre = i #threshold 설정
        thre_year = (years_interest>i)*1 #보고자하는 연대 입력해야함. threshold 이상에 해당하는 항목 1이 되는 array
        diag = np.sum(thre_year.diagonal()) #대각선에서 threshold 넘는 개수
        corr = np.sum(thre_year) #매트릭스에서 threshold 넘는co-occur 개수(diag 포함)
        den = (corr-diag)/len(nodes)/(len(nodes)-1) #edge density 구하는 과정
        #print(den) 
        np.fill_diagonal(thre_year, 0)
        sum_per_thre = np.sum(thre_year[:,j])
        X[i,2+j] = sum_per_thre
        X[i,1] = den

plt.figure(figsize=(30,30))
plt.set_cmap('Reds')
plt.imshow(X[:,2:])
plt.title('before 1980 wide threshold', Fontsize=20)
plt.yticks(range(len(X[:,0])),X[:,0], Fontsize=28)
plt.xticks(range(len(nodes.Node)),node_name.Node, rotation = 90, Fontsize=28)
plt.savefig('1970s_wide_thre'+'.png', dpi=100)


# newtwork 생성하고 toCytoscape 모듈 이용하여 csv파일 생성
# threshold 넘는 요소 boolean->int type array 생성 
#아직 co_thre값은 자동화하지 못함

co_thre = [0,3,9,34,35] # 연대 순서별 threshold값
#years_inc = np.arange(1970,2020,10) 
for i in range(len(years_mat)):
    corr_occur_thre = (years_mat[i]>co_thre[i])*1 
    years = np.copy(corr_occur_thre)
    np.fill_diagonal(years,0)
    years = pd.DataFrame(years)
    years.index = node_name.Node
    years.columns = node_name.Node
    G = nx.from_numpy_matrix(years.values)
    G = nx.relabel_nodes(G,dict(enumerate(years.columns)))
    cy.interactions(G,str(years_inc[i]))


#corr_years_연대 변수를 만든 이후에 node 별 diagonal의 빈도를 csv파일로 저장
#cytoscape에서 node size 조절하기 위한 table로 사용

for i in range(len(years_mat)):
    diag_years = years_mat[i].diagonal()
    diag_years = pd.DataFrame(diag_years)
    diag_years.index = node_name.Node
    diag_years.to_csv('diag'+str(years_inc[i])+'.csv')

