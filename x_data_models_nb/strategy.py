import pandas as pd 
import os 
from collections import Counter

#模型融合的不同策略及保留最大模型概率
def strategy4(label_list,prob_list):
    #优化考虑投票多的且只有一个的，否则按累计权重取最大，
    dict_for_wordcount = dict(Counter(label_list))
#     print("dict_for_wordcount",dict_for_wordcount)
    max_value= max(dict_for_wordcount.values())
#     print("max_value",max_value)
    max_list = []
    for i in dict_for_wordcount.keys():
        if dict_for_wordcount[i]==max_value:
            max_list.append(i)
        else:
            pass
#     print("max_list",max_list)
    if len(max_list)==1:
        target = max_list[0]
        prob = 0
        for i,j in enumerate(label_list):
            if target == j:
                prob += prob_list[i]       
#         print("target,prob",target,prob)    
        return target,prob
    else:
        dict0 ={}
        for i,j in enumerate(label_list):
            if j in dict0.keys():
                dict0[j] = dict0[j]+ prob_list[i]
            else:
                dict0[j] = prob_list[i]
                
#         print("dict0",dict0)
        
        dict0_sorted = sorted(dict0.items(),key=lambda d :d[1],reverse=True)
        #print("dict0_sorted",dict0_sorted)
        return dict0_sorted[0][0],dict0_sorted[0][1]
    
def strategy4_1(label_list,prob_list):
    #累加后取最大
    dict_for_wordcount = dict(Counter(label_list))
    #print("dict_for_wordcount",dict_for_wordcount)
    max_value= max(dict_for_wordcount.values())
    #print("max_value",max_value)
    max_list = []
    for i in dict_for_wordcount.keys():
        if dict_for_wordcount[i]==max_value:
            max_list.append(i)
        else:
            pass
     
    dict0 ={}
    for i,j in enumerate(label_list):
        if j in dict0.keys():
            dict0[j] = dict0[j]+ prob_list[i]
        else:
            dict0[j] = prob_list[i]
                
    #print("dict0",dict0)
        
    dict0_sorted = sorted(dict0.items(),key=lambda d :d[1],reverse=True)[:10]
#     print("dict0_sorted",dict0_sorted)
    #return dict0_sorted[0][0],dict0_sorted[0][1]
    return [i[0] for i in dict0_sorted]

def strategy5(label_list,prob_list):
    target,prob = strategy4(label_list,prob_list)
    max_list = []
    for i,j in enumerate(label_list):
        if target == j:
            max_list.append(prob_list[i])
        else:
            pass
    #print("max_list",max_list)
    return target,max(max_list)

def strategy5_1(label_list,prob_list):
    target,prob = strategy4_1(label_list,prob_list)
#     print(target,prob)
    max_list = []
    for i,j in enumerate(label_list):
        if target == j:
            max_list.append(prob_list[i])
        else:
            pass
    #print("max_list",max_list)
    return target,max(max_list)

def strategy5_1_list(label_list,prob_list):
    target_list = strategy4_1(label_list,prob_list)
    target_prob_list = []
    for k in target_list:
    
        max_list = []
        for i,j in enumerate(label_list):
            if k == j:
                max_list.append(prob_list[i])
            else:
                pass
        #print("max_list",max_list)
        target_prob_list.append(max(max_list))
    return target_list,target_prob_list

#数据拼接
def eval_data(file_number,file_name,base_data_path,file_list,weight=False):
    
    data = pd.read_csv(os.path.join(base_data_path,file_list[file_number]),sep='\t',header=-1)
    data.columns= ['target']
    data['target_eval']=data['target'].apply(lambda s:eval(s))
    data['%s_label'%file_name]=data['target_eval'].apply(lambda s:s[0])
    data['%s_prob'%file_name]=data['target_eval'].apply(lambda s:s[1])
    if weight==True:
        weight = weight_list_dict[model_weight[file_name]]
        data['%s_prob'%file_name]=data['%s_prob'%file_name].apply(lambda s:[i*weight for i in s])
    return data[['%s_label'%file_name,'%s_prob'%file_name]]

#数据拼接
def list_data(file_number,file_name,base_data_path,file_list,weight=False):
    data = pd.read_csv(os.path.join(base_data_path,file_list[file_number]),sep='\t',header=-1)
    data['%s_label'%file_name]= data[[i for i in data.columns if i%2==0]].apply(lambda s:tuple(s),axis=1)
    data['%s_prob'%file_name]= data[[i for i in data.columns[:-1] if i%2==1]].apply(lambda s:tuple(s),axis=1)
    if weight==True:
        weight = weight_list_dict[model_weight[file_name]]
        data['%s_prob'%file_name]=data['%s_prob'%file_name].apply(lambda s:[i*weight for i in s])
    return data[['%s_label'%file_name,'%s_prob'%file_name]]

#把元素展开
def flatten(elements,class_type='label'):
    try:
        for i,j in enumerate(elements):
            if i==0:
                elements_final = list(j)
            else:
                elements_final.extend(j)
        if class_type=='label':
            return [str(i) for i in elements_final]
        else:
            return elements_final
    except:
        return 'error'