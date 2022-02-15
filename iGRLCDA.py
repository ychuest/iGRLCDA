
from sklearn.model_selection import KFold

Positive = pd.read_csv('./data/PositiveNum.csv',header=None)

i=0
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for train_index, test_index in kf.split(Positive):
    X_train, X_test = Positive.iloc[train_index], Positive.iloc[test_index]
    X_train.to_csv('./data/train'+str(i)+'.txt',sep='\t',header=0,index=0)
    i+=1



import scipy.sparse as sp
def load_data(adj,node_features):
    features = sp.csr_matrix(node_features, dtype=np.float32)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias 
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'




import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x, self.dropout, training = self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training = self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x2, dim = 1), x1



import numpy as np
import pandas as pd
def load_file_as_Adj_matrix(filename):
    import scipy.sparse as sp
    AlledgeTrain = pd.read_csv(filename,header=None)
    relation_matrix = np.zeros((len(AllNode),len(AllNode)))
    for i, j in np.array(AlledgeTrain):
        lnc, mi = int(i), int(j)
        relation_matrix[lnc, mi] = 1
    Adj = sp.csr_matrix(relation_matrix, dtype=np.float32)
    return Adj




AllNode = pd.read_csv('./data/AllNode.csv',header=None)
Adj = load_file_as_Adj_matrix('./data/PositiveNum.csv')
features = pd.read_csv('./data/AllNodeAttribute.csv', header = None)
features = features.iloc[:,1:]
adj, train_features = load_data(Adj,features)

dropout = 0.02
in_size = features.shape[1]
hi_size = 64
name = locals()

model = GCN(nfeat=train_features.shape[1],
        nhid=hi_size,
        nclass= 64,
        dropout=dropout)
model.train()
global Emdebding_train, output
output, Emdebding_train = model(train_features, adj)
Emdebding_GCN = pd.DataFrame(Emdebding_train.detach().numpy())
Emdebding_GCN.to_csv('./data/Emdebding_GCN.csv', header=None,index=False)



def NegativeGenerate(LncDisease, AllRNA,AllDisease):
    import random
    NegativeSample = []
    counterN = 0
    while counterN < len(LncDisease):
        counterR = random.randint(0, len(AllRNA) - 1)
        counterD = random.randint(0, len(AllDisease) - 1)
        DiseaseAndRnaPair = []
        DiseaseAndRnaPair.append(AllRNA[counterR])
        DiseaseAndRnaPair.append(AllDisease[counterD])
        flag1 = 0
        counter = 0
        while counter < len(LncDisease):
            if DiseaseAndRnaPair == LncDisease[counter]:
                flag1 = 1
                break
            counter = counter + 1
        if flag1 == 1:
            continue
        flag2 = 0
        counter1 = 0
        while counter1 < len(NegativeSample):
            if DiseaseAndRnaPair == NegativeSample[counter1]:
                flag2 = 1
                break
            counter1 = counter1 + 1
        if flag2 == 1:
            continue
        if (flag1 == 0 & flag2 == 0):
            NamePair = []
            NamePair.append(AllRNA[counterR])
            NamePair.append(AllDisease[counterD])
            NegativeSample.append(NamePair)
            counterN = counterN + 1
    return NegativeSample
DDI = pd.read_csv('./data/PositiveNum.csv',header=None)
AllNode = pd.read_csv('./data/AllNode.csv', header=0,names=['id','name'])
Dr = AllNode.iloc[:660]
Di = AllNode.iloc[660:760]
NegativeSample = NegativeGenerate(DDI.values.tolist(),Dr['id'].values.tolist(),Di['id'].values.tolist())
NegativeSample = pd.DataFrame(NegativeSample)
NegativeSample.to_csv('./data/NegativeSample1.csv', header=None,index=False)
NegativeSample


import math
import numpy as np
def MyConfusionMatrix(y_real,y_predict): 
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1] 
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1 = 2 * (((TP/(TP+FP))*(TP/(TP+FN))) / ((TP/(TP+FP))+(TP/(TP+FN))))
    
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))
    print('F1:', round(F1, 4))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    Result.append(round(F1, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return

def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1
    return result

def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)

        return

    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)


    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)


    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1


    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return
def MyEnlarge_PR(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)

        return

    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    xp = np.linspace(x1 + width2, x0,num=20)
    yp = np.linspace(y1 + height2, y0,num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)


    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1


    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot( XDottedLine, YDottedLine,  color=color, lw=thickness, alpha=1)
    return

import csv
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return




import pandas as pd
import numpy as np
Positive = pd.read_csv('./data/PositiveNum.csv',header=None)
Negative = pd.read_csv('./data/NegativeSample1.csv',header=None) 
Attribute = pd.read_csv('./data/Emdebding_GCN.csv',header = None)
Embedding_Node2vec = pd.read_csv('./data/CDA_GF.txt', sep=' ',header=None,skiprows=1)
Embedding_Node2vec = Embedding_Node2vec.sort_values(0,ascending=True).dropna(axis=1)
Embedding_Node2vec.set_index([0], inplace=True)
Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive,Negative]).reset_index(drop=True)
X = pd.concat([pd.concat([Attribute.loc[result[0].values.tolist()],Embedding_Node2vec.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
    pd.concat([Attribute.loc[result[1].values.tolist()],Embedding_Node2vec.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)
Y = result[2]




import pandas as pd
import numpy as np
Positive = pd.read_csv('./data/PositiveNum.csv',header=None)
Negative = pd.read_csv('./data/NegativeSample1.csv',header=None) 
Attribute = pd.read_csv('./data/Emdebding_GCN.csv',header = None)
Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive,Negative]).reset_index(drop=True)
X = pd.concat([Attribute.loc[result[0].values.tolist()].reset_index(drop=True),Attribute.loc[result[1].values.tolist()].reset_index(drop=True)],axis=1)
Y = result[2]




import pandas as pd
import numpy as np
Positive = pd.read_csv('./data/PositiveNum.csv',header=None)
Negative = pd.read_csv('./data/NegativeSample1.csv',header=None) 
Embedding_Node2vec = pd.read_csv('./data/CDA_GF.txt', sep=' ',header=None,skiprows=1)
Embedding_Node2vec = Embedding_Node2vec.sort_values(0,ascending=True).dropna(axis=1)
Embedding_Node2vec.set_index([0], inplace=True)
Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive,Negative]).reset_index(drop=True)
X = pd.concat([Embedding_Node2vec.loc[result[0].values.tolist()].reset_index(drop=True),Embedding_Node2vec.loc[result[1].values.tolist()].reset_index(drop=True)],axis=1)
Y = result[2]




from sklearn.model_selection import  StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from scipy import interp
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


print("5-fold")
i=0
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)
colorlist = ['firebrick', 'magenta', 'royalblue', 'limegreen', 'blueviolet', 'black']
AllResult = []
skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):


    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = RandomForestClassifier(n_estimators=999,n_jobs=-1)
    model.fit(np.array(X_train), np.array(Y_train))
    y_score0 = model.predict(np.array(X_test))
    y_score_RandomF = model.predict_proba(np.array(X_test))
    RandomF_data = pd.DataFrame(np.vstack([Y_test, y_score_RandomF[:,1]]).T)
    RandomF_data.to_csv('./results/predict/RandomF_N_'+ str(i)+ 'Prob.csv', header = False, index = False)
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1.5,alpha=0.8,color=colorlist[i],
             label='ROC fold %d (AUC=%0.4f)'% (i,roc_auc))
    print("---------------------------------------------")
    print("fold = ", i)
    print("---------------------------------------------\n")
    Result = MyConfusionMatrix(Y_test, y_score0)
    PR=[]
    average_precision = average_precision_score(Y_test, y_score_RandomF[:,1])
    PR.append(round(average_precision, 4))
    AllResult.append(Result+PR)
    AllResult[i].append(roc_auc)
    i+=1
    
MyAverage(AllResult)
MyNew = MyStd(AllResult)
df = pd.DataFrame(data = MyNew)
df.to_csv('./results/evaluate/RandomF_N_.csv', encoding='utf-8',header=None,index=False)

plt.plot([0,1],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color=colorlist[i],label=r'Mean ROC (AUC=%0.4f)'%mean_auc,lw=2,alpha=1)
std_tpr=np.std(tprs,axis=0)

plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('./results/image/RandomF_N_ROC.svg')
plt.show() 




from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy import interp
import numpy as np

mean_fpr = np.linspace(0, 1, 1000)
i = 0
colorlist = ['firebrick', 'magenta', 'royalblue', 'limegreen', 'blueviolet', 'black']

Ps = []
RPs = []
mean_R = np.linspace(0, 1, 1000)

for i in range(5):
    RAPNameProb = './results/predict/RandomF_GN_'+ str(i)+ 'Prob.csv'
    RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)
    
    Real = RealAndPredictionProb[0]
    PredictionProb = RealAndPredictionProb[1]

    average_precision = average_precision_score(Real, PredictionProb)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb)

    Ps.append(interp(mean_R, precision, recall))
    RPs.append(average_precision)


    plt.plot(recall, precision, lw=1.5, alpha=0.8, color=colorlist[i],
             label='fold %d (AUPR = %0.4f)' % (i, average_precision))
    print('average_precision', average_precision)

mean_P = np.mean(Ps, axis=0)
mean_RPs = np.mean(RPs, axis=0)
plt.plot(mean_P, mean_R, color='black',
         label=r'Mean (AUPR = %0.4f)' % (mean_RPs),
         lw=2, alpha=1)
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.plot([1, 0], [0, 1], color='rosybrown', lw=2, linestyle='--')
plt.legend(loc='best')
plt.savefig('./results/image/RandomF_GN_PR.svg')
plt.show()



from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp

print("5-fold")
i=0
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)
colorlist = ['firebrick', 'magenta', 'royalblue', 'limegreen', 'blueviolet', 'black']
AllResult = []
skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]


    best_SVC = SVC(probability=True)
    print('开始训练')

    best_SVC.fit(np.array(X_train), np.array(Y_train))
    y_score0 = best_SVC.predict(np.array(X_test))
    y_score_SVC = best_SVC.predict_proba(np.array(X_test))
    print(confusion_matrix(Y_test, y_score0))
    
    dd = np.vstack([Y_test, y_score_SVC[:,1]]).T
    SVC_data = pd.DataFrame(dd)
    SVC_data.to_csv('./results/predict/SVM_'+ str(i)+ 'Prob.csv', header = False, index = False)
    
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_SVC[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1.5,alpha=0.8,color=colorlist[i],
             label='ROC fold %d(AUC=%0.4f)'% (i,roc_auc))
    print("---------------------------------------------\n")
    print("fold = ", i)
    print("---------------------------------------------\n")
    Result = MyConfusionMatrix(Y_test, y_score0)
    AllResult.append(Result)
    AllResult[i].append(roc_auc)
    i +=1

MyAverage(AllResult)
MyNew = MyStd(AllResult)
df = pd.DataFrame(data = MyNew)
df.to_csv('./results/evaluate/SVM_5-fold.csv', encoding='utf-8',header=None,index=False)

plt.plot([0,1],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color=colorlist[i],label=r'Mean ROC (AUC=%0.4f)'%mean_auc,lw=2,alpha=1)
std_tpr=np.std(tprs,axis=0)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('./results/image/SVM_ROC.svg')
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
import time

print("5-fold")
i=0
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)
colorlist = ['firebrick', 'magenta', 'royalblue', 'limegreen', 'blueviolet', 'black']
AllResult = []
skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    best_Logistic = LogisticRegression(n_jobs=-1)
    best_Logistic.fit(np.array(X_train), np.array(Y_train))

    y_score0 = best_Logistic.predict(np.array(X_test))
    y_score_Logistic = best_Logistic.predict_proba(np.array(X_test))
    print(confusion_matrix(Y_test, y_score0))
    
    dd = np.vstack([Y_test, y_score_Logistic[:,1]]).T
    Logistic_data = pd.DataFrame(dd)
    Logistic_data.to_csv('./results/predict/Logistic_'+ str(i)+ 'Prob.csv', header = False, index = False)
    
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_Logistic[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1.5,alpha=0.8,color=colorlist[i],
             label='ROC fold %d(AUC=%0.4f)'% (i,roc_auc))
    print("---------------------------------------------\n")
    print("fold = ", i)
    print("---------------------------------------------\n")
    Result = MyConfusionMatrix(Y_test, y_score0)
    AllResult.append(Result)
    AllResult[i].append(roc_auc)
    i +=1

MyAverage(AllResult)
MyNew = MyStd(AllResult)
df = pd.DataFrame(data = MyNew)
df.to_csv('./results/evaluate/Logistic_5-fold.csv', encoding='utf-8',header=None,index=False)

plt.plot([0,1],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color=colorlist[i],label=r'Mean ROC (AUC=%0.4f)'%mean_auc,lw=2,alpha=1)
std_tpr=np.std(tprs,axis=0)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('./results/image/Logistic_ROC.svg')
plt.show()



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import time



print("5-fold")
i=0
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)
colorlist = ['firebrick', 'magenta', 'royalblue', 'limegreen', 'blueviolet', 'black']
AllResult = []
skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    best_KNN = KNeighborsClassifier(n_jobs=-1)
    best_KNN.fit(np.array(X_train), np.array(Y_train))

    y_score0 = best_KNN.predict(np.array(X_test))
    y_score_KNN = best_KNN.predict_proba(np.array(X_test))
    print(confusion_matrix(Y_test, y_score0))
    
    dd = np.vstack([Y_test, y_score_KNN[:,1]]).T
    KNN_data = pd.DataFrame(dd)
    KNN_data.to_csv('./results/predict/KNN_'+ str(i)+ 'Prob.csv', header = False, index = False)
    
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_KNN[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1.5,alpha=0.8,color=colorlist[i],
             label='ROC fold %d(AUC=%0.4f)'% (i,roc_auc))
    print("---------------------------------------------\n")
    print("fold = ", i)
    print("---------------------------------------------\n")
    Result = MyConfusionMatrix(Y_test, y_score0)
    AllResult.append(Result)
    AllResult[i].append(roc_auc)
    i +=1

MyAverage(AllResult)
MyNew = MyStd(AllResult)
df = pd.DataFrame(data = MyNew)
df.to_csv('./results/evaluate/KNN_5-fold.csv', encoding='utf-8',header=None,index=False)


plt.plot([0,1],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color=colorlist[i],label=r'Mean ROC (AUC=%0.4f)'%mean_auc,lw=2,alpha=1)
std_tpr=np.std(tprs,axis=0)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('./results/image/KNN_ROC.svg')
plt.show()   


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
import time

print("5-fold")
i=0
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)
colorlist = ['firebrick', 'magenta', 'royalblue', 'limegreen', 'blueviolet', 'black']
AllResult = []
kf = KFold(n_splits=5, random_state=0, shuffle=True)
skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    model = GaussianNB()
    model.fit(np.array(X_train), np.array(Y_train))

    y_score0 = model.predict(np.array(X_test))
    y_score1 = model.predict_proba(np.array(X_test))

    print(confusion_matrix(Y_test, y_score0))
    
    dd = np.vstack([Y_test, y_score1[:,1]]).T
    Predict_data = pd.DataFrame(dd)
    Predict_data.to_csv('./results/predict/GaussianNB_'+ str(i)+ 'Prob.csv', header = False, index = False)
    
    fpr,tpr,thresholds=roc_curve(Y_test,y_score1[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1.5,alpha=0.8,color=colorlist[i],
             label='ROC fold %d(AUC=%0.4f)'% (i,roc_auc))
    print("---------------------------------------------\n")
    print("fold = ", i)
    print("---------------------------------------------\n")
    Result = MyConfusionMatrix(Y_test, y_score0)
    AllResult.append(Result)
    AllResult[i].append(roc_auc)
    i +=1

MyAverage(AllResult)
MyNew = MyStd(AllResult)
df = pd.DataFrame(data = MyNew)
df.to_csv('./results/evaluate/GaussianNB_5-fold.csv', encoding='utf-8',header=None,index=False)


plt.plot([0,1],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color=colorlist[i],label=r'Mean ROC (AUC=%0.4f)'%mean_auc,lw=2,alpha=1)
std_tpr=np.std(tprs,axis=0)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('./results/image/GaussianNB_ROC.svg')
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
import time
import numpy as np
import pandas as pd

colorlist = ['red', 'gold', 'purple', 'limegreen', 'darkblue', 'black']
for i in range(1):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/GaussianNB_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,color=colorlist[4],label=r'GaussianNB (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb =  './results/predict/Logistic_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc=auc(fpr,tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,linestyle='-',color=colorlist[2],label=r'LR (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)  

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb =  './results/predict/KNN_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,linestyle='-',color=colorlist[1],label=r'KNN (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)  

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):

        RAPNameProb =  './results/predict/RandomF_GN_' + str(j) + 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,linestyle='-',color=colorlist[0],label=r'iGRLCDA (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)  


plt.plot([0,1],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.legend(loc='best')  
plt.savefig('./results/image/CompareClassifier_ROC.svg')
plt.show()    



from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy import interp
import numpy as np
import pandas as pd
colorlist = ['red', 'gold', 'purple', 'limegreen', 'darkblue', 'black']

for i in range(1): 
    Ps = []
    RPs = []
    mean_R = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/GaussianNB_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)
    
    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,color=colorlist[4],label=r'GaussianNB (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)

    Ps = []
    RPs = []
    mean_R = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb =  './results/predict/Logistic_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)

    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,linestyle='-',color=colorlist[2],label=r'LR (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)

    Ps = []
    RPs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb =  './results/predict/KNN_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)

    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,linestyle='-',color=colorlist[1],label=r'KNN (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)

    Ps = []
    RPs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb =  './results/predict/RandomF_GN_' + str(j) + 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)

    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,linestyle='-',color=colorlist[0],label=r'iGRLCDA (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)

plt.plot([1,0],[0,1],linestyle='--',lw=2,color='rosybrown',alpha=0.8)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.legend(loc='best')  
plt.savefig('./results/image/CompareClassifier_PR.svg')
plt.show() 



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
import time

colorlist = ['red', 'gold', 'purple', 'limegreen', 'darkblue', 'black']
for i in range(1):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/RandomF_N_' + str(j) + 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,color=colorlist[2],label=r'iGRLCDA-N (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)
    MyEnlarge(0.1, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_fpr, tprs[j], 2, colorlist[2])
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/RandomF_G_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,linestyle='-',color=colorlist[3],label=r'iGRLCDA-G (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)
    MyEnlarge(0.1, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_fpr, tprs[j], 2, colorlist[3])
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/RandomF_GN_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        fpr, tpr, thresholds = roc_curve(Real,PredictionProb)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0


    mean_tpr = np.mean(tprs, axis=0)
    mean_auc=auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr,mean_tpr,linestyle='-',color=colorlist[0],label=r'HINGRL (AUC=%0.4f)'%mean_auc,lw=1.5,alpha=1)  
    MyEnlarge(0.1, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_fpr, tprs[j], 2, colorlist[0])

plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.legend(bbox_to_anchor=(0.45, 0.7))   
plt.savefig('./results/image/CompareFeature_ROC.svg')
plt.show()    



from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy import interp
import numpy as np

colorlist = ['red', 'gold', 'purple', 'limegreen', 'darkblue', 'black']
mean_fpr = np.linspace(0, 1, 1000)
for i in range(1): 
    Ps = []
    RPs = []
    mean_R = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/RandomF_N_' + str(j) + 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)
    

    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,color=colorlist[2],label=r'iGRLCDA-N (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)
    MyEnlarge_PR(0.7, 0.65, 0.25, 0.25, 0, 0, 2, mean_fpr, Ps[j], 2, colorlist[2])
    
    Ps = []
    RPs = []
    mean_R = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/RandomF_G_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)

    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,linestyle='-',color=colorlist[3],label=r'iGRLCDA-G (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)
    MyEnlarge_PR(0.7, 0.65, 0.25, 0.25, 0, 0, 2, mean_fpr, Ps[j], 2, colorlist[3])
    
    Ps = []
    RPs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(5):
        RAPNameProb = './results/predict/RandomF_GN_'+ str(j)+ 'Prob.csv'
        RealAndPredictionProb = pd.read_csv(RAPNameProb, header=None)

        Real = RealAndPredictionProb[0]
        PredictionProb = RealAndPredictionProb[1]

        average_precision = average_precision_score(Real, PredictionProb)
        precision, recall, _ = precision_recall_curve(Real, PredictionProb)

        Ps.append(interp(mean_R, precision, recall))
        RPs.append(average_precision)

    mean_P = np.mean(Ps, axis=0)
    mean_RPs = np.mean(RPs, axis=0)
    plt.plot(mean_P, mean_R,linestyle='-',color=colorlist[0],label=r'iGRLCDA (AUPR=%0.4f)'%mean_RPs,lw=1.5,alpha=1)
    MyEnlarge_PR(0.7, 0.65, 0.25, 0.25, 0, 0, 2, mean_fpr, Ps[j], 2, colorlist[0])
    

plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.legend(bbox_to_anchor=(0.55, 0.7))   
plt.savefig('./results/image/CompareFeature_PR.svg')
plt.show() 





