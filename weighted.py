import numpy as np
import pandas as pd 
import random
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



data = pd.read_csv("diabetic_data.csv", index_col=0)
labels = data.keys()

data.loc[data['readmitted']=='NO', 'readmitted'] = 0
data.loc[data['readmitted']=='<30', 'readmitted'] = 1
data.loc[data['readmitted']=='>30', 'readmitted'] = 0
data['readmitted'] = pd.to_numeric(data['readmitted'])

# read = data[data.readmitted == 1]
# notread=data[data.readmitted ==0]
# notread=notread.reset_index(drop=True)
# lengthnotread=notread.shape

# data=read.append(notread.head(11357),ignore_index=True)
# data=data.sample(frac=1)
data=data.reset_index(drop=True)
print(data.shape)
nominialCats=['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',]
onehots=pd.DataFrame()
for cat in nominialCats:
    cat = pd.get_dummies(data[cat])
    onehots=pd.concat([onehots,cat],axis=1)
# races= data.drop_duplicates(subset = ['race'])
# races = races['race']
data.loc[data['race'] == '?','race' ]= 0
data.loc[data['race'] == 'Caucasian', 'race'] = 1
data.loc[data['race'] == 'AfricanAmerican', 'race'] = 2
data.loc[data['race'] == 'Other', 'race'] = 3
data.loc[data['race'] == 'Asian', 'race'] = 4
data.loc[data['race'] == 'Hispanic', 'race'] = 5
data['race']=pd.to_numeric(data['race'])
# races = pd.get_dummies(data['race'])

genders = data.drop_duplicates(subset = ['gender'])
data.loc[data['gender'] == 'Unknown/Invalid','gender' ]= 0
data.loc[data['gender'] == 'Female','gender' ]= 1
data.loc[data['gender'] == 'Male','gender' ]= 2
data['gender']=pd.to_numeric(data['gender'])
# genders = pd.get_dummies(data['gender'])

ages= data.drop_duplicates(subset = ['age'])
data.loc[data['age'] == '[0-10)','age' ]= 1
data.loc[data['age'] == '[10-20)','age' ]= 2
data.loc[data['age'] == '[20-30)','age' ]= 3
data.loc[data['age'] == '[30-40)','age' ]= 4
data.loc[data['age'] == '[40-50)','age' ]= 5
data.loc[data['age'] == '[50-60)','age' ]= 6
data.loc[data['age'] == '[60-70)','age' ]= 7
data.loc[data['age'] == '[70-80)','age' ]= 8
data.loc[data['age'] == '[80-90)','age' ]= 9
data.loc[data['age'] == '[90-100)','age' ]= 10
data['age']=pd.to_numeric(data['age'])
# ages = pd.get_dummies(data['age'])
# data = pd.concat([data,races,genders,ages],axis=1)
# data=data.drop(columns=['race','age','gender'])

# weightNAcount = len(data.loc[data['weight']=='?'])
# weightpercentblanl =weightNAcount/101766
# print(weightpercentblanl)
# print(len(data.loc[data['payer_code']=='?'])/101766)
# print(len(data.loc[data['medical_specialty']=='?'])/101766)
data = data.drop(columns = ['weight','payer_code','medical_specialty'])

from sklearn.preprocessing import minmax_scale

numLabspro=np.array(data['num_lab_procedures'].copy())
data['num_lab_procedures']=minmax_scale(data['num_lab_procedures'])
data['num_procedures']=minmax_scale(data['num_procedures'])
data['num_medications']=minmax_scale(data['num_medications'])
data['number_outpatient']=minmax_scale(data['number_outpatient'])
data['number_emergency']=minmax_scale(data['number_emergency'])
data['number_inpatient']=minmax_scale(data['number_inpatient'])
data['number_diagnoses']=minmax_scale(data['number_diagnoses'])
data['time_in_hospital']=minmax_scale(data['time_in_hospital'])

possibllyRedundant=['num_procedures','num_lab_procedures','num_medications','number_outpatient','number_inpatient',
'number_diagnoses','num_lab_procedures','number_emergency']
# RS=[]
# for fname in possibllyRedundant:
#     rs=[]
#     for sname in possibllyRedundant:
#         r=np.corrcoef(data[fname],data[sname])
#         rs.append(r[0,1])
#     RS.append(rs)
# rDF=pd.DataFrame(RS,columns=possibllyRedundant,index=possibllyRedundant)
# print(rDF)
data = data.drop(columns= ['num_medications'])

diag3s = data.drop_duplicates(subset = ['diag_3'])
diag1s = data.drop_duplicates(subset = ['diag_1'])
data.loc[data['diag_1']>'999','diag_1']= np.nan
data['diag_1']=pd.to_numeric(data['diag_1'])
avgDiag1 = data['diag_1'].mean(skipna=True)
data['diag_1']=data['diag_1'].replace(np.nan,avgDiag1)

diag2s = data.drop_duplicates(subset = ['diag_2'])
data.loc[data['diag_2']>'999','diag_2']= np.nan
data['diag_2']=pd.to_numeric(data['diag_2'])
avgDiag2 = data['diag_2'].mean(skipna=True)
data['diag_2']=data['diag_2'].replace(np.nan,avgDiag2)

diag3s = data.drop_duplicates(subset = ['diag_3'])
data.loc[data['diag_3']>'999','diag_3']= np.nan
data['diag_3']=pd.to_numeric(data['diag_3'])
avgDiag3 = data['diag_3'].mean(skipna=True)
data['diag_3']=data['diag_3'].replace(np.nan,avgDiag3)

data['diag_1']=minmax_scale(data['diag_1'])
data['diag_2']=minmax_scale(data['diag_2'])
data['diag_3']=minmax_scale(data['diag_3'])

maxglu = data.drop_duplicates(subset = ['max_glu_serum'])
data.loc[data['max_glu_serum']=='None', 'max_glu_serum'] = 0
data.loc[data['max_glu_serum']=='Norm', 'max_glu_serum'] = 1
data.loc[data['max_glu_serum']=='>200', 'max_glu_serum'] = 2
data.loc[data['max_glu_serum']=='>300', 'max_glu_serum'] = 3
data['max_glu_serum']=pd.to_numeric(data['max_glu_serum'])

a1c = data.drop_duplicates(subset = ['A1Cresult'])
data.loc[data['A1Cresult']=='None','A1Cresult'] = 0
data.loc[data['A1Cresult']=='Norm','A1Cresult'] = 1
data.loc[data['A1Cresult']=='>7','A1Cresult'] = 2
data.loc[data['A1Cresult']=='>8','A1Cresult'] = 3
data['A1Cresult']=pd.to_numeric(data['A1Cresult'])

metformins =  data.drop_duplicates(subset = ['metformin'])
data.loc[data['metformin']=='No', 'metformin'] = 0
data.loc[data['metformin']=='Steady', 'metformin'] = 1
data.loc[data['metformin']=='Up', 'metformin'] = 2
data.loc[data['metformin']=='Down', 'metformin'] = 3
data['metformin'] = pd.to_numeric(data['metformin'])

wanted= ['repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide']
for want in wanted:
    x= data.drop_duplicates(subset = [want])
    # print(x)
    # print(want)

data.loc[data['repaglinide']=='No', 'repaglinide'] = 0
data.loc[data['repaglinide']=='Steady', 'repaglinide'] = 1
data.loc[data['repaglinide']=='Up', 'repaglinide'] = 2
data.loc[data['repaglinide']=='Down', 'repaglinide'] = 3
data['repaglinide'] = pd.to_numeric(data['repaglinide'])

data.loc[data['acetohexamide']=='No', 'acetohexamide'] = 0
data.loc[data['acetohexamide']=='Steady', 'acetohexamide'] = 1
data['acetohexamide'] = pd.to_numeric(data['acetohexamide'])

data.loc[data['glipizide']=='No', 'glipizide'] = 0
data.loc[data['glipizide']=='Steady', 'glipizide'] = 1
data.loc[data['glipizide']=='Up', 'glipizide'] = 2
data.loc[data['glipizide']=='Down', 'glipizide'] = 3
data['glipizide'] = pd.to_numeric(data['glipizide'])

data.loc[data['glyburide']=='No', 'glyburide'] = 0
data.loc[data['glyburide']=='Steady', 'glyburide'] = 1
data.loc[data['glyburide']=='Up', 'glyburide'] = 2
data.loc[data['glyburide']=='Down', 'glyburide'] = 3
data['glyburide'] = pd.to_numeric(data['glyburide'])

wanted2 = ['acetohexamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','insulin','glyburide-metformin',
'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',
'metformin-pioglitazone','change','diabetesMed']
for want in wanted2:
    x= data.drop_duplicates(subset = [want])
    # print(x)
    # print(want)

data = data.drop(columns=['examide','citoglipton'])

data.loc[data['insulin']=='No', 'insulin'] = 0
data.loc[data['insulin']=='Steady', 'insulin'] = 1
data.loc[data['insulin']=='Up', 'insulin'] = 2
data.loc[data['insulin']=='Down', 'insulin'] = 3
data['insulin'] = pd.to_numeric(data['insulin'])

data.loc[data['change']=='No', 'change'] = 0
data.loc[data['change']=='Ch', 'change'] = 1
data['change'] = pd.to_numeric(data['change'])

data.loc[data['diabetesMed']=='No', 'diabetesMed'] = 0
data.loc[data['diabetesMed']=='Yes', 'diabetesMed'] = 1
data['diabetesMed'] = pd.to_numeric(data['diabetesMed'])





y=data['readmitted'].copy()
y=y.reset_index(drop=True)
# y=y.drop(columns='encounter_id')
X = data.drop(columns=['readmitted','patient_nbr','nateglinide','chlorpropamide','glimepiride','tolbutamide',
'pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
'tolazamide','glyburide-metformin',
'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',
'metformin-pioglitazone'])

c_names= X.keys()
# print(c_names)
# print(len(c_names))
from numpy import linalg as la
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV as gscv


X.reset_index(drop=True)
xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size=.3,random_state=0)




from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image, display
# import pydotplus  
from six import StringIO
import PIL

initRFC = RandomForestClassifier(max_depth=3, n_estimators=10, class_weight='balanced')
initRFC.fit(xtrain,ytrain)
print('Initial Random Forest Classifier Accuracy: ' ,initRFC.score(xtest,ytest))

Ss=[]
for i  in range(10,50,5):
    model=RandomForestClassifier(n_estimators=i , max_depth=None, class_weight='balanced')
    model.fit(xtrain,ytrain)
    s = model.score(xtest,ytest)
    print('Accuracy RFC with ',i,'estimators: ', s)
    Ss.append(s)
plt.plot(range(10,50,5),Ss)
plt.ylabel('Accuracy')
plt.xlabel('Number of estimators')
plt.xticks(range(10,50,5))
plt.title('RTC accuacy vs nestimators')
plt.show()

model=gscv(RandomForestClassifier( class_weight='balanced'),
{'n_estimators':[10,15,20,25,30,35,40],'max_depth':[3,4,5,10,20]},
cv=5,return_train_score=False)
model.fit(X,y)
bestRFCgscv= model.best_estimator_
ressRFC = model.cv_results_
resRFC=pd.DataFrame(ressRFC)
# print(resRFC)
print('Best parameters for RFC: ',bestRFCgscv)
savegscv = resRFC.to_csv('gscvRFCWEIGHTED.csv', index=True)

bestRTC = RandomForestClassifier(n_estimators=40,max_depth=20, class_weight='balanced')
bestRTC.fit(xtrain,ytrain)
bestypredRFC = bestRTC.predict(xtest)
# bestRFCgscv.fit(xtrain,ytrain)
# bestypredRFC = bestRFCgscv.predict(xtest)
print('best RFC accuracy_score: ', accuracy_score(ytest,bestypredRFC))

# dot_data = StringIO()
# export_graphviz(bestRTC.estimators_[0], out_file=dot_data, filled=True, rounded=True, special_characters=True, 
# feature_names=c_names,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('di.png')
# for i in range(len(bestRTC.estimators_)):

#     dot_data = StringIO()
#     export_graphviz(bestRTC.estimators_[i], out_file=dot_data, filled=True, rounded=True, special_characters=True, 
#     feature_names=c_names,class_names=['0','1'])
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     name= 'diabetes',str(i),'.png'
#     graph.write_png(name)
#     Image(graph.create_png())


from sklearn.linear_model import Perceptron
# ppn = Perceptron( n_iter_no_change=5,eta0=.1, random_state=0)
# ppn.fit(xtrain,ytrain)
# y_pred=ppn.predict(xtest)
# acc= accuracy_score(ytest,y_pred)
# print(acc*100)



from sklearn.neighbors import KNeighborsClassifier
KNNS=[]
# Is=[]
# for i in range (3,40):
#     knn = KNeighborsClassifier(n_neighbors=i, class_weight='balanced')
#     knn.fit(xtrain,ytrain)
#     ypredKNN=knn.predict(xtest)
#     Is.append(i)
#     KNNS.append(accuracy_score(ytest,ypredKNN))

# plt.figure()
# plt.plot(range(3,40),KNNS)
# plt.ylabel('number of neighbors')
# plt.xlabel('accuarcy')
# plt.title('Knearestneighbor accuracy')
# plt.show()
# maxKNNAC = 0
# maxKNNat = 0
# for i in range(len(KNNS)):
#     if KNNS[i] > maxKNNACCat:
#         maxKNNACCat = KNNS[i]
#         maxKNNat = Is[i]

# bestknn = KNeighborsClassifier(n_neighbors=maxKNNat, class_weight='balanced')
# bestknn.fit(xtrain,ytrain)
# bestypredKNN=bestknn.predict(xtest)
# print('Best KNN acc score: ', accuracy_score(ytest,bestypredKNN))



from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
# classweights = compute_class_weight('balanced',[0,1],y)
# print (classweights)
X=X.drop(columns=nominialCats)
X=pd.concat([X,onehots],axis=1)

mlp=MLPClassifier(hidden_layer_sizes=5,activation='logistic',learning_rate_init=.05,batch_size=30,
solver='lbfgs',random_state=0,max_iter=200, class_weight='balanced')
mlp.fit(xtrain,ytrain)
ypredMLP=mlp.predict(xtest)
accscoreMLP=accuracy_score(ytest,ypredMLP)
print('initial MLP accscore: ',accscoreMLP)

mplres=[]
for i in range(1,11):
    clf = MLPClassifier(hidden_layer_sizes=i,activation='logistic',learning_rate_init=.05,
    batch_size=30,solver='lbfgs',random_state=0,max_iter=300)
    clf.fit(xtrain,ytrain)
    result = clf.predict(xtest)
    mlpacc=accuracy_score(ytest,result)
    mplres.append(mlpacc)
plt.plot(range(1,11),mplres)
plt.ylabel('Accuracy score')
plt.xlabel('Hidden Layer Values')
plt.title('MLP hidden layers vs. accuarcy')
plt.show()

clf=gscv(MLPClassifier(activation='logistic',batch_size=30,solver='lbfgs',random_state=0,
max_iter=300,learning_rate_init=.05, class_weight='balanced'),
{'hidden_layer_sizes':[2,3,4,5,6,7,8,9,10]},cv=5,return_train_score=False)
clf.fit(X,y)
bestMLPgscv=clf.best_estimator_
ressMLP = clf.cv_results_
resMLP=pd.DataFrame(ressMLP)
print('best MLP params: ',bestMLPgscv)
# print(resMLP)
savegscv = resMLP.to_csv('gsvMLPWEIGHTED.csv', index=True)

bestMLP = MLPClassifier(hidden_layer_sizes=5,activation='logistic',batch_size=30,solver='lbfgs',
random_state=0,max_iter=300,learning_rate_init=.05, class_weight='balanced')
bestMLP.fit(xtrain,ytrain)
bestypredMLP=bestMLP.predict(xtest)
bestaccMLP=accuracy_score(ytest,bestypredMLP)
print('best MLP accscore: ',bestaccMLP)



from sklearn.metrics import confusion_matrix
import seaborn as sn
cmRTC = confusion_matrix(ytest,bestypredRFC)
cmMLP = confusion_matrix(ytest,bestypredMLP)
cmKNN = confusion_matrix(ytest,bestypredKNN)
# bestMLP1s = 0
# bestMLPs0s = 0
# print('MLP guess')
# for i in range(len(bestypredMLP)):
#     if bestypredMLP[i]==0:
#         bestMLPs0s+=1
#     else:
#         bestMLP1s+=1
# print('1s:',bestMLP1s)
# print('0s:' , bestMLPs0s)
# bestRfcs1s = 0
# bestRfcs0s = 0
# print('RFC guess')
# for i in range(len(bestypredRFC)):
#     if bestypredRFC[i]==0:
#         bestRfcs0s+=1
#     else:
#         bestRfcs1s+=1
# print('1s:',bestRfcs1s)
# print('0s:' , bestRfcs0s)
# print('knn guess')
# bestknn1s = 0
# bestknn0s = 0
# for i in range(len(bestypredKNN)):
#     if bestypredKNN[i]==0:
#         bestknn0s+=1
#     else:
#         bestknn1s+=1
# print('1s:',bestknn1s)
# print('0s:' , bestknn0s)
# print('true')
# tru1s=0
# tru0s=0
# for i in range(len(ytest)):
#     if ytest.iloc[i]==0:
#         tru0s+=1
#     else:
#         tru1s+=1
# print('1s:',tru1s)
# print('0s:' , tru0s)

plt.figure(figsize=(10,7))
sn.heatmap(cmRTC,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for RFC')
plt.show()
plt.figure(figsize=(10,7))
sn.heatmap(cmMLP,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for MLP')
plt.show()
plt.figure(figsize=(10,7))
sn.heatmap(cmKNN,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN')
plt.show()
# # xmean= np.mean(X,0)
# # cov=np.cov(X.T)
# # w,v = la.eig(cov)
# # print(w)
# from sklearn import decomposition
# # pca = decomposition.PCA(n_components=4)
# # pca.fit(X)
# # cpca=pca.transform(X)


