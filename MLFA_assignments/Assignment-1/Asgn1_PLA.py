#Gohil Happy
#21IM30006
#importing a libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split


# importing a data
input_path_1 = "Dataset-1\inputs_Dataset-1.npy"
output_path_1 = "Dataset-1\outputs_Dataset-1.npy"
input_1 = np.load(input_path_1)
output_1 = np.load(output_path_1)
output_1[output_1 == 0] = -1

input_path_2 = "Dataset-2\inputs_Dataset-2.npy"
output_path_2 = "Dataset-2\outputs_Dataset-2.npy"
input_2 = np.load(input_path_2)
output_2 = np.load(output_path_2)
output_2[output_2 == 0] = -1

input_path_3 = "Dataset-3\inputs_Dataset-3.npy"
output_path_3 = "Dataset-3\outputs_Dataset-3.npy"
input_3 = np.load(input_path_3)
output_3 = np.load(output_path_3)
output_3[output_3 == 0] = -1

df_data_1 = pd.DataFrame(input_1)
df_data_1['Label'] = output_1

df_data_2 = pd.DataFrame(input_2)
df_data_2['Label'] = output_2

df_data_3 = pd.DataFrame(input_3)
df_data_3['Label'] = output_3


# class of PLA which initiat a object with X(input) and y(output) data and randomly generate wight metrix and train_PLA method is 
# used to train/update the wigth vector 
class PLA_hy:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = np.random.rand(self.X.shape[1])
        self.error = []

    def y_predict(self):
        return np.sign(np.dot(self.X, self.w))

    def train_PLA(self, interations):
        for i in range(interations):
            y_predict = self.y_predict()
            index_error = np.where(y_predict != self.y)[0]
            for j in index_error:
                self.w += self.X[j].T * self.y[j]
            self.error.append(index_error.shape[0])

# prediction method is used for predicting output for given wigth vector
def prediction(X_test,w):
   return np.sign(np.dot(X_test, w))

# confution_matrix is used to determine a TP, FP, FN, TN which is later used to give a precision, recall, accuracy, and F1 in series
# eg
# precision, recall, accuracy, F1 = confusion_matrix(prediction, output)
def confusion_matrix(prediction,output):
      TP = 0
      FP = 0
      FN = 0
      TN = 0
      for i in range(len(prediction)):
        if (output[i] == prediction[i] and output==1).any():
          TP += 1
        elif (output[i] == prediction[i] and output[i]== -1).any():
          TN += 1
        elif (output[i] != prediction[i] and output[i]==1).any():
          FN += 1
        else:
          FP += 1

      return TP/(TP+FP), TP/(TP+FN), (TP+TN)/len(prediction), (2*TP)/(2*TP + FP +FN)

#experiment 1a
print("\n")
print("_______________________________Experiment-1,2,3 (a)__________________________________\n")

# k_fold_matrix returns a pandas data frame with coloumn as precision, recall, accuracy, F1 and indexed for k from 2 to 10

def k_fold_matrix(n,df):
  PLA_wights = []
  precision_list = []
  recall_list = []
  accuracy_list = []
  F1_list = []
  kf = model_selection.KFold(n_splits = n,shuffle = False)
  for fold, (train_index, test_index) in enumerate(kf.split(X=df)):
    X_train, X_test = input_1[train_index], input_1[test_index]
    y_train, y_test = output_1[train_index], output_1[test_index]
    exp_1 = PLA_hy(X_train,y_train)
    exp_1.train_PLA(2000)
    y_pred = prediction(X_test,exp_1.w)
    precision = 0
    recall = 0
    accuracy = 0
    F1 = 0
    precision, recall, accuracy, F1 = confusion_matrix(y_pred,y_test)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)
    F1_list.append(F1)
  precision_np = np.array(precision_list)
  recall_np = np.array(recall_list)
  accuracy_np = np.array(accuracy_list)
  F1_np = np.array(F1_list)

  df = pd.DataFrame()
  df['precision'] = precision_np
  df['recall'] = recall_np
  df['accuracy']=accuracy_np
  df['F1'] = F1_np
  print(df)
  return df

  print("For Dataset 1 \n\n")
for n in range(2,11):
  print(f"For K = {n}")
  k_fold_matrix(n, df_data_1)
  print("\n\n")

print("For Dataset 2 \n\n")
for n in range(2,11):
  print(f"For K = {n}")
  k_fold_matrix(n, df_data_2)
  print("\n\n")

print("For Dataset 3 \n\n")
for n in range(2,11):
  print(f"For K = {n}")
  k_fold_matrix(n, df_data_3)
  print("\n\n")


#experiment 1b
print("\n")
print("_______________________________Experiment-1,2,3 (mean and variance of performance value of K from 2 to 10)__________________________________\n")



# k_fold_mv returns numpy array of mean and variance value of preformance values(prescion, recall, accurassy, F1) for n splits 
# which is later converted into data frame to easly print the data for different value of k
def k_fold_mv(n,df):
  PLA_wights = []
  precision_list = []
  recall_list = []
  accuracy_list = []
  F1_list = []
  kf = model_selection.KFold(n_splits = n,shuffle = False)
  for fold, (train_index, test_index) in enumerate(kf.split(X=df)):
    X_train, X_test = input_1[train_index], input_1[test_index]
    y_train, y_test = output_1[train_index], output_1[test_index]
    exp_1 = PLA_hy(X_train,y_train)
    exp_1.train_PLA(2000)
    y_pred = prediction(X_test,exp_1.w)
    precision = 0
    recall = 0
    accuracy = 0
    F1 = 0
    precision, recall, accuracy, F1 = confusion_matrix(y_pred,y_test)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)
    F1_list.append(F1)
  precision_np = np.array(precision_list)
  recall_np = np.array(recall_list)
  accuracy_np = np.array(accuracy_list)
  F1_np = np.array(F1_list)

  mean = np.array([precision_np.mean(),recall_np.mean(),accuracy_np.mean(),F1_np.mean()])
  var = np.array([precision_np.var(),recall_np.var(),accuracy_np.var(),F1_np.var()])
  #print(pd.DataFrame(answer,columns=['mean','variance'],index=['precision','recall','accuracy','F1']))
  return mean, var


mean_list = []
var_list = []
for i in range(2,11):
  mean, var = k_fold_mv(i,df_data_1)
  mean_list.append(mean)
  var_list.append(var)

mean_df_1 = pd.DataFrame(mean_list,columns=['Precision','recall','accuracy','F1'])
var_df_1 = pd.DataFrame(var_list,columns=['Precision','recall','accuracy','F1'])

mean_list = []
var_list = []
for i in range(2,11):
  mean, var = k_fold_mv(i,df_data_2)
  mean_list.append(mean)
  var_list.append(var)

mean_df_2 = pd.DataFrame(mean_list,columns=['Precision','recall','accuracy','F1'])
var_df_2 = pd.DataFrame(var_list,columns=['Precision','recall','accuracy','F1'])

mean_list = []
var_list = []
for i in range(2,11):
  mean, var = k_fold_mv(i,df_data_3)
  mean_list.append(mean)
  var_list.append(var)

mean_df_3 = pd.DataFrame(mean_list,columns=['Precision','recall','accuracy','F1'])
var_df_3 = pd.DataFrame(var_list,columns=['Precision','recall','accuracy','F1'])
print("for data set 1______________________________________\n")
print("mean values: \n")
print(mean_df_1)
print("\n")
print(var_df_1)
print("___________________________________________________________________\n\n\n")

print("for data set 2______________________________________\n")
print("mean values: \n")
print(mean_df_2)
print("\n")
print(var_df_2)
print("___________________________________________________________________\n\n\n")

print("for data set3______________________________________\n")
print("mean values: \n")
print(mean_df_3)
print("\n")
print(var_df_3)
print("___________________________________________________________________\n\n\n")

print("\n")
print("_______________________________Experiment-1,2,3 (c)__________________________________\n")



#run_8020 is used for spliting data in 80-20 and then training it and also print the performance value on test data
def run_8020(X,y,iterations,name):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  exp = PLA_hy(X_train,y_train)
  exp.train_PLA(iterations)
  s = np.array(exp.error)
  y_pred = prediction(X_test,exp.w)
  p,r,a,f1 = confusion_matrix(y_pred,y_test)
  print("Performance value on test data_________________________________\n")
  print("precision:",p,"/n")
  print("recall:",r,"/n")
  print("accuracy:",a,"/n")
  print("F1:",f1,"/n")
  print("/n/n")
  plt.plot(s)
  plt.title('iteration vs #misclassified instances')
  plt.xlabel('iteration')
  plt.ylabel('#misclassified instances')
  plt.savefig(name)


print("for data 1___________________________________________________\n")
run_8020(input_1,output_1,25,'experiment-1')
print("for data 2____________________________________________________\n")
run_8020(input_2,output_2,25,'experiment-2')
print("for data 3____________________________________________________\n")
run_8020(input_3,output_3,2000,'experiment-3')
