#Gohil Happy Kanaiyalal
#21IM30006

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import seaborn 
import seaborn as sns
from sklearn.model_selection import train_test_split

# reading data in df
path = 'dataset.csv'
df = pd.read_csv(path)

# printing the info about data
df.info()

# droping the 'User_ID' and 'Product_ID'
df.drop(['User_ID','Product_ID'], axis = 1,inplace= True)

# remove the feature which has nan values more then 50%
for features in df.columns:
    if df[features].isna().sum()/len(df)>0.5:
        print(f'removing {features} becuase it has {(df[features].isna().sum()/len(df))*100} NaN values\n')
        print('_________________________________________________\n')
        df.drop(features, axis = 1)

# printing the unique values for each features and determining where to use which feature
for feature in df.columns:
    print(f'for feature: {feature:}')
    print(f'_'*100)
    print(len(df[feature].unique()))
    print(df[feature].unique())
    print(f'_'*100)

# getting info about nan values 
print("_________________________________________________________\n\n")
print(df.Product_Category_2.value_counts())
print("from above we observe that there is not much difference in value count of 8 and 14 so using a mode to fill NaN value will not be fesibel \n")
print("So we will use forward fill")

# first filling the data with forward fill and then applying the backward fill
# for filling the fisrt element of product_cata-2 feature
df['Product_Category_2'].ffill(inplace=True)
df['Product_Category_2'].bfill(inplace=True)


# ploting the count data for each feature
df_male = df.loc[df['Gender']=='M']
df_female = df.loc[df['Gender']=='F']
# Gender
df['Gender'].value_counts().plot(kind='bar',title='male female count in data')
# Stay_In_Current_City_Years
df['Stay_In_Current_City_Years'].value_counts().plot(kind='bar',title='count distribution year stayed in city wise',
                                                     xlabel='years stayed in city',ylabel='frequancy')
#Age
df['Age'].value_counts().plot(kind='bar',title='count distribution age wise',
                                                     xlabel='age groups',ylabel='frequancy')
# Occupation
df['Occupation'].value_counts().plot(kind='bar',title='count distribution occupation wise',
                                                     xlabel='occupation',ylabel='frequancy')

# product catagory 1
df['Product_Category_1'].value_counts().plot(kind='bar',title='count distribution Product_Category_1 wise',
                                                     xlabel='Product_Category_1',ylabel='frequancy')
# product catagory 2
df['Product_Category_2'].value_counts().plot(kind='bar',title='count distribution Product_Category_2 wise',
                                                     xlabel='Product_Category_2',ylabel='frequancy')
# marital 
df['Marital_Status'].value_counts().plot(kind='bar',title='count distribution Marital_Status wise',
                                                     xlabel='Marital_Status',ylabel='frequancy')

# Histogram of purchase of male and female
df_male = df.loc[df['Gender']=='M']
plt.style.use('ggplot')
sns.histplot(x='Purchase', data = df, hue='Gender',multiple = "dodge",bins = 20,legend= True)
#plt.hist(df_male['Purchase'],color='red',bins=50,alpha = 0.5,label='Male')
#plt.hist(df_female['Purchase'],color='blue',bins=50,alpha = 0.5, label='Female')
plt.title('Histogram of purchase of male and female')
plt.xlabel('purchase amount')
plt.ylabel('Frequency')
plt.tight_layout

#distribution of purchase w.r.t age
sns.histplot(x = 'Purchase', hue = 'Age',
             multiple = "dodge", data= df, bins = 10)
plt.title('distribution of purchase w.r.t age')

#distribution of purchase w.r.t year stay in city
sns.histplot(x = 'Purchase', hue = 'Stay_In_Current_City_Years',
             multiple = "dodge", data= df, bins = 10)
plt.title('distribution of purchase w.r.t year stay in city')

#distribution of purchase w.r.t marital status
sns.histplot(x = 'Purchase', hue = 'Marital_Status',
             multiple = "dodge", data= df, bins = 20)
plt.title('distribution of purchase w.r.t marital status')

# using one-hot encoding on City_Category and Gender 
df = pd.get_dummies(df, columns = ['City_Category', 'Gender'],drop_first=True)

#replacing the the value of age into int
df['Age'].replace({'0-17':17, '55+':56, '26-35':35, '46-50':50, '51-55':55, '36-45':45, '18-25':25},inplace=True)

#replacing the the value of stay in current city year into int
df['Stay_In_Current_City_Years'].replace({'2':2, '4+':4, '3':3, '1':1, '0':0},inplace=True)

#LIN_MODEL_CLOSED class
class LinearRegressionMatrixForm:
    def __init__(self):
        self.weights = None
    
    def fit(self, X_train, Y_train):
        # Add a column of ones for the intercept term
        X_train_with_intercept = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        
        # Calculate the weights using the normal equation
        self.weights = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept) @ X_train_with_intercept.T @ Y_train
    
    def predict(self, X_test):
        # Add a column of ones for the intercept term
        X_test_with_intercept = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
        # Predict using the calculated weights
        predictions = X_test_with_intercept.dot(self.weights)
        return predictions
    
    def calculate_mse(self, X, Y):
        predictions = self.predict(X)
        mse = np.mean((predictions - Y)**2)
        return mse
    

# class for gradient decent linear regression
class LinearRegression_sc:
    def __init__(self, learning_rate, batch_size=256, epochs=50):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.theta = None
    
    def fit(self, X_train, Y_train):
        X_train_with_intercept = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        self.theta = np.random.rand(X_train_with_intercept.shape[1], 1)
        
        for epoch in range(self.epochs):
            shuffled_indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train_with_intercept[shuffled_indices]
            Y_shuffled = Y_train[shuffled_indices]
            
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                Y_batch = Y_shuffled[i:i+self.batch_size]
                
                Y_pred_batch = X_batch.dot(self.theta)
                gradient = X_batch.T.dot(Y_pred_batch - Y_batch) / self.batch_size
                
                self.theta -= self.learning_rate * gradient

    
    def predict(self, X_test):
        X_test_with_intercept = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        Y_pred = X_test_with_intercept.dot(self.theta)
        return Y_pred

    def calculate_mse(self, X, Y):
        Y_pred = self.predict(X)
        mse = np.mean((Y_pred - Y)**2)
        return mse

# with out feature scaling
print(df.head())
X = df.iloc[:,[0,1,2,3,4,5,7,8,9]]
y = df.iloc[:,[6]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

lin_close = LinearRegressionMatrixForm()
lin_close.fit(X_train,y_train)
lin_close_mse = lin_close.calculate_mse(X_test,y_test)
print(f'Mse value for [LIN_MODEL_CLOSED]: {lin_close_mse}\n')

# making a data frame for mse value for multiple lr
mse_val_lr = []
for lr in ([1e-5,1e-4,1e-3,1e-2,1e-1,0.5]):
    lin_close = LinearRegression_sc(lr)
    lin_close.fit(X_train,y_train)
    mse_val_lr.append(lin_close.calculate_mse(X_test,y_test))
mse_df = pd.DataFrame()
mse_df['lr'] = pd.Series([1e-5,1e-4,1e-3,1e-2,1e-1,0.5])
mse_df['mse'] = pd.Series(mse_val_lr)


print(mse_df)

# ploting the data
mse_df.plot(kind='line')

normalized_X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(normalized_X, y, test_size=0.33, random_state=42)

X_train_s = X_train_s.to_numpy()
X_test_s = X_test_s.to_numpy()
y_train_s = y_train_s.to_numpy()
y_test_s = y_test_s.to_numpy()

lin_close = LinearRegressionMatrixForm()
lin_close.fit(X_train_s,y_train_s)
mse_close = lin_close.calculate_mse(X_test_s,y_test_s)

lin_grad = LinearRegression_sc(1e-4)
lin_grad.fit(X_train,y_train)
lin_grad.calculate_mse(X_test_s,y_test_s)

mse_val_lr = []
for lr in ([1e-5,1e-4,1e-3,1e-2,1e-1,0.5]):
    lin_grade = LinearRegression_sc(lr)
    lin_grade.fit(X_train_s,y_train_s)
    mse_val_lr.append(lin_grade.calculate_mse(X_test_s,y_test_s))
mse_df_lr = pd.DataFrame()
mse_df_lr['lr'] = pd.Series([1e-5,1e-4,1e-3,1e-2,1e-1,0.5])
mse_df_lr['mse'] = pd.Series(mse_val_lr)

optimal_lr = mse_df_lr.loc[mse_df_lr['mse']==min(mse_df_lr['mse'][0])]

# mse_df_lr is data frame containing the values of mse for different value of lr

#ridge regression class with gradient dicent
class RidgeRegressionGradientDescent:
    def __init__(self, alpha=1.0, learning_rate=optimal_lr, epochs=50):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
    
    def fit(self, X_train, Y_train):
        # Add a column of ones for the intercept term
        X_train_with_intercept = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        
        num_samples, num_features = X_train_with_intercept.shape
        
        # Initialize weights to zeros
        self.weights = np.zeros((num_features, 1))
        
        for _ in range(self.epochs):
            # Calculate gradients with regularization term
            gradients = (1/num_samples) * X_train_with_intercept.T @ (X_train_with_intercept @ self.weights - Y_train) + 2 * self.alpha * self.weights
            
            # Update weights using gradient descent
            self.weights -= self.learning_rate * gradients
    
    def predict(self, X_test):
        # Add a column of ones for the intercept term
        X_test_with_intercept = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
        # Predict using the calculated weights
        predictions = X_test_with_intercept @ self.weights
        return predictions
    
    def calculate_mse(self, X, Y):
        predictions = self.predict(X)
        mse = np.mean((predictions - Y)**2)
        return mse
    
alpha_list = np.arange(0,1.1,step=0.1)
rr_mse_list = []
for alpha in alpha_list:
    rr = RidgeRegressionGradientDescent(alpha)
    rr.fit(X_train,y_train)
    rr_mse_list.append(rr.calculate_mse(X_test,y_test))
mse_df_alpha = pd.DataFrame()
mse_df_alpha['alpha'] = pd.Series(alpha_list)
mse_df_alpha['mse'] = pd.Series(rr_mse_list)

