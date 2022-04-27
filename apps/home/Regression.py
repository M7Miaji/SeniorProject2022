from scraping import stock, history
import numpy as np
from numpy import array
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler    
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error
# read dataset
#filename = "datasets/500_Person_Gender_Height_Weight_Index"
#df = pd.read_csv(f"{filename}.csv", usecols=[1,2,3], header=0, names=["height", "weight", "index"])
#print(df)

startDate='2020-1-1'

df=history(stock('AAPL'), startDate)

df['TradeDate']=df.index
 
df1 = df[['Close']].values

df1 = np.array(df1)
df1 = df1.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(df1)
print(df1)

# splitting dataset into train and test split
training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data,test_data  =df1[0:training_size,:], df1[training_size:len(df1),:1]
train_data.shape

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+ time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
train_data.shape, test_data.shape


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted Value",predictions[:10][0])
print("Expected Value",y_test[:10][0])

pred_df= pd.DataFrame(predictions)
pred_df['TrueValues']=y_test

new_pred_df=pred_df.rename(columns={0: 'Predictions'})
new_pred_df.head()