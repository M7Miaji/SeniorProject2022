# imports
import pandas as pd
import numpy as np
from scraping import stock, history

# read dataset
#filename = "datasets/500_Person_Gender_Height_Weight_Index"
#df = pd.read_csv(f"{filename}.csv", usecols=[1,2,3], header=0, names=["height", "weight", "index"])
#print(df)

startDate='2020-1-1'

df=history(stock('AAPL'), startDate)

df['TradeDate']=df.index

print(df)
# useful functions
def split_train_test(df, p):
    """
    Takes a dataframe and the number of data in the train set.
    Returns a list of dataframes.
    """
    n = int(p*len(df))
    train = df.iloc[0:n, :]
    test = df.iloc[n:len(df), :]
    return train, test

# split dataset into train and test
train_df, test_df = split_train_test(df, 0.8)

print(train_df)
print(test_df)

# create array-like objects for train and test data
x_train = np.array(train_df["Open"])
y_train = np.array(train_df["Close"])
z_train = np.array(train_df["High"])

x_test = np.array(test_df["Open"])
y_test = np.array(test_df["Close"])
z_test = np.array(test_df["High"])

# set initial values for learnable parameters
a = 1
b = 1
c = 0

lr = 0.000005 # learning rate
epochs = 1000 # number of iterations

n = len(z_train)

for i in range(epochs):
    z_predicted = a*x_train + b*y_train + c # make a prediction
    error = z_predicted - z_train # calculate the error
    loss = np.sum(error**2)/n # calculate the loss 
    loss_a = 2*np.sum(error*x_train)/n # partial derivatives of the loss
    loss_b = 2*np.sum(error*y_train)/n
    loss_c = 2*np.sum(error)/n
    a = a - loss_a*lr # adjust the parameters 
    b = b - loss_b*lr
    c = c - loss_c*lr
    #print(f"loss: {loss}  \t({i+1}/{epochs})")
    


#print(f"a: {a}")
#print(f"b: {b}")
#print(f"c: {c}")

from sklearn.metrics import r2_score

z_prediction = a*x_test + b*y_test + c
print(f"R2 Score: {r2_score(z_test, z_prediction)}")

for i in range(len(z_prediction)):
    print(f"BMI: {z_test[i]}  Predicted: {z_prediction[i]}")

