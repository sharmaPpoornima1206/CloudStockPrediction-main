import numpy as np
import math
import datetime
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

stocks = {"Apple": "AAPL", "Tesla": "TSLA", "Microsoft": "MSFT", "Google": "GOOG", "Amazon": "AMZN"}
st.title('Cloud Based Stock Price Prediction')
stock_name = st.selectbox('Select The Stock Name', ("Apple", "Tesla", "Microsoft", "Google", "Amazon"))
ticker = stocks[stock_name]
st.write('You selected: ', stock_name)
st.write('Symbol: ', ticker)
predict_day_size = 100
start = datetime.datetime(2012, 1, 1)
today = datetime.date.today()
print("todays date: " + str(today))

yesterday = today - datetime.timedelta(days=1)
df = yf.download(ticker, start=start, end=today)
print(df)
df_date = df.filter(['Close'])

data = df_date.values
train_data_len = math.ceil(len(data) * .9)

# load model
model = load_model(ticker + '_stock_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# create a new array containnig scaled values
test_data = scaled_data[train_data_len - predict_day_size:, :]
# create dataset x_test and y_test
x_test = []
y_test = data[train_data_len:, :]
for i in range(predict_day_size, len(test_data)):
    x_test.append(test_data[i - predict_day_size:i, 0])

# convert the test data to numpy
x_test = np.array(x_test)

# reshape the test data to 3d
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get the models predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# get root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print('error: ' + str(rmse))
mse = mean_squared_error(y_test, predictions)
print('mse: ' + str(mse))
mae = mean_absolute_error(y_test, predictions)
print('mae: ' + str(mae))

# plot the data
train = df_date[:train_data_len]
test = df_date[train_data_len:].copy()
test['predictions'] = predictions

# testing model
predict_days = df_date[-predict_day_size:].values
predict_days_scaled = scaler.transform(predict_days)
X_test = []
X_test.append(predict_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
today_price = df_date['Close'].iloc[-1]
tomorrow_price = pred_price[0][0]
print('Today\'s price: ' + str(today_price))
print('Tomorrow\'s price: ' + str(tomorrow_price))

# graph1
st.subheader('Time Chart vs Stock Closing Price (' + str(start.date().strftime("%m-%d-%Y")) +
             ' - ' + str(today.strftime("%m-%d-%Y")) + ')')
st.write('Today\'s price: ' + str(round(today_price, 2)))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig1 = plt.figure(figsize=(14, 7))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig1)

# graph2
st.subheader('Time Chart vs Stock Closing Price Prediction (' +
             str(start.date().strftime("%m-%d-%Y")) + ' - ' + str(today.strftime("%m-%d-%Y")) + ')')
st.write('Predicted Tomorrow\'s price: ' + str(round(tomorrow_price, 2)))
fig2 = plt.figure(figsize=(14, 7))
plt.title('model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('price($USD)', fontsize=18)
plt.plot(train['Close'])
plt.plot(test[['Close', 'predictions']])
plt.legend(['training', 'testing', 'predictions'], loc='lower right')
st.pyplot(fig2)

st.subheader('Error')
st.write('Error of ${:.2f} USD can be observed'.format(rmse))
st.write('Mean Squared Error (MSE): {:.3f}'.format(mse))
st.write('Root Mean Squared Error (RMSE): {:.3f}'.format(rmse))
st.write('Mean Absolute Error (MAE): {:.3f}'.format(mae))
