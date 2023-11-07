import numpy as np
import math
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 16, 8

stocks = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN"]
predict_day_size = 100
for stock in stocks:

    stock_name = stock
    start = datetime.datetime(2012, 1, 1)
    today = datetime.date.today()
    print('date: ' + str(today))
    yesterday = today - datetime.timedelta(days=1)
    predict_day = today - datetime.timedelta(days=predict_day_size + (predict_day_size * 0.5))
    df = yf.download(stock_name, start=start, end=yesterday)
    # show the data
    # print(df)
    # get number of ros and colmns
    print(df.shape)

    # create a new data frame with close data
    data = df.filter(['Close'])
    # convert to numpy array
    dataset = data.values
    # get the number of rows to train model on
    train_data_len = math.ceil(len(dataset) * .8)

    print('train data len: ' + str(train_data_len))

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # create the training dataset
    # create the scaled training dataset
    train_data = scaled_data[0:train_data_len, :]
    # split the data into xtrain and ytrain datasets
    x_train = []
    y_train = []

    for i in range(predict_day_size, len(train_data)):
        x_train.append(train_data[i - predict_day_size:i, 0])
        y_train.append(train_data[i, 0])
        if i <= predict_day_size + 1:
            print('x_train and y_train generated')
            # print(y_train)
            # print()

    # convert x_train and y_train to nummpy array
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshape the data to 3d
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))

    # compile th model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    model.fit(x_train, y_train, epochs=50)

    model.save(str(stock_name) + '_stock_model.h5')
    print('model saved as ' + stock_name + '_stock_model.h5')
