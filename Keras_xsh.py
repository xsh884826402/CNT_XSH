
import sys
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import sklearn
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import pickle
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import ibmiotf.application
# from Queue import Queue
# %matplotlib inline


print(sys.getdefaultencoding())
# data_healthy = pickle.load(open('watsoniotp.healthy.phase_aligned.pickle', 'rb'))
# data_broken = pickle.load(open('watsoniotp.broken.phase_aligned.pickle', 'rb'))

# data_healthy = np.random.random((3000,3))
# data_broken = np.random.random((3000,3))
data_healthy = np.array([1005,0.08,1008.68,0.08,1011.41,0.09,1014.19,0.07,9,35,0.13,6621.23,0.04])
print(np.shape(data_healthy))
data_healthy = data_healthy.reshape(_, 2)
data_broken = data_broken.reshape(_, 2)

data_healthy_fft = np.fft.fft(data_healthy)
data_broken_fft = np.fft.fft(data_broken)

def scaleData(data): 
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)
	

data_healthy_scaled = scaleData(data_healthy)
data_broken_scaled = scaleData(data_broken)

timesteps = 10
dim = 2
samples = 2000
data_healthy_scaled_reshaped = data_healthy_scaled
#reshape to (300,10,3)
print(np.shape(data_healthy_scaled_reshaped))
data_healthy_scaled_reshaped.shape = (int(samples/timesteps),timesteps,dim)
print(np.shape(data_healthy_scaled_reshaped))
losses = []
 
def handleLoss(loss): 
        global losses
        losses+=[loss]
        # print("hanleLoss", losses)
        # print(loss)
 
class LossHistory(Callback): 
    def on_train_begin(self, logs={}): 
        self.losses = []
 
    def on_batch_end(self, batch, logs={}): 
        self.losses.append(logs.get('loss'))
        handleLoss(logs.get('loss'))

model = Sequential()
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(LSTM(50,input_shape=(timesteps,dim),return_sequences=True))
model.add(Dense(2))
model.compile(loss='mae', optimizer='adam')
 
def train(data): 
    data.shape = (200, 10, 2)
    model.fit(data, data, epochs=20, batch_size=200, validation_data=(data, data), verbose=0, shuffle=False,callbacks=[LossHistory()])
    data.shape = (2000, 2)
 
def score(data): 
    data.shape = (200, 10, 2)
    yhat =  model.predict(data)
    print('score_shape', np.shape(yhat))
    yhat.shape = (2000, 2)
    return yhat

for i in range(2):
     
    print("----------------")
    train(data_healthy_scaled)
    yhat_healthy = score(data_healthy_scaled)
    yhat_broken = score(data_broken_scaled)
    data_healthy_scaled.shape = (2000, 2)
    data_broken_scaled.shape = (2000, 2)
 
 
print("----------------broken")
train(data_broken_scaled)
yhat_healthy = score(data_healthy_scaled)
yhat_broken = score(data_broken_scaled)
data_healthy_scaled.shape = (2000, 2)
data_broken_scaled.shape = (2000, 2)
print('shape_input_predict',np.shape(yhat_healthy), np.shape(data_healthy_scaled))
print(yhat_healthy)
print(data_healthy_scaled)
fig, ax = plt.subplots(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
size = len(data_healthy_fft)
#ax.set_ylim(0,energy.max())
ax.plot(range(0,len(losses)), losses, '-', color='blue', animated = False, linewidth=1)
plt.show()