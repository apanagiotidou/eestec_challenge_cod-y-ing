from utils import *
from sklearn.model_selection import train_test_split

# STEP 1
path = 'myData'
data = importDataInfo(path)

# STEP 2
data = balanceData(data, display=False)

# STEP 3
imagesPath, steerings = loadData(path, data)
print(imagesPath[0], steerings[0])

# STEP 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steerings, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

# STEP 5
# Augmentation

#STEP 6

#STEP 7

#STEP 8
model = createModel()
model.summary()

#STEP 9
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=330, epochs=20,
          validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=220)

#STEP 10
model.save('model.h5')
print("Model saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title("Losses")
plt.xlabel('Epoch')
plt.show()




