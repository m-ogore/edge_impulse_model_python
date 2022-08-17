
# load and evaluate a saved model
from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import load_model

#copy labels as they are ordered in edge impulse
labels=['DISCHARGEHIGH_D2',	'DISCHARGELOW_D1', 'NOFAULT',	'PARTIALDISCHARGE',	'THERMALFT12',	'THERMALFT3',	'ANOMALY']
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset set of features
X = [891.0000, 2020.0000, 46.0000, 1309.0000, 731.0000, 1328.0000]
X=np.reshape(X,(1,6))
# evaluate the model
# Evaluate the model.
prediction=model.predict(X)
prediction=np.argmax(prediction[0])
print(labels[prediction])
score = model.evaluate(X, verbose=0)
print("%s: %.2f%%" % (model.metrics_names, score*100))