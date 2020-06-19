# https://www.geeksforgeeks.org/command-line-arguments-in-python/
import sys
# total arguments 
nargs = len(sys.argv) 
if(nargs < 3):
	print('not enough arguments')
	exit()

filename_model = sys.argv[1]
filename_test = sys.argv[2]
filename_ypredict = sys.argv[3]



# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# read csv
dataX = pd.read_csv(filename_test)
dataX.fillna(value=0, inplace=True) 
print('--> read csv done\n');

def get_headers_name():
	headers = []
	for i in range(1, 785):
		colName = 'pixel'+str(i)
		headers.append(colName)
	return headers

X = dataX[get_headers_name()].values
Y = dataX['label'].values

print('--> show X, Y data\n')
print(X)
print(Y)



# load & predict
print('--> loading model...\n')
# load the model from disk
import pickle
loaded_model = pickle.load(open(filename_model, 'rb'))




print('--> predicting...\n')
y_pred = loaded_model.predict(X)

print('--> predict done, calculate score')
score = loaded_model.score(X, Y)

print(score)

print('------\n')
print(Y)
print(y_pred)

print(y_pred.shape)
# print(loaded_model.coef_.shape)

print('--> save prediction to csv file')
pd.DataFrame(y_pred).to_csv(filename_ypredict)


# draw table
print('--> drawing confusion matrix\n')
# confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


cm = metrics.confusion_matrix(Y, y_pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

plt.show()
