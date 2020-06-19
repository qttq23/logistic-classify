# https://www.geeksforgeeks.org/command-line-arguments-in-python/
import sys
# total arguments 
nargs = len(sys.argv) 
if(nargs < 3):
	print('not enough arguments')
	exit()

filename_train = sys.argv[1]
filename_model = sys.argv[2]


# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# read csv
dataX = pd.read_csv(filename_train)
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




#####
# https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
def vec2img(list_vector, list_label):
	import numpy as np 
	import matplotlib.pyplot as plt
	
	figure = plt.figure(figsize=(20,40))
	ncols = 4
	nrows = len(list_vector)/ncols + 1
	for index, (image, label) in enumerate(zip(list_vector, list_label)):
		figure.add_subplot(nrows, ncols, index + 1)
		plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
		plt.title('Training: %i\n' % label, fontsize = 10)

	plt.show()


print('--> show image at index: 3 \n')
# vec2img(X[3:40], Y[3:40])
vec2img(X[3:4], Y[3:4])


# https://stackoverflow.com/questions/37558523/converting-2d-numpy-array-of-grayscale-values-to-a-pil-image?rq=1
def vec2img2(vector, path):
	import numpy as np
	from PIL import Image

	print(vector, vector.shape)
	mat = np.reshape(vector, (28,28))
	mat = np.uint8(mat * 255)
	img = Image.fromarray(mat, 'L')

	img.save(path)
	# img.show()
# vec2img2(X[3:4], 'resources/img.png')
print('\n------\n')



def img2vec(path):

	from PIL import Image
	im = Image.open(path).convert('L')
	# im.show()

	mat = np.array(im)
	mat = np.uint8(mat * 255)
	res = np.reshape(mat, (1,28**2))

	return res
# outVector = img2vec('resources/img.png')
# print(outVector, outVector.shape)

#####


print('--> vector of data at index: 3\n')
X3 = (dataX[get_headers_name()].values)[3]
print(X3)


print('--> split train data')
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


print('--> trainning...\n')
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
print('--> train done\n')


print('--> saving model...\n')
# save the model to disk
import pickle
pickle.dump(logisticRegr, open(filename_model, 'wb'))

print('--> loading model...\n')
# load the model from disk
loaded_model = pickle.load(open(filename_model, 'rb'))



print('--> predicting...\n')
y_pred = loaded_model.predict(x_test)

print('--> predict done, calculate score')
score = loaded_model.score(x_test, y_test)

print(score)

print('------\n')
print(y_test)
print(y_pred)

print(y_pred.shape)
# print(logisticRegr.coef_.shape)

print('---my_score\n')
def my_score(y_test, y_pred):
	return sum([y_pred[i] == y_test[i] for i in range(len(y_pred))])/len(y_pred)
print(my_score(y_test, y_pred))
print('------')


print('--> drawing confusion matrix\n')
# confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

plt.show()
