import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt

# Loading data set
print('Loading MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='mldata')
print(mnist.data.shape)
print(mnist.target.shape)

# Sampling
print('Sampling dataset into parts of 5000')
sample = np.random.randint(70000, size=5000) #Trivial sampling, should be resampled to avoid bias in case of size > 5000
data = mnist.data[sample]
target = mnist.target[sample]

# Splitting training/testing set
print('Splitting dataset into training/testing set')
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8, test_size=0.2)

# k-NN Algorithm
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

# Error testing
print('Error testing for a given k-NN, here 3')
error = 1 - knn.score(xtest, ytest)
print('Error for 3-NN: %f' % error)

# Score optimization
print('Selecting best score based on error')
errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
best_score = errors.index(np.amin(errors))
print(best_score)
plt.plot(range(2,15), errors, 'o-')
plt.show()

# Classifier predictions for 3-NN
print('Predicting using best score')
knn = neighbors.KNeighborsClassifier(best_score)
knn.fit(xtrain, ytrain)
predicted = knn.predict(xtest)
images = xtest.reshape((-1, 28, 28))
select = np.random.randint(images.shape[0], size=12)
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % predicted[value])
plt.show()
