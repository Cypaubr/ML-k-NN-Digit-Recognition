# Machine Learning Digit Recognition

Recognizes digits using the k-NN algorithm and the MNIST datasets for training and testing.

## Steps
1. The MNIST dataset is loaded
2. The 70000 pieces dataset is samplet into parts of 5000
3. The dataset is splitted into training & testing sets
4. Error is tested using a given k-NN (3 in the code)
5. Score is optimized by testing various k values and holding the best score
6. The prediction is given by the model using the k value of the best score
