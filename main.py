import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cost_function(X, Y, theta):
    term = np.power((np.dot(X, theta.T) - Y), 2)
    return np.sum(term) / (2 * len(X))

def gradient_descent(X, Y, theta, alpha, iterations):
    cost_per_iteration = []
    for i in range(iterations):
        tempTheta = theta[0]
        for j in range(len(theta[0])):
            tempTheta[j] = theta[0][j] - (alpha / len(X)) * (
                np.sum((np.dot(X, theta.T) - Y) * np.array(X[:, j]).reshape(len(X), 1)))
        theta = np.array([tempTheta])
        cost_per_iteration.append(cost_function(X, Y, theta))
    # print("MSE in iteration ", i+1, " is ", cost_function(X, Y, theta))
    draw_cost_per_iteration(cost_per_iteration)
    return theta

def hypothesis_function(X, theta):
    Y = X @ theta.T
    return Y

def normalization(X, mean, std):
    x = np.asarray(X)
    res = (x - mean) / std
    return res

def draw_cost_per_iteration(cost_per_iteration):
    plt.figure(figsize=(10, 10))
    plt.plot(range(len(cost_per_iteration)), cost_per_iteration, 'bo')
    plt.grid(True)
    plt.xlabel("Iteration num")
    plt.ylabel("MSE")
    plt.show()

# read dataset
df = pd.read_csv("house_data.csv", index_col=0)
x = df[["sqft_living"]]
y = df["price"]

X_train = x[:21614]
Y_train = y[:21614]

alpha = 0.001
iters = 1000
theta = np.array([[1.0, 1.0]])

meanTrain = np.asarray(X_train).mean()
stdTrain = np.asarray(X_train).std()

X_train_norm = normalization(X_train, meanTrain, stdTrain)

onesTrain = np.ones([X_train_norm.shape[0], 1])  # create a array containing only ones
Xtrain = np.concatenate([onesTrain, X_train_norm], 1)  # cocatenate the ones to X matrix
yArrayTrain = np.array(Y_train)
Ytrain = yArrayTrain.reshape(-1, 1)

print("b) ")
thetaMat = gradient_descent(Xtrain, Ytrain, theta, alpha, iters)

print("Optimal Thetas:", theta)

print("c) ")
newdata = int(input("Enter a number: "))
xinput = [[newdata]]
input_norm = normalization(xinput, meanTrain, stdTrain)
x0 = np.ones([input_norm.shape[0], 1])
inputMat = np.concatenate([x0, input_norm], 1)
print("Predicted price: ", hypothesis_function(inputMat, thetaMat)[0][0])

print("d) ")

alpha_list = [0.001, 0.003, 0.01, 0.03, 0.3, 1]
for i in alpha_list:
    print("For alpha = ", i)
    thetaMat = gradient_descent(Xtrain, Ytrain, theta, i, iters)
