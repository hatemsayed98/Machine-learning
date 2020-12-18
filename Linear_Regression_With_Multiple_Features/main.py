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

x2 = df[["grade"]]
x3 = df[["bathrooms"]]
x4 = df[["lat"]]
x5 = df[["sqft_living"]]
x6 = df[["view"]]
y = df["price"]

X2_train = x2[:21614]
X3_train = x3[:21614]
X4_train = x4[:21614]
X5_train = x5[:21614]
X6_train = x6[:21614]
Y_train = y[:21614]

alpha = 0.01
iters = 1000
theta = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

meanTrain_X2 = np.asarray(X2_train).mean()
stdTrain_X2 = np.asarray(X2_train).std()

meanTrain_X3 = np.asarray(X3_train).mean()
stdTrain_X3 = np.asarray(X3_train).std()

meanTrain_X4 = np.asarray(X4_train).mean()
stdTrain_X4 = np.asarray(X4_train).std()

meanTrain_X5 = np.asarray(X5_train).mean()
stdTrain_X5 = np.asarray(X5_train).std()

meanTrain_X6 = np.asarray(X6_train).mean()
stdTrain_X6 = np.asarray(X6_train).std()

X2_train_norm = normalization(X2_train, meanTrain_X2, stdTrain_X2)
X3_train_norm = normalization(X3_train, meanTrain_X3, stdTrain_X3)
X4_train_norm = normalization(X4_train, meanTrain_X4, stdTrain_X4)
X5_train_norm = normalization(X5_train, meanTrain_X5, stdTrain_X5)
X6_train_norm = normalization(X6_train, meanTrain_X6, stdTrain_X6)

onesTrain = np.ones([X2_train_norm.shape[0], 1])  # create a array containing only ones

Xtrain = np.concatenate([onesTrain, X2_train_norm, X3_train_norm, X4_train_norm, X5_train_norm, X6_train_norm],
                        1)  # cocatenate the ones to X matrix

yArrayTrain = np.array(Y_train)
Ytrain = yArrayTrain.reshape(-1, 1)

print("b) ")
thetaMat = gradient_descent(Xtrain, Ytrain, theta, alpha, iters)

print("Otimal Thetas:", theta)

print("c) ")

newdata1 = int(input("Enter house's grade: "))
newdata2 = int(input("Enter house's bathrooms: "))
newdata3 = float(input("Enter house's lat: "))
newdata4 = int(input("Enter house's sqft_living: "))
newdata5 = int(input("Enter house's view: "))

xinput1 = [[newdata1]]
xinput2 = [[newdata2]]
xinput3 = [[newdata3]]
xinput4 = [[newdata4]]
xinput5 = [[newdata5]]

input_norm1 = normalization(xinput1, meanTrain_X2, stdTrain_X2)
input_norm2 = normalization(xinput2, meanTrain_X3, stdTrain_X3)
input_norm3 = normalization(xinput3, meanTrain_X4, stdTrain_X4)
input_norm4 = normalization(xinput4, meanTrain_X5, stdTrain_X5)
input_norm5 = normalization(xinput5, meanTrain_X6, stdTrain_X6)

x0 = np.ones([input_norm1.shape[0], 1])
inputMat = np.concatenate([x0, input_norm1, input_norm3, input_norm3, input_norm4, input_norm5], 1)
print("Predictied price: ", hypothesis_function(inputMat, thetaMat)[0][0])

print("d) ")

alpha_list = [0.001, 0.003, 0.01, 0.03, 0.3, 1]
for i in alpha_list:
    print("For alpha = ", i)
    thetaMat = gradient_descent(Xtrain, Ytrain, theta, i, iters)
