import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cost_function(x, y, theta):
    total_cost = (-1 / len(x)) * np.sum(
        (y * np.log(hypothesis_function(x, theta))) + ((1 - y) * np.log(1 - hypothesis_function(x, theta))))
    return total_cost

def gradient_descent(X, Y, theta, alpha, iterations):
    cost_per_iteration = []
    for i in range(iterations):
        tempTheta = theta[0]
        for j in range(len(theta[0])):
            tempTheta[j] = theta[0][j] - (alpha / len(X)) * (
                np.sum((hypothesis_function(X, theta) - Y) * np.array(X[:, j]).reshape(len(X), 1)))
        theta = np.array([tempTheta])
        cost_per_iteration.append(cost_function(X, Y, theta))
        # print("MSE in iteration ", i + 1, " is ", cost_function(X, Y, theta))
    draw_cost_per_iteration(cost_per_iteration)
    return theta

def hypothesis_function(X, theta):
    y = 1 / (1 + np.exp(-X @ theta.T))
    return y

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
df = pd.read_csv("heart.csv", index_col=0)
x2 = df[["trestbps"]]
x3 = df[["chol"]]
x4 = df[["thalach"]]
x5 = df[["oldpeak"]]

y = df["target"]

X2_train = x2[:303]
X3_train = x3[:303]
X4_train = x4[:303]
X5_train = x5[:303]
Y_train = y[:303]

alpha = 1
iters = 1000
theta = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])

meanTrain_X2 = np.asarray(X2_train).mean()
stdTrain_X2 = np.asarray(X2_train).std()

meanTrain_X3 = np.asarray(X3_train).mean()
stdTrain_X3 = np.asarray(X3_train).std()

meanTrain_X4 = np.asarray(X4_train).mean()
stdTrain_X4 = np.asarray(X4_train).std()

meanTrain_X5 = np.asarray(X5_train).mean()
stdTrain_X5 = np.asarray(X5_train).std()

X2_train_norm = normalization(X2_train, meanTrain_X2, stdTrain_X2)
X3_train_norm = normalization(X3_train, meanTrain_X3, stdTrain_X3)
X4_train_norm = normalization(X4_train, meanTrain_X4, stdTrain_X4)
X5_train_norm = normalization(X5_train, meanTrain_X5, stdTrain_X5)

onesTrain = np.ones([X2_train_norm.shape[0], 1])  # create a array containing only ones

Xtrain = np.concatenate([onesTrain, X2_train_norm, X3_train_norm, X4_train_norm, X5_train_norm],
                        1)  # cocatenate the ones to X matrix

yArrayTrain = np.array(Y_train)
Ytrain = yArrayTrain.reshape(-1, 1)

print("b) ")
thetaMat = gradient_descent(Xtrain, Ytrain, theta, alpha, iters)

print("Optimal Threats: ", thetaMat)

print("c) ")

newdata1 = int(input("Enter Patient's trestbps: "))
newdata2 = int(input("Enter Patient's chol: "))
newdata3 = int(input("Enter Patient's thalach: "))
newdata4 = float(input("Enter Patient's oldpeak: "))

xinput1 = [[newdata1]]
xinput2 = [[newdata2]]
xinput3 = [[newdata3]]
xinput4 = [[newdata4]]

input_norm1 = normalization(xinput1, meanTrain_X2, stdTrain_X2)
input_norm2 = normalization(xinput2, meanTrain_X3, stdTrain_X3)
input_norm3 = normalization(xinput3, meanTrain_X4, stdTrain_X4)
input_norm4 = normalization(xinput4, meanTrain_X5, stdTrain_X5)

x0 = np.ones([input_norm1.shape[0], 1])
inputMat = np.concatenate([x0, input_norm1, input_norm3, input_norm3, input_norm4], 1)
target = hypothesis_function(inputMat, thetaMat)[0][0]
if (target >= 0.5):
    print("Predictied target: patient have heart disease")
else:
    print("Predictied target: patient not have heart disease")

print("d) ")

alpha_list = [0.001, 0.003, 0.01, 0.03, 0.3, 1]
for i in alpha_list:
    print("For alpha = ", i)
    thetaMat = gradient_descent(Xtrain, Ytrain, theta, i, iters)
