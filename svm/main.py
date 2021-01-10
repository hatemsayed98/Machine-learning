import pandas as pd
import numpy as np

def cost_fun(w, x):
    return np.dot(x, w)

def normalization(x, mean, std):
    x = np.asarray(x)
    res = (x - mean) / std
    return res

df = pd.read_csv('heart.csv')

listOfFeatures = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak', 'slope',	'ca', 'thal']
#listOfFeatures = ['cp', 'fbs', 'restecg', 'thalach', 'slope', 'oldpeak', 'ca', 'thal']
#listOfFeatures = ['cp', 'fbs', 'restecg', 'ca', 'slope', 'thalach']
best1 = listOfFeatures[0]
best2 = listOfFeatures[1]
bestAccuracy = 0
bestAlpha = 0

for k1 in range(len(listOfFeatures)):
    for k2 in range(k1 + 1, len(listOfFeatures)):
        sub_data = df[[listOfFeatures[k1], listOfFeatures[k2], 'target']]
        iterations = 5
        lamda = 1 / iterations

        train_data = sub_data.sample(frac=0.80)
        test_data = sub_data.drop(train_data.index.tolist())

        x1_train = train_data[[listOfFeatures[k1]]]
        x2_train = train_data[[listOfFeatures[k2]]]

        meanTrain_X1 = np.asarray(x1_train).mean()
        stdTrain_X1 = np.asarray(x1_train).std()

        meanTrain_X2 = np.asarray(x2_train).mean()
        stdTrain_X2 = np.asarray(x2_train).std()

        X1_train_norm = normalization(x1_train, meanTrain_X1, stdTrain_X1)
        X2_train_norm = normalization(x2_train, meanTrain_X2, stdTrain_X2)

        onesTrain = np.ones([X2_train_norm.shape[0], 1])  # create a array containing only ones
        x_train = np.concatenate([onesTrain, X1_train_norm, X2_train_norm], 1)


        y_train = np.array(train_data['target'])
        for i in range(len(y_train)):
            if y_train[i] == 0:
             y_train[i] = -1


        x1_test = test_data[[listOfFeatures[k1]]]
        x2_test = test_data[[listOfFeatures[k2]]]

        meanTest_x1_test = np.asarray(x1_test).mean()
        stdTest_x1_test = np.asarray(x1_test).std()
        meanTest_x2_test = np.asarray(x2_test).mean()
        stdTest_x2_test = np.asarray(x2_test).std()

        X1_test_norm = normalization(x1_test, meanTest_x1_test, stdTest_x1_test)
        X2_test_norm = normalization(x2_test, meanTest_x2_test, stdTest_x2_test)

        onesTest = np.ones([X1_test_norm.shape[0], 1])
        x_test = np.concatenate([onesTest, X1_test_norm, X2_test_norm], 1)


        y_test = np.array(test_data['target'])
        for i in range(len(y_test)):
            if y_test[i] == 0:
             y_test[i] = -1

        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)

        alpha_list = [0.0000001, 0.001, 0.01, 0.1, 1]
        for alpha in alpha_list:
            w = [[0.0], [0.0], [0.0]]
            b=0
            for i in range(iterations):
                cost_values = (cost_fun(w, x_train)+b) * y_train
                for j in range(len(cost_values)):
                    if cost_values[j][0] >= 1:
                        for k in range(len(w)):
                            w[k][0] = w[k][0] - (alpha * (2 * lamda * w[k][0]))
                        b = b - alpha * 2 * lamda * b
                    else:
                        for k in range(len(w)):
                           # print(x_train[j][k])
                            w[k][0] = w[k][0] + (alpha * ((y_train[j][0] * x_train[j][k]) - (2 * lamda * w[k][0])))
                        b = b + alpha * (y_train[j][0] - 2 * lamda * b)

            cost_values = cost_fun(w, x_test)+b
            count = 0
            mat = np.zeros((len(y_test), 1))
            i = 0
            for i in range(len(cost_values)):
                if cost_values[i][0] < 0:
                    mat[i][0] = -1
                else:
                    mat[i][0] = 1
            i = 0
            for i in range(len(y_test)):
                if mat[i][0] == y_test[i][0]:
                    count = count + 1

            #print("for alpha: ", alpha)
            accuracy = (count / len(y_test)) * 100
            # print("feature 1: ", listOfFeatures[k1])
            # print("feature 2: ", listOfFeatures[k2])
            # print("Accuracy: ", accuracy)
            # print("\n")
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestAlpha = alpha
                best1 = listOfFeatures[k1]
                best2 = listOfFeatures[k2]

print("Best alpha: ", bestAlpha)
print("Best feature 1: ", best1)
print("Best feature 2: ", best2)
print("Best accuracy: ", bestAccuracy," %")

