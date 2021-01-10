import numpy as np
import pandas as pd


def H(data):
    featureValues, featureValuesCounts = np.unique(data, return_counts=True)
    entropy = 0.0
    for i in range(len(featureValues)):
        entropy += np.sum(
            [(-featureValuesCounts[i] / np.sum(featureValuesCounts)) * np.log2(featureValuesCounts[i] / np.sum(featureValuesCounts))])
    return entropy

def informationGain(data, feature, target):
    entropyTarget = H(data[target])
    featureValues, featureValuesCounts = np.unique(data[feature], return_counts=True)
    entropyTargetToFeature = 0.0
    for i in range(len(featureValues)):
        entropyTargetToFeature += np.sum((featureValuesCounts[i] / np.sum(featureValuesCounts)) * H(data.where(data[feature] == featureValues[i]).dropna()[target]))
    infoGain = entropyTarget - entropyTargetToFeature
    return infoGain


def getMaxInfo(data, featureList):
    maxFeature = informationGain(data, featureList[0], 0)
    maxIndex = 0
    for i in range(1, len(featureList)):
        temp = informationGain(data, featureList[i], 0)
        if temp > maxFeature:
            maxFeature = temp
            maxIndex = i
    return maxIndex


def decisionTree(data, featureList):
    isPure = len(np.unique(data[0])) == 1
    if isPure:
        return np.unique(data[0])[0]

    elif len(featureList) == 0:
        return np.unique(data[0])[np.argmax(np.unique(data[0], return_counts=True)[1])]

    else:
        maxFeatureIndex = getMaxInfo(data, featureList)
        maxFeature = featureList[maxFeatureIndex]
        tree = {maxFeature: {}}
        featureList.pop(maxFeatureIndex)
        for featureValue in np.unique(data[maxFeature]):
            subData = data.where(data[maxFeature] == featureValue).dropna()
            subtree = decisionTree(subData, featureList)
            tree[maxFeature][featureValue] = subtree
        return tree


def predictPoliticalParty(input, tree):
    for key in list(input.keys()):
        if key in list(tree.keys()):
            try:
                 result = tree[key][input[key]]
            except:
                continue
            result = tree[key][input[key]]
            if type(result) is dict:
                return predictPoliticalParty(input, result)
            else:
                return result


def sizeOfTree(tree, cnt):
   for node in tree:
     if type(tree[node]) is dict:
       cnt += sizeOfTree(tree[node], len(tree[node]))
     else:
       cnt += 1
   return cnt

def partA_1(df_):
    for iter in range(5):
        allDataCount = 3             # number of values
        featureList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        count = 0
        train_data = df_.sample(frac=.25)

        while True:
            if count == 16:
                break
            count = 0
            train_data = df_.sample(frac=.25)
            for i_ in range(1, 17):
                if allDataCount == len(np.unique(train_data[i_])):
                    count = count + 1
                else:
                    break

        test_data = df_.drop(train_data.index.tolist())
        testQueries = test_data.iloc[:, 1:17].to_dict(orient="records")
        tree = decisionTree(train_data, featureList)
        count = 0
        for j_ in range(len(testQueries)):
            if predictPoliticalParty(testQueries[j_], tree) == np.array(test_data[0])[j_]:
                count = count + 1
        print(iter + 1, ")")
        print("Size: ", sizeOfTree(tree, len(tree)))
        print("Accuracy: ", (count / len(test_data)) * 100, " %")

def partA2(random, df):
    featureList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    allDataCount = 2
    count = 0
    train_data = df.sample(frac=random)
    while True:
        if count == 16:
            break
        count = 0
        train_data = df.sample(frac=random)
        for index in range(1, 17):
            if allDataCount == len(np.unique(train_data[index])):
                count = count + 1
            else:
                break
    test_data = df.drop(train_data.index.tolist())
    testQueries = test_data.iloc[:, 1:17].to_dict(orient="records")
    tree = decisionTree(train_data, featureList)
    count = 0
    for jIndex in range(len(testQueries)):
        if predictPoliticalParty(testQueries[jIndex], tree) == np.array(test_data[0])[jIndex]:
            count = count + 1
    size = sizeOfTree(tree, len(tree))
    accuracy = ((count / len(test_data)) * 100) #lol
    print(accuracy)
    return size, accuracy



df = pd.read_csv('house-votes-84.data.txt', header=None)

# PART A - 1
print("\nPART A - 1")
partA_1(df)

for i in df.index:
   for j in range(1, len(df.loc[i])):
       majorityVote = np.unique(df.drop(0, axis=1).loc[i])[np.argmax(np.unique(df.drop(0, axis=1).loc[i], return_counts=True)[1])]
       if majorityVote == '?':
           majorityVote = np.unique(df.drop(0, axis=1).loc[i])[np.argmin(np.unique(df.drop(0, axis=1).loc[i], return_counts=True)[1])]
           if majorityVote == 'n':
               majorityVote = 'y'
           else:
               majorityVote = 'n'
       if df.loc[i][j] == "?":
           df.loc[i][j] = majorityVote

print("\nPART A - 2")
listOfRandom = [.3, .4, .5, .6, .7]
sizes = []
accuracies = []

for random in range(len(listOfRandom)):
    size, accuracy = partA2(listOfRandom[random], df)
    sizes.append(size)
    accuracies.append(accuracy)

print("mean of sizes: ", np.mean(sizes))
print("min of sizes: ", np.min(sizes))
print("max of sizes: ", np.max(sizes))
print("mean of accuracies: ", np.mean(accuracies), "%")
print("min of accuracies: ", np.min(accuracies), " %")
print("max of accuracies: ", np.max(accuracies), " %")
