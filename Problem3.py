
import re
import collections
import os
import math

C = {'ham': 1.0, 'spam': 0.0}
docCount = {'ham': 0.0, 'spam': 0.0}

def extractData():
    data = {'ham': [], 'spam': []}
    cwd = os.getcwd()
    for fileName in os.listdir(cwd + '/train/ham'):
        words = re.findall(r'\w+', open(cwd + '/train/ham/' + fileName).read().lower())
        if len(words) > 0:
            data['ham'].extend(words)
        docCount['ham'] += 1.0
    for fileName in os.listdir(cwd + '/train/spam'):
        words = re.findall(r'\w+', open(cwd + '/train/spam/' + fileName, encoding="ISO-8859-1").read().lower())
        if len(words) > 0:
            data['spam'].extend(words)
        docCount['spam'] += 1.0
    return data

def extractVocabulary(data):
    vocab = []
    completeWordList = collections.Counter(data['ham'] + data['spam'])
    vocab = list(completeWordList.keys())
    return vocab

def trainMultinomialNB(data, V):
    prior = {}
    condprob = {}
    for c in C.keys():
        prior[c] = docCount[c] / sum(docCount.values())
        text_c = data[c]
        condprob[c] = {}
        T_ct = {}
        for t in V:
            T_ct[t] = text_c.count(t) * 1.0
        for t in V:
            condprob[c][t] = (T_ct[t] + 1.0) / (len(text_c) + len(V))
    return prior, condprob

def applyMultinomialNB(V, prior, condprob):
    accuracy = {'ham': 0.0, 'spam': 0.0}
    totalSize = 0.0
    for c in C.keys():
        for fileName in os.listdir(os.getcwd() + '/test/' + c):
            predictedClass = predictClass(os.getcwd() + '/test/' + c + '/' + fileName, prior, condprob)
            if predictedClass == c:
                accuracy[c] += 1.0
            totalSize += 1.0
    return (sum(accuracy.values()) / totalSize) * 100

def predictClass(filePath, prior, condProb):
    W = re.findall(r'\w+', open(filePath).read().lower())
    score = {'ham': 0.0, 'spam': 0.0}
    for c in C.keys():
        score[c] = math.log(prior[c])
        for t in W:
            if t in condProb[c]:
                score[c] += math.log(condProb[c][t])
    return max(score, key=score.get)        #returns the class with maximum value

if __name__ == '__main__':
    data = extractData()
    V = extractVocabulary(data)
    prior, condProb = trainMultinomialNB(data, V)
    print("Naive Bayes Accuracy : " + str(applyMultinomialNB(V, prior, condProb)))
