# eigendecomposition
from numpy import array
from numpy.linalg import eig
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

class user:
    def __init__(self, id, name):
        self.name = name
        self.id = id
        self.friends = []

class Review:
    def __init__(self, score, summary, content):
        self.score = score
        self.summary = summary
        self.content = content
        self.featureVector = None

def createDictionaryOfTermsFromFile(path):
    reviews = []
    bagOfWords = {}
    with open(path, "r") as ins:
        for line in ins:
            if "review/score:" in line:
                score = line.split(": ")[1].replace("\n","")
            if "review/summary:" in line:
                summary = remove_html_tags(line.split("review/summary: ")[1].lower().replace("\n","")).replace("'","")
            if "review/text:" in line:
                content = remove_html_tags(line.split("review/text: ")[1].lower().replace("\n","")).replace("'","")
                reviews.append(Review(score, summary, content))
    wordIndexCounter = 0
    for review in reviews:
        for word in review.summary.split(' '):
            if word not in bagOfWords:
                bagOfWords[word] = wordIndexCounter
                wordIndexCounter += 1
        for word in review.content.split(' '):
            if word not in bagOfWords:
                bagOfWords[word] = wordIndexCounter
                wordIndexCounter += 1
    for review in reviews:
        createFeatureVector(review,bagOfWords)
    return reviews


def sentimentAnalysis():
    max_review_length = 256;
    train_reviews = createDictionaryOfTermsFromFile("C:/users/palmi/desktop/SentimentTrainingData.txt")
    train_seq = []
    for train_review in train_reviews:
        train_seq.append(train_review.featureVector)
    test_reviews = createDictionaryOfTermsFromFile("C:/users/palmi/desktop/SentimentTestingData.txt")
    test_seq = []
    for test_review in test_reviews:
        test_seq.append(test_review.featureVector)
    train_pad = pad_sequences(train_seq, maxlen=max_review_length)
    test_pad = pad_sequences(test_seq,maxlen=max_review_length)



def createFeatureVector(review, bagOfWords):
    negativeFlag = False
    negationWords = ["never","no","nothing","nowhere","not","havent","hasnt","hadnt","cant","couldnt","shouldnt","wont","wouldnt","dont","dosnt","didnt","isnt","arent","aint"]
    featureVector = []
    for word in review.summary.split(' '):
        if negativeFlag:
            featureVector.append((-1)*bagOfWords[word])
        else: featureVector.append(bagOfWords[word])
        if word in negationWords: negativeFlag = True
        if '.' in word or '?' in word or ':' in word or '!' in word or ';' in word:
            negativeFlag = False
    negativeFlag = False
    for word in review.content.split(' '):
        if negativeFlag:
            featureVector.append((-1)*bagOfWords[word])
        else: featureVector.append(bagOfWords[word])
        if word in negationWords: negativeFlag = True
        if '.' in word or '?' in word or ':' in word or '!' in word or ';' in word:
            negativeFlag = False
    review.featureVector = featureVector[0:256]






def loadUsersFromFile(path,mode):
    users = {}
    file = open(path,mode)
    lines = file.readlines()
    lineIndex = 0
    userCounter = 0
    for line in lines:
        if (lineIndex % 5 == 0):
            name = line.split(': ')[1].replace("\n","")
            users[name] = user(userCounter,name)
            userCounter += 1
        lineIndex += 1

    userCounter = 0

    for line in lines:
        if(line.startswith("friends")):
            friendNames = line.split('\t')
            userName = NameOnIndex(userCounter,users)
            for friendName in friendNames[1:]:
                users.get(userName).friends.append(users[friendName.replace("\n","")])
            userCounter += 1
    return users


def NameOnIndex(index,users):
    for key,value in users.items():
        if(value.id == index): return key


def friendArray(users):
    Arr = []
    userVec = []

    for OuterUser in users:
        for InnerUser in users:
            userVec.append(0)
        Arr.append(userVec)
        userVec = []

    for user in users.values():
        Arr[user.id][user.id] = len(user.friends)
        for friend in user.friends:
            Arr[user.id][friend.id] = -1

    print(Arr[0])

def remove_html_tags(text):
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def degress(arr):
    vec = []
    counter = 0
    for vector in arr:
        for num in vector:
            if num == 1: counter += 1
        vec.append(counter)
        counter = 0
    return vec

def laplacian(arr,deg):
    L = []
    currentVec = []
    x = 0
    y = 0
    for vector in arr:
        for num in vector:
            if(x == y): currentVec.append(deg[x])
            else: currentVec.append(num*-1)
            x += 1
        L.append(currentVec)
        currentVec = []
        y += 1
        x = 0
    return L

def main():
    #A = array([[0,1,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0], [1,1,0,1,1,0,0,0,0],[0,0,1,0,1,1,1,0,0], [0,0,1,1,0,1,1,0,0], [0,0,0,1,1,0,1,1,0], [0,0,0,1,1,1,0,1,0], [0,0,0,0,0,1,1,0,1], [0,0,0,0,0,0,0,1,0]])
    #friendArray(loadUsersFromFile("c:/users/palmi/desktop/friendships.txt","r"))
    #D = degress(A)
    #L = laplacian(A,D)

    #print(L)
    #print("")

    #eigenValues,eigenVectors = eig(L)

    #idx = eigenValues.argsort()[:-1]
    #eigenValues = eigenValues[idx]
    #eigenVectors = eigenVectors[:, idx]
    #print(eigenValues)
    #print("")
    #print(eigenVectors[7])
    createDictionaryOfTermsFromFile()
main()