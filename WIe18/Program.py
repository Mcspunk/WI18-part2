# eigendecomposition
import keras
from numpy import array
from numpy.linalg import eig
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import  load_model
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
    train_lab = []
    for train_review in train_reviews:
        train_seq.append(train_review.featureVector)
        if((float (train_review.score)) < 3): train_lab.append(0)
        else: train_lab.append(1)





    test_reviews = createDictionaryOfTermsFromFile("C:/users/palmi/desktop/SentimentTestingData.txt")
    test_seq = []
    test_lab = []
    for test_review in test_reviews:
        test_seq.append(test_review.featureVector)
        if((float (test_review.score)) < 3): test_lab.append(0)
        else: test_lab.append(1)
    train_pad = pad_sequences(train_seq, maxlen=max_review_length)
    test_pad = pad_sequences(test_seq,maxlen=max_review_length)

    train_data = np.array(train_pad)
    train_labels = np.array(train_lab)
    test_data = np.array(test_pad)
    test_labels = np.array(test_lab)

    model = keras.Sequential([
        keras.layers.Dense(200,input_shape=(256,)),
        keras.layers.Dense(300, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.1)),
        keras.layers.Dense(1, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,decay=0.1),loss='mean_squared_error',metrics=['accuracy'],)

    model.fit(train_data,train_labels,epochs=5,batch_size=32)
    model.save("c:/users/palmi/desktop/savedModel.h5")
    #test_loss,test_acc = model.evaluate(test_data,test_labels)
    #print("Test acc", test_acc, "test loss", test_loss)
    predictions = model.predict(test_data)
    print(predictions[0])


def testModel(path):
    model = load_model(path)
    numpyVector = np.array([434, 435, 436, 437, 19, 20, 438, 9, 53, 42, 439, 440, 10, 96, 441, 442, 443, 32, 444, 32, 8, 57, 445, 446, 435, 397, 82, 364, 447, 448, 449, 158, 435, 96, 450, 24, 451, 452, 453, 47, 454, 455, 97, 456, 40, 457, 428, 435, 458, 32, 444, 32, 364, 34, 57, 459, 460, 9, 461, 13, 452, 462, 463, 5, 464, 32, 465, 16, 42, 466, 10, 467, 468, 44, 469, 27, 409, 470, 471, 27, 416, 472, 473, 274, 474, 44, 450, 37, 417, 324, 57, 475, 453, 446, 476, 477, 44, 364, 447, 478, 479, 480, 9, 364, 34, 10, 401, 401, 481, 482, 5, 483, 37, 484, 56, 13, 485, 44, 5, 220, 13, 486, 16, 487, 488, 72, 32, 489, 13, 490, 40, 491, 492, 493, 494, 417, 495, 339, 137, 69, -496, -497, -78, -498, -499, -67, -21, -487, -500, -501, -309, -502, -372, -503, -37, -504, -505, 35, 13, 452, 506, 32, 507, 508, 509, 17, 387, 30, 510, 74, 212, 511, 52, 48, 512, 513, 398, 514, 48, 515, 13, 516, 517, 40, 13, 518, 68, 32, 519, 520, 24, 13, 521, 522, 435, 22, 523, 524, 44, 525, 16, 68, 212, 526, 527, 68, 528, 48, 13, 466, 68, 529, 530, 13, 518, 44, 531, 532, 24, 13, 533, 35, 534, 535, 68, 51, -324, -32, -536, -537, -13, -466, -520, -538, -539, -13, -540, 0, 13, 541, 542, 543, 97, 13, 518, 544, 545, 13, 520, 546, 547, 487, 548, 544, 401, 40])
    model.predict(numpyVector)

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
    sentimentAnalysis()
    #testModel("c:/users/palmi/desktop/savedModel.h5")
main()