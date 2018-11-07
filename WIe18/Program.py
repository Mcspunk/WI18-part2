# eigendecomposition
from numpy import array
from numpy.linalg import eig
class user:
    def __init__(self, id, name):
        self.name = name
        self.id = id
        self.friends = []

class review:
    def __init__(self, score, summary, content):
        self.score = score
        self.summary = summary
        self.content = content

def createDictionaryOfTermsFromFile():
    reviews = []

    with open("friendships.reviews.txt", "r") as ins:
        array = []
        index = 0
        for line in ins:
            if "review/score:" in line:
                score = line.split(": ")[1]
                ins.__next__()
                ins.__next__()
                summary = line.split("review/summary: ")[1]
                ins.__next__()
                content = line.split("review/text: ")[1]
                reviews.append(score, summary, content)
            array.append(line)

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
    A = array([[0,1,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0], [1,1,0,1,1,0,0,0,0],[0,0,1,0,1,1,1,0,0], [0,0,1,1,0,1,1,0,0], [0,0,0,1,1,0,1,1,0], [0,0,0,1,1,1,0,1,0], [0,0,0,0,0,1,1,0,1], [0,0,0,0,0,0,0,1,0]])
    #friendArray(loadUsersFromFile("c:/users/palmi/desktop/friendships.txt","r"))
    D = degress(A)
    L = laplacian(A,D)

    #print(L)
    #print("")

    eigenValues,eigenVectors = eig(L)

    idx = eigenValues.argsort()[:-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    print(eigenValues)
    print("")
    print(eigenVectors[7])
main()