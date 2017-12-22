from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
from ensemble import AdaBoostClassifier
from feature import NPDFeature
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # write your code here
    X=[]
    y=[]
    
    for i in range(0,500):
        path="C:\\Users\\Administrator\\Desktop\\ML2017-lab-03\\datasets\\original\\face\\face_%.3d.jpg" % (i)
        img=Image.open(path).convert('L').resize((24,24))
        X.append(NPDFeature(np.array(img)).extract())
        y.append(1)
        print(i)

    for i in range(0,500):
        path="C:\\Users\\Administrator\\Desktop\\ML2017-lab-03\\datasets\\original\\nonface\\nonface_%.3d.jpg" % (i)
        img=Image.open(path).convert('L').resize((24,24))
        X.append(NPDFeature(np.array(img)).extract())
        y.append(-1)
        print(i)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  
    ada = AdaBoostClassifier(DecisionTreeClassifier,20)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))
    
    ada.fit(X_train,y_train)
    h = ada.predict(X_test)
    yes=0
    no=0
    for i in range(0,len(h)):
        if (h[i] == y_test[i]): yes += 1
        if (h[i] != y_test[i]): no += 1
    print(yes,"   ",no)
    
    report = classification_report(y_test, h, target_names=["nonface", "face"])
    
    file = open('report.txt', 'w')
    file.write(report)
    file.close()