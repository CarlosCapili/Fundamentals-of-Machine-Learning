import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, accuracy_score, f1_score, PrecisionRecallDisplay)

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def main():
    data = load_breast_cancer()
    features = data.data
    target = data.target

    # malignant = 1 and benign = 0
    # Split the data into training and test sets
    X_train, X_test, t_train, t_test = train_test_split(features, target, test_size=0.33, random_state=4894)
   
    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Logistic Regression using my implentation
    print("--------------------------LOGISTIC REGRESSION USING MY IMPLEMENTATION--------------------------\n")
    logReg(X_train, X_test, t_train, t_test)
    
    # Logistic Regression using sklearn
    print("--------------------------LOGISTIC REGRESSION USING SKLEARN--------------------------\n")
    logRegSkLearn(X_train, X_test, t_train, t_test)
    
    # k-nearest neighbor classifier
    print("--------------------------KNN USING MY IMPLEMENTATION--------------------------\n")
    kNN(X_train, X_test, t_train, t_test)
    
    # k-nearest neighbor using scikit-learn
    print("--------------------------KNN USING SKLEARN--------------------------\n")
    kNNSkLearn(X_train, X_test, t_train, t_test)

    
def logReg(X_train, X_test, t_train, t_test):
    # Logistic Regression using Batch Gradient Descent
    lr = 0.5
    w_vec = np.zeros(31)
    X_matrix = compute_X(X_train)
    N = len(X_matrix[:,0])
    
    # Training linear classifier
    for i in range(3000):
        z = np.matmul(X_matrix, w_vec)
        y = sigmoid_func(z)
        
        gradient_w = np.matmul(X_matrix.T, (y - t_train))
        w_vec = w_vec - (lr/N)*gradient_w
    
    print("w_vec = ", w_vec)
    
    # Test linear classifier
    X_test_matrix = compute_X(X_test)
    predictor = prediction(X_test_matrix, w_vec) 
    classify = threshold(predictor, t_test, 0.5)
    
    # Calculate Misclassification Rate, Precision, Recall, and F1 score on the TEST DATA
    MR = misclass_rate(classify, t_test)
    P = precision(classify, t_test)
    R = recall(classify, t_test)
    F1 = F1_Score(P,R)
    
    print("\nMisclassification Rate = ", round(MR, 5))
    print("F1 Score = ", round(F1, 5))
    
    # Plot PR curve
    sort_pred = sorted(predictor)
    N_thresh = len(sort_pred)
    P_list = []
    R_list = []
    
    # Calculate precision and recall for each threshold in threshold_list
    for i in range(N_thresh):
        classify = threshold(predictor, t_test, sort_pred[i])
        P = precision(classify, t_test)
        R = recall(classify, t_test)
        P_list.append(P)
        R_list.append(R)
    
    plotPR(P_list, R_list)
    
def logRegSkLearn(X_train, X_test, t_train, t_test):
    # Train model and use it to predict with on the test set
    logreg = LogisticRegression(random_state=4894).fit(X_train, t_train)
    print("w_vec = ", logreg.coef_)
    prediction = logreg.predict(X_test)
    
    # Calculate Misclassification Rate, Precision, Recall, and F1 score
    MR = 1 - accuracy_score(t_test, prediction)
    P, R, thresholds = precision_recall_curve(t_test, prediction)
    F1 = f1_score(t_test, prediction)
    
    print("\nMisclassification Rate = ", round(MR, 5))
    print("F1 Score = ", round(F1, 5))
    
    # Plot Precision/Recall PR Curve
    display = PrecisionRecallDisplay(precision = P, recall = R)
    display.plot()
    
def kNN(X_train, X_test, t_train, t_test):
    kf = KFold(n_splits=5)
    kfold_avg_err = []
    
    # Find k-nearest neighbour and select best k using K-fold cross-validation
    for k in range(1,6):

        # Used to hold the error of each fold
        kFold_err = []
        validation_err = []

        for train_index, test_index in kf.split(X_train):
            kX_train, kX_valid = X_train[train_index], X_train[test_index]
            kt_train, kt_valid = t_train[train_index], t_train[test_index]

            # Holds k nearest points
            valid_kNN_list = []

            # Repeat for each row in the validation set
            for i in range(len(kX_valid)):
                dist_list = []
                sort_points = []

                for j in range(len(kX_train)):
                    dist = computeDistance(kX_valid[i], kX_train[j])

                    # Append distance and keep track of row
                    dist_list.append([dist, j])

                # Sort from least to greatest distance and take only the jth kX_train element removing the distance    
                dist_list.sort()
                for d, j in dist_list:
                    sort_points.append(j)

                # Classify element based on k-NN
                classify_ele = classifyElement(sort_points, kt_train, k)
                valid_kNN_list.append(classify_ele)

            valid_kNN_list = np.array(valid_kNN_list)

            # Compute Error for fold
            err = computeError(valid_kNN_list, kt_valid)
            validation_err.append(err)
    
        avg_err = avgErr(validation_err)
        print("When k = {} fold error = {}".format(k, validation_err))
        print("Average Error = ", avg_err)
        
        kfold_avg_err.append(avg_err)
        
    print("\nK-Fold Errors: ", kfold_avg_err)
   
    min_kfold_err = min(kfold_avg_err)
    index = kfold_avg_err.index(min_kfold_err)
    k = index + 1
    print("The min error is found when k =", k)
    
    # Using the best classifier on whole data set
    valid_kNN_list = []
    for i in range(len(X_test)):
        dist_list = []
        sort_points = []

        for j in range(len(X_train)):
            dist = computeDistance(X_test[i], X_train[j])
            dist_list.append([dist, j])
        
        dist_list.sort()
        for d, j in dist_list:
            sort_points.append(j)
        
        classify_ele = classifyElement(sort_points, t_train, k) # CHANGE 3 TO K EHRE
        valid_kNN_list.append(classify_ele)
    
    valid_kNN_list = np.array(valid_kNN_list)
        
    MR = misclass_rate(valid_kNN_list, t_test)
    P = precision(valid_kNN_list, t_test)
    R = recall(valid_kNN_list, t_test)
    F1 = F1_Score(P, R)
    
    print("\nMisclassification Rate = ", round(MR, 5))
    print("F1 Score = ", round(F1, 5))
    print("")

def kNNSkLearn(X_train, X_test, t_train, t_test):
    kNN_score_list = []
    
    for i in range(1, 6):
        KNN = KNeighborsClassifier(n_neighbors=i)
        CV_sklearn = cross_val_score(KNN, X_train, t_train, scoring="accuracy", cv=5)
        print("For k = ", i, "CV Accuracy Score = ",  CV_sklearn)
        
        kNN_score_list.append(round(np.mean(CV_sklearn), 5))
        
    print("\nAccuracy Scores = ", kNN_score_list)
    
    # Find highest score
    max_score = max(kNN_score_list)
    index = kNN_score_list.index(max_score)
    k = index + 1
    print("The max score is found when k = ", k)
    
    # Use k for whole data set, predict class labels, and get accuracy score
    model_test = KNeighborsClassifier(n_neighbors=k).fit(X_train, t_train)
    y_test = model_test.predict(X_test)
    score_test = round(model_test.score(X_test, t_test,), 5)
    
    MR = misclass_rate(y_test, t_test)
    P = precision(y_test, t_test)
    R = recall(y_test, t_test)
    F1 = F1_Score(P, R)
        
    print("\nMisclassification Rate = ", round(MR, 5))
    print("F1 Score = ", round(F1, 5))
    print("")
    
    
# LOGISTIC REGRESSION FUNCTIONS -------------------------------------------------------
def prediction(X_test_matrix, w_vec):
    result = []

    # Calculate z
    z = np.matmul(X_test_matrix, w_vec)
    
    for i in range(len(z)):
        y = sigmoid_func(z[i])
        result.append(y)
    
    return result

def threshold(predictor, t_test, theta):
    classified = []
    
    # If the value of the sigmoid function is greater than equal to theta assign a class of 1
    for i in range(len(predictor)): 
        if predictor[i] >= theta:
            classified.append(1)
        else:
            classified.append(0)  
    return np.array(classified)
    
def sigmoid_func(z):
    return (1/(1+np.exp(-z)))

def compute_X(X_data):
    N = len(X_data[:,0])
    ones_vector = np.ones(N) # create ones row vector 
    X_matrix = np.array([ones_vector]).reshape(-1,1) # reshape to column vector
    X_matrix = np.concatenate((X_matrix, X_data), axis=1) # add ones column vector to training data
    return X_matrix


# KNN FUNCTIONS -------------------------------------------------------

# Find error for the k-cross fold
def computeError(t_pred, t_valid):
    err = 0
    
    for i in range(len(t_valid)):
        if (t_pred[i] != t_valid[i]):
            err += 1
    
    return round(err/len(t_pred), 5)

def avgErr(validation_err):
    err = 0
    for i in validation_err:
        err += i
    
    return round(err/len(validation_err), 5)

def classifyElement(sort_points, kt_train, k):
    NN = []
    classify = []
    classification = 0
    count_ones = 0
    count_zeros = 0
        
    # Append nearest neighbours to NN list
    for i in range(k):
        NN.append(sort_points[i])
        
    # Find label in kt_train
    classify = kt_train[NN]
    
    for i in classify:
        if i == 0:
            count_zeros += 1
        else:
            count_ones += 1
    
    # If the number of 1's or 0's are equal, assign a value of 1 (malignant), better to be wrong about it than not
    if ((k == 2 or k == 4) and count_zeros == count_ones):
        classification = 1
    else:
        if count_zeros > count_ones:
            classification = 0
        else:
            classification = 1
    
    return classification

# Calculate the distance between rows of data
def computeDistance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)


# METRICS -------------------------------------------------------
def F1_Score(P, R):
    return (2*P*R/(P+R))
    
def misclass_rate(predictor, actual):
    rate = 0
    
    for i in range(len(predictor)):
        if predictor[i] != actual[i]:
            rate += 1
            
    rate = rate/len(predictor)
    return rate
        
def precision(predictor, actual):
    TP = 0 # True positive
    FP = 0 # False positive
    
    # Test for true positives and false positives
    for i in range(len(predictor)):
        if (predictor[i] == 1 and actual[i] == 1):
            TP += 1
        elif (predictor[i] == 1 and actual[i] == 0):
            FP += 1
            
    if TP == 0 and TP+FP == 0:
        P = 1
    else:
        P = TP/(TP+FP)
    return P

def recall(predictor, actual):
    TP = 0 # True positive
    FN = 0 # False negative

    # Test for true positives and false negatives
    for i in range(len(predictor)):
        if (predictor[i] == 1 and actual[i] == 1):
            TP += 1
        elif (predictor[i] == 0 and actual[i] == 1):
            FN += 1
        
    R = TP/(TP+FN)
    return R

def plotPR(P_list, R_list):
    plt.title("Precision/Recall PR Curve")
    plt.plot(R_list, P_list)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
            
if __name__ == "__main__":
    main()