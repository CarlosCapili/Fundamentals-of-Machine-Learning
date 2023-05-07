import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def main():
   
    # 1 -> spam | 0 -> nonspam
    # Read data spambase dataset downloaded from UCI machine learning repository
    dataset = pd.read_csv("spambase.data", header=None)
    X = dataset.iloc[:, :-1].values
    t = dataset.iloc[:, -1].values

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.33, random_state=4894)
   
    # Decision Tree Classifier between 2-400 leaves
    cv_scores = []
    display = 0
   
    # Create decision tree and use cross validation to obtain the accuracy score
    for i in range(2, 401):
        decision_tree = DecisionTreeClassifier(random_state=4894, max_leaf_nodes=i)
        score = cross_val_score(decision_tree, X_train, t_train, scoring="accuracy", cv=5)
        cv_scores.append(round(score.mean(), 5))
       
        if display == 12:
            print("Max Leaves = ", i)
            display = 0
        display += 1
   
    # Find highest accuracy score
    plotcv_err(cv_scores)
    max_score = max(cv_scores)
    max_leaves = cv_scores.index(max_score) + 2
    print("The best maximum number of leaves is {} with an accuracy of {}".format(max_leaves, max_score))
   
    # Test Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=4894, max_leaf_nodes=max_leaves)
    decision_tree = decision_tree.fit(X_train, t_train)
    dt_pred_score = round(decision_tree.score(X_test, t_test), 5)
    MR = 1 - dt_pred_score
    print("Decision Tree Accuracy Score = ", dt_pred_score) # Average score out of 5 cv folds
    print("Decision Tree Misclassification Rate = ", MR)

    
    # Accuracy Scores
    bg_test_score = []
    randfor_test_score = []
    AB_stump_test_score = []
    AB_max10_test_score = []
    AB_test_score = []
    
    for i in range(50, 2550, 50):
        print("Number of predictors = ", i)
        
        # Bagging 
        bag_class = BaggingClassifier(n_estimators=i,random_state=4894)
        bag_class.fit(X_train, t_train)
        bg_pred_score = round(bag_class.score(X_test, t_test), 5)
        bg_test_score.append(bg_pred_score)
        print("Bagging Classifier has been completed")
        
        # Random Forest 
        randfor = RandomForestClassifier(n_estimators=i, random_state=4894)
        randfor.fit(X_train, t_train)
        randfor_pred_score = round(randfor.score(X_test, t_test), 5)
        randfor_test_score.append(randfor_pred_score)
        print("Random Forest has been completed")
        
        # Adaboost with decision stumps
        AB_stump = AdaBoostClassifier(n_estimators=i, random_state=4894)
        AB_stump.fit(X_train, t_train)
        AB_stump_pred_score = round(AB_stump.score(X_test, t_test), 5)
        AB_stump_test_score.append(AB_stump_pred_score)
        print("AB with decision stumps has been completed")
        
        # Adaboost with decision trees with at most 10 leaves as base
        base_class10 = DecisionTreeClassifier(max_leaf_nodes=10, random_state=4894)
        AB_max10 = AdaBoostClassifier(base_estimator=base_class10, n_estimators=i, random_state=4894)
        AB_max10.fit(X_train, t_train)
        AB_max10_pred_score = round(AB_max10.score(X_test, t_test), 5)
        AB_max10_test_score.append(AB_max10_pred_score)
        print("AB with at most 10 leaves as base has been completed")
        
        # Adaboost with decision trees with on restriction depth or node as base
        base_class = DecisionTreeClassifier(random_state=4894)
        AB = AdaBoostClassifier(base_estimator=base_class, n_estimators=i, random_state=4894)
        AB.fit(X_train, t_train)
        AB_pred_score = round(AB.score(X_test, t_test), 5)
        AB_test_score.append(AB_pred_score)
        print("AB with no restriction depth has been completed\n")
    
    # Write scores to txt files since it takes a long time to run program    
    with open('decisiontreeCV.txt', 'w') as fp1:
        for listitem in cv_scores:
            fp1.write(f'{listitem}\n')

    with open('baggingclassifier.txt', 'w') as fp2:
        for listitem in bg_test_score:
            fp2.write(f'{listitem}\n')

    with open('randomforest.txt', 'w') as fp3:
        for listitem in randfor_test_score:
            fp3.write(f'{listitem}\n')

    with open('Adaboost_stump.txt', 'w') as fp4:
        for listitem in AB_stump_test_score:
            fp4.write(f'{listitem}\n')

    with open('Adaboost_max10.txt', 'w') as fp5:
        for listitem in AB_max10_test_score:
            fp5.write(f'{listitem}\n')

    with open('Adaboost.txt', 'w') as fp6:
        for listitem in AB_test_score:
            fp6.write(f'{listitem}\n')
            
    plotScores(dt_pred_score, bg_test_score, randfor_test_score, AB_stump_test_score, AB_max10_test_score, AB_test_score)
    print("PROGRAM COMPLETED")
                        
def plotcv_err(cv_scores):
    X_axis = np.arange(2,401)
    plt.title("Accuracy vs Number of Leaves")
    plt.xlabel("Number of Leaves")
    plt.ylabel("Accuracy")
    plt.plot(X_axis, cv_scores, color = "blue")
    plt.show()
    
def plotScores(dt, bg, randfor, AB_stump, AB_max10, AB):
    X_axis = np.arange(50, 2550, 50)
    plt.title("Accuracy vs Number of Predictors")
    plt.xlabel("Number of Predictors")
    plt.ylabel("Accuracy")
    plt.plot(X_axis, dt, color="blue", label="Decision Tree")
    plt.plot(X_axis, bg, color="red", label="Bagging")
    plt.plot(X_axis, randfor, color="green", label="Random Forest")
    plt.plot(X_axis, AB_stump, color="purple", label="Adaboost with Decision Stumps")
    plt.plot(X_axis, AB_max10, color="orange", label="Adaboost with max 10 leaves")
    plt.plot(X_axis, AB, color="cyan", label="Adaboost with no restriction on depth")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()

if __name__ == "__main__":
    main()