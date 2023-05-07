import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    
    # Import and extract data
    data = pd.read_csv("data_banknote_authentitcation.txt", header=None)
    X = data.iloc[:, :-1].values
    t = data.iloc[:, -1].values
    
    # Split data to training and remaining set - 80% training and 20% validation/test
    X_train, X_split, t_train, t_split = train_test_split(X, t, test_size=0.2, shuffle=True, random_state=4894)
    
    # Split remaining set into validation and test set equally
    X_valid, X_test, t_valid, t_test = train_test_split(X_split, t_split, test_size=0.5, random_state=4894)
    
    # Standardize the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_valid = sc.transform(X_valid)
    
    # Variables
    hidden_layer_units = [2,3,4,5] 
    alpha = 0.03
    epochs = 1000
    best_weight_vec = {}
    best_err = 10000
    best_hl_size = [] # 1st element is number of units in hidden layer 1 and 2nd is hidden layer 2
    
    # Try different unit sizes for both hidden layers
    for hl1_size in hidden_layer_units:
        for hl2_size in hidden_layer_units:
            
            lowest_err = 10000 # set a large value
            lowest_train_err = 10000
            w_vec_low_err = {} # weight vector associated with the lowest err
            
            # For each iteration the initial value of weights is different 
            for i in range(5):
                err = 0
                weight_vec= initialWeights(hl1_size, hl2_size, X_train, t_train)
                
                # Train Neural Network
                for j in range(epochs):
                    # 1. Forward Propagation
                    forward_prop_res = forwardProp(weight_vec, X_train)
                    # Validation Cross Entropy Loss
                    err = error(weight_vec, X_valid, t_valid)
                    # 2. Backward Propagation - Stochastic Gradient Descent
                    gradientRes = computeGradient(X_train, t_train, weight_vec, forward_prop_res)
                    # 3. Update Weight Vector
                    weight_vec = updateWeights(alpha, weight_vec, gradientRes)
                    
                    # For each epoch take the lowest validation entropy loss and the associated weight vector
                    if err < lowest_err:
                        lowest_err = err
                        w_vec_low_err = weight_vec

                    if err < best_err:
                        best_err = err
                        best_weight_vec = weight_vec
                        best_hl_size = [hl1_size, hl2_size]
            
            print("hl1_size = {}\thl2_size = {}\tValid CEL = {:0.5f}\tMR = {:0.5f}".format(hl1_size, hl2_size, lowest_err[0], 
                                                                              misclass_rate(w_vec_low_err, X_train, t_train)))

    print("---------------------------------------------")
    print("Best Model Info/Stats")
    print("Best Hidden Layer Sizes = ", best_hl_size)
    print("Number of Units in Hidden Layer 1 = ", best_hl_size[0])
    print("Number of Units in Hidden Layer 2 = ", best_hl_size[1])
    print("Vector of Weights = ", best_weight_vec)
    print("Misclassification Rate on Test Set = ", misclass_rate(best_weight_vec, X_test, t_test))
    plotLearningCurve(best_weight_vec, X_train, t_train, hl1_size, hl2_size, epochs, alpha)
                
# Activation Function ReLU
def ReLU(z):
    # Check if an element in the list is less than 0, then change its value to 0
    z[z < 0] = 0
    return z

# Cost function
def cross_entropy_loss(y, t):
    N = len(t)
    cost = 0
    for i in range(N):
        # np.logaddexp was used to avoid division by zero
        cost += t[i] * np.logaddexp(0, y[i]) + (1-t[i]) * np.logaddexp(0, 1-y[i])
        
    return cost / N

def error(weight_vec, X, t):
    result = forwardProp(weight_vec, X)
    err = cross_entropy_loss(result["outl_output"].T, t)
    return err

def misclass_rate(weight_vec, X, t):
    rate = 0
    
    result = forwardProp(weight_vec, X)["outl_output"]
    result = np.rint(result) != t / t.shape
    result = (np.array(result[0], dtype=bool)).astype(int)
    
    for i in range(len(t)):
        if result[i] != t[i]:
            rate += 1
            
    rate = rate/len(t)
    return rate 

# Initialize weights
def initialWeights(hl1_size, hl2_size, X_set, t_set):
    w1 = np.random.randn(hl1_size, X_set.shape[1])
    w2 = np.random.randn(hl2_size, hl1_size)
    w3 = np.random.randn(1, hl2_size)
    return {"w1" : w1, "w2" : w2, "w3" : w3}

# Forward Propagation
def forwardProp(weight_vec, X):
    # Use a dictionary to make it easier to access values
    results = {} 
    
    # Compute z then h(output) for each layer
    
    # Hidden Layer 1
    results["hl1_z"] = np.dot(weight_vec["w1"], X.T)
    results["hl1_output"] = ReLU(results["hl1_z"])
    
    # Hidden Layer 2
    results["hl2_z"] = np.dot(weight_vec["w2"], results["hl1_output"])
    results["hl2_output"] = ReLU(results["hl2_z"])
    
    # Output Layer
    results["outl_z"] = np.dot(weight_vec["w3"], results["hl2_output"])
    results["outl_output"] = ReLU(results["outl_z"])
    
    return results

# Backward Propagation
def computeGradient(X, t, weight_vec, forwardPropRes):
    results = {}
    m_inv = 1/len(t.T)
    X = X.T
    t = t.T
   
    # Output Layer
    dA = m_inv * (forwardPropRes["outl_output"] - t)
    dZ = dA
    results["dW3"] = m_inv * np.dot(dZ, forwardPropRes["hl2_output"].T)
    
    # Hidden Layer 2
    dA = np.dot(weight_vec["w3"].T, dZ)
    dZ = np.multiply(dA, np.where(forwardPropRes["hl2_output"] > 0, 1, 0))
    results["dW2"] = m_inv * np.dot(dZ, forwardPropRes["hl1_output"].T)
    
    # Hidden Layer 1
    dA = np.dot(weight_vec["w2"].T, dZ)
    dZ = np.multiply(dA, np.where(forwardPropRes["hl1_output"] > 0, 1, 0))
    results["dW1"] = m_inv * np.dot(dZ, X.T)
    
    return results

# Update Weights
def updateWeights(alpha, weight_vec, gradientRes):
    w1 = weight_vec["w1"] - alpha * gradientRes["dW1"]
    w2 = weight_vec["w2"] - alpha * gradientRes["dW2"]
    w3 = weight_vec["w3"] - alpha * gradientRes["dW3"]
    return {"w1" : w1, "w2" : w2, "w3" : w3}

def plotLearningCurve(weight_vec, X_train, t_train, hl1_size, hl2_size, epochs, alpha):
    # Train Neural Network
    err = []
    
    for i in range(epochs):
        forward_prop_res = forwardProp(weight_vec, X_train)
        loss = error(weight_vec, X_train, t_train)
        err.append(loss)
        gradientRes = computeGradient(X_train, t_train, weight_vec, forward_prop_res)
        weight_vec = updateWeights(alpha, weight_vec, gradientRes)

    plt.title("Learning Curve")
    plt.plot(np.linspace(0, epochs, epochs), err)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()
