import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def main():
    
    # Import Boston housing data set into a dataframe
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # The dimension of the data set is (506, 13)
    
    # Split the data into training and test sets
    X_train, X_test, t_train, t_test = train_test_split(data, target, test_size=0.33, random_state=4894)
    
    # Iterate 3 times, twice for basis expansion and once for regular
    for i in range(3):
        boston_data_features = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        kf = KFold(n_splits=5)
        S = []
        test_errors = []
        validation_errors = []
        w_vector_list = []
        
        for k in range(13):
            cross_valid_errors = []
            for feature in boston_data_features:
                accum_err = 0
                
                # Apply K-fold cross validation on  X_train
                for train_index, test_index in kf.split(X_train[:,feature]):

                    kX_train, kX_test = X_train[train_index, feature], X_train[test_index, feature]
                    kt_train, kt_test = t_train[train_index], t_train[test_index]
                    
                    # When i=1 or i=2, the X_matrix and X_valid will be augemented to the function chosen
                    if i == 0:
                        X_matrix = compute_X_matrix(S, X_train, kX_train, train_index) # Compute training matrix
                        X_valid = compute_X_matrix(S, X_train, kX_test, test_index) # Compute validation matrix
                    elif i == 1:
                        X_matrix = np.power(compute_X_matrix(S, X_train, kX_train, train_index), 0.5)
                        X_valid = np.power(compute_X_matrix(S, X_train, kX_test, test_index), 0.5)
                    else:
                        X_matrix = np.power(compute_X_matrix(S, X_train, kX_train, train_index), 2)
                        X_valid = np.power(compute_X_matrix(S, X_train, kX_test, test_index), 2)

                    t_matrix = compute_t_matrix(kt_train)
                    t_valid = compute_t_matrix(kt_test)

                    w_vector = compute_w_vector(X_matrix, t_matrix)

                    error = compute_error(X_valid, t_valid, w_vector)
                    accum_err += error

                cross_valid_errors.append(accum_err/5)
    

            # Determine the min cross validation error of the features and append to validation_error list
            min_valid_err = min(cross_valid_errors)
            validation_errors.append(min_valid_err)

            # Find the index of the min validation error from the list
            index = cross_valid_errors.index(min_valid_err)
            
            # Remove feature from the the boston_data_features list given index
            feature_to_append = boston_data_features.pop(index)
            S.append(feature_to_append)

            # Train model using the whole data set
            if i == 0:
                newX_matrix = compute_X_matrix(S, X_train)
                newX_test = compute_X_matrix(S, X_test)
            elif i == 1:
                newX_matrix = np.power(compute_X_matrix(S, X_train), 0.5)
                newX_test = np.power(compute_X_matrix(S, X_test), 0.5)
            else:
                newX_matrix = np.power(compute_X_matrix(S, X_train), 2)
                newX_test = np.power(compute_X_matrix(S, X_test), 2)

            newt_matrix = compute_t_matrix(t_train)
            newt_test = compute_t_matrix(t_test)

            w_vector = compute_w_vector(newX_matrix, newt_matrix)
            w_vector_list.append(w_vector)
            
            test_error = compute_error(newX_test, newt_test, w_vector)
            test_errors.append(test_error)

        print("S = ", S)
#         print("w_vector = ", w_vector_list)
        print("")

        for k in range(13):
            print("k = {0} \t Validation Erorr = {1}\t Test Error = {2}".format(k+1,round(validation_errors[k],5), 
                                                                                round(test_errors[k], 5)))
            print("w_vector = ", w_vector_list[k])
            print("")
            
        # Plot validation and test errors
        title = ""
        if i == 0:
            title = "No Basis"
        elif i == 1:
            title = "sqrt(x)"
        else:
            title = "x^2"
            
        plotErrors(test_errors, validation_errors, title)

    
def plotErrors(test_errors, validation_errors, title):
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    plt.figure(1)
    plt.title("Error vs k - {}".format(title))
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.plot(x, test_errors, ls="-", marker = "o", color = "blue", label = "testing set")
    plt.plot(x, validation_errors, ls="-", marker = "o", color = "red", label = "validation set")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()
    
# Compute the X_matrix based on the feature set S
def compute_X_matrix(S, X_train, X_set=[-1], index=[-1]):
    
    if (X_set[0] == -1 and index[0] == -1):
        # Used for testing error 
        N = len(X_train)
    else:
        # Used for cross validation error
        N = len(index)
        
    ones_vector = np.ones(N)
    X_matrix = np.array([ones_vector])
    
    if (X_set[0] == -1 and index[0] == -1):
        for f in S:
            # Insert the whole feature into the X_matrix. Used when calculated test error for S set
            X_matrix = np.concatenate((X_matrix, [X_train[:, f]]), axis=0)
    else:
        # If there are features in set S, insert them into the X_matrix
        for f in S:
            X_matrix = np.concatenate((X_matrix, [X_train[index, f]]), axis=0)

        # Insert new feature to X_matrix
        X_matrix = np.concatenate((X_matrix, [X_set]), axis=0)

    return X_matrix.transpose()

def compute_t_matrix(t_train):
    t_matrix = np.array(t_train)
    return t_matrix

# Compute the vector of parameters using least squares method
def compute_w_vector(X_matrix, t_matrix):
    
    # Compute (1) = inverse(X(transpose)*X)
    XT_X_matrix = np.matmul(X_matrix.transpose(), X_matrix)
    XT_X_matrix_inv = np.linalg.inv(XT_X_matrix)
    
    # Compute (2) = X(transpose)*t
    XT_t_matrix = np.matmul(X_matrix.transpose(), t_matrix)
    
    # Compute w = (1)*(2)
    w_vector = np.matmul(XT_X_matrix_inv, XT_t_matrix)
    
    return w_vector

# Compute Mean Squared Error
def compute_error(X, t, w_vector):
    N = len(X)
    y_vector = np.matmul(X, w_vector)
    diff_matrix = np.subtract(y_vector, t)
    error = np.matmul(diff_matrix, diff_matrix)/N 
    return error
    
if __name__ == "__main__":
    main()
    