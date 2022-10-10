import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Run the code and all will be printed out in the console
# Was written in Jupyter Notebook!

def main():
    
    X_train = np.linspace(0.,1.,10) # training set
    X_valid = np.linspace(0.,1.,100) # validation set

    np.random.seed(4894)

    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)

    training_err_list = []
    valid_err_list = []
    
    for M in range(10):
        # 1. Develop X matrix
        X_matrix = compute_X_matrix(M, X_train)

        # 2. Develop vector t
        t_matrix = compute_t_matrix(t_train)
        # print("\nt = \n", t_matrix)

        # 3. Develop vector of parameters(w)
        w_vector = compute_w_vector(X_matrix, t_matrix)
        print("\nw = \n", w_vector)

        # Plot ftrue, fm, training, and validation set
        generate_plot(M, w_vector, X_train, X_valid, t_train, t_valid)

        # 4. Record predictor outputs based on training set and validation set
        predictor_output_training = compute_fM_output(M, w_vector, X_train)
        predictor_output_valid = compute_fM_output(M, w_vector, X_valid)

        # 5. Compute training and validation error
        training_err = compute_error(predictor_output_training, t_train)
        valid_err = compute_error(predictor_output_valid, t_valid)
        
        training_err_list.append(training_err)
        valid_err_list.append(valid_err)
        
        print("Training error = ", training_err)
        print("Validation error = ", valid_err)
        
    print("\nTraining Error List = ", training_err_list)
    print("\nValidation Error List = ", valid_err_list)
    
    generate_error_plot(training_err_list, valid_err_list)
    
    # --------------- Regularization of M=9, same steps as above but equations will vary --------------
    
    # Create the 10x9 matrix for M=9
    XX_train = standardizeFeatures(X_train)
    XX_valid = standardizeFeatures(X_valid)
    
    sc = StandardScaler()
    XX_train = sc.fit_transform(XX_train)
    XX_valid = sc.transform(XX_valid)
    
    # Define value of lambda
    lam1 = math.e**-28
    
    # Insert the 1's column in matrix making it a 10x10 matrix
    ones_column = [1,1,1,1,1,1,1,1,1,1]
    XX_matrix = np.insert(XX_train, 0, ones_column, axis=1)
    
    # Compute the t_matrix
    t_matrix = compute_t_matrix(t_train)
    
    # Compute the w_vector
    w_vector = compute_reg_w_vector(XX_matrix, t_matrix, lam1) 
    print("w_vector = ", w_vector)
    
    generate_plot(9, w_vector, X_train, X_valid, t_train, t_valid)
    
    # Compute the regularized cost
    reg_pred_output_training = compute_fM_output(9, w_vector, X_train)
    reg_pred_output_valid = compute_fM_output(9, w_vector, X_valid)
    
    training_reg_err = compute_error(reg_pred_output_training, t_train)
    valid_reg_err = compute_error(reg_pred_output_valid, t_valid)
    
    training_reg_err = compute_reg_error(training_reg_err, w_vector, lam1)
    valid_reg_err = compute_reg_error(valid_reg_err, w_vector, lam1)
    
    #generate_reg_error_plot(training_reg_err, valid_reg_err, w_vector)
    
def compute_X_matrix(M, X_train):
    X_matrix = np.array([])
    
    if M == 0:
        x_vector = np.array([[1,1,1,1,1,1,1,1,1,1]])
        X_matrix = x_vector #axis=0 creates new rows
    else:
        # Create the feature vector depending on the value of M
        for m in range(M+1):
            x_list = []

            # Compute the feature vector by raising the training set to the appropriate value of M
            # If M=1, raise all training set values to the power of 1, if M=2, raise to power of 2, etc
            for x in X_train:
                x_list.append(x**m)

            x_vector = np.array([x_list])

            # After feature vector has been computed, concatenate to X_matrix
            if m == 0:
                X_matrix = x_vector # X_matrix is vertical ones vector
            else:
                X_matrix = np.concatenate((X_matrix, x_vector), axis=0)
    
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

def compute_reg_w_vector(XX_matrix, t_matrix, lambda_val):
    
    XXT_XX_matrix = np.matmul(XX_matrix.transpose(), XX_matrix)
    XXT_t_matrix = np.matmul(XX_matrix.transpose(), t_matrix)
    
    # Since XX_matrix transpose multiplied by XX_matrix is a 10x10 matrix
    # B must be a 10x10 matrix
    B = np.array([])
    
    for row in range(10):
        row_list = []
        for col in range(10):
            
            # Make the diagonal elements 2*lambda while the rest are 0
            if row == col:
                if row == 0:
                    row_list.append(0)
                else:
                    row_list.append(2*lambda_val)
            else:
                row_list.append(0)
        
        row_vector = np.array([row_list])
        
        if row == 0:
            B = row_vector
        else:
            B = np.concatenate((B, row_vector), axis=0)
        
    XXT_XX_add_B = XXT_XX_matrix + ((len(XX_matrix)/2)*B)
    XXT_XX_add_B_inv = np.linalg.inv(XXT_XX_add_B)
    
    w_vector = np.matmul(XXT_XX_add_B_inv, XXT_t_matrix)
    
    return w_vector
            
def compute_fM_output(M, w_vector, X_set):
    predictor_results = []
    
    # Check if M=0, if so the result is a constant
    if M == 0:
        predictor_results.append(w_vector[0])
    else:
        for x in X_set:
            result = 0
            # Depending on the value of M raise the training set value of x to the respective M power
            for m in range(M+1):
                if m == 0:
                    result = w_vector[m]
                else:
                    result += w_vector[m]*(x**m)
        
            predictor_results.append(result)
            
    return predictor_results


def compute_error(predictor_output, true_output):
    error = 0
    
    for i in range(len(true_output)):
        if len(predictor_output) == 1:
            # If M=1, then the there is only one element in w, hence predictor_output[0]
            error += (predictor_output[0] - true_output[i])**2
        else:
            error += (predictor_output[i] - true_output[i])**2
    return error/len(true_output)

def compute_reg_error(error, w_vector, lambda_val):
    reg_error = error
    for w in w_vector:
        reg_error += lambda_val*(w**2)
    return reg_error
    

def standardizeFeatures(X_set):
    XX_matrix = np.array([])
    
    # For each value in the set, create a new feature vector, (x,x^2,x^3...,x^9)
    for x in X_set:
        XX_list = []
        for i in range(1,10):
            XX_list.append(x**i)
            
        XX_vector = np.array([XX_list])
        
        if XX_matrix.size == 0:
            XX_matrix = XX_vector
        else:
            XX_matrix = np.concatenate((XX_matrix, XX_vector), axis=0)
        
    return XX_matrix

def root_mean_squared(err_list):
    rms_list = np.array([])
    for i in err_list:
        rms_list = np.append(rms_list, [math.sqrt(i)])
    return rms_list

def generate_error_plot(training_err_list, valid_err_list):
    x = np.linspace(0, 9, 10)
    
    training_err = root_mean_squared(training_err_list)
    valid_err =  root_mean_squared(valid_err_list)
    
    plt.xlabel("M")
    plt.ylabel("Error (RMS)")
    plt.plot(x, training_err, ls="-", marker="o", color = "blue", label = "training")
    plt.plot(x, valid_err,  ls="-", marker="o", color = "red", label = "validation")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()
    
def generate_reg_error_plot(training_reg_err, valid_reg_err, w_vector):
    x = np.linspace(math.log(math.e**-40), math.log(math.e**-10), 10)
    
    squared_sum_w = 0
    
    for w in w_vector:
        squared_sum_w += w**2
        
    
    train_err = root_mean_squared(training_reg_err + x*squared_sum_w)
    valid_err = root_mean_squared(valid_reg_err + x*squared_sum_w)
    
    plt.xlabel("ln λ")
    plt.ylabel("Error (RMS)")
    plt.plot(x, train_err , ls="-", marker="o", color = "blue", label = "training")
    plt.plot(x, valid_err,  ls="-", marker="o", color = "red", label = "validation")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()
    
    
    
def generate_plot(M, w_vector, X_train, X_valid, t_train, t_valid):
    x = np.linspace(0, 1, 100)
    fM = 0
    
    if M == 0:
        fM = w_vector[0]*(x**0)
    elif M == 1:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1)
    elif M == 2:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2)
    elif M == 3:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3)
    elif M == 4:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3) + w_vector[4]*(x**4)
    elif M == 5:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3) + w_vector[4]*(x**4) + w_vector[5]*(x**5)
    elif M == 6:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3) + w_vector[4]*(x**4) + w_vector[5]*(x**5) + w_vector[6]*(x**6)
    elif M == 7:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3) + w_vector[4]*(x**4) + w_vector[5]*(x**5) + w_vector[6]*(x**6) + w_vector[7]*(x**7)
    elif M == 8:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3) + w_vector[4]*(x**4) + w_vector[5]*(x**5) + w_vector[6]*(x**6) + w_vector[7]*(x**7) + w_vector[8]*(x**8)
    else:
        fM = w_vector[0]*(x**0) + w_vector[1]*(x**1) + w_vector[2]*(x**2) + w_vector[3]*(x**3) + w_vector[4]*(x**4) + w_vector[5]*(x**5) + w_vector[6]*(x**6) + w_vector[7]*(x**7) + w_vector[8]*(x**8) + w_vector[9]*(x**9)
    
    plt.title("M = {0}".format(M))
    plt.xlabel("x")
    plt.ylabel("t")
    plt.plot(x, fM, color = "red", label = "fM")
    plt.plot(x, np.sin(4*np.pi*x), color = "green", label = "ftrue")
    plt.plot(X_train, t_train, "o", color = "blue", label = "training set")
    plt.plot(X_valid, t_valid, "o", color = "violet", label = "validation set")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()
    
if __name__ == "__main__":
    main()
