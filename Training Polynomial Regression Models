import numpy as np
import matplotlib.pyplot as plt

def main():
    
    X_train = np.linspace(0.,1.,10) # training set
    X_valid = np.linspace(0.,1.,100) # validation set

    np.random.seed(4894)  # Student Number = 400184894

    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)
    
    while True:
        M = input("Enter a value for the model capacity, M, between 0-9: ")
        M = int(M)
        
        # 1. Develop feature matrix (X), passing arguements understand that M is = argument-1
        X_matrix = compute_X_matrix(M, X_train)

        # Display X Matrix
        # print("\nX = \n", X_matrix)
        # print("\nX^T = \n", X_matrix.transpose())

        # 2. Develop true label matrix (t)
        t_matrix = compute_t_matrix(t_train)
        # print("\nt = \n", t_matrix)

        # 3. Develop vector of parameters(w), and obtain predictor model fM
        w_vector = compute_w_vector(X_matrix, t_matrix)
        print("\nw = \n", w_vector)

        # Plot ftrue, fm, training, and validation set
        generate_plot(M, w_vector, X_train, X_valid, t_train, t_valid)


        # 4. Record predictor outputs based on training set and validation set
        predictor_output_training = compute_fM_output(M, w_vector, X_train)
        # print("\npredictor_output_training = \n", predictor_output_training)

        predictor_output_valid = compute_fM_output(M, w_vector, X_valid)
        #print("\predictor_output_valid = \n", predictor_output_valid)

        # 5. Compute training and validation error
        training_err = compute_error(predictor_output_training, t_train)
        valid_err = compute_error(predictor_output_valid, t_valid)
        
        print("Training error = ", training_err)
        print("Validation error = ", valid_err)

        # Plot ftrue, fm, training, and validation set
        print("\nPredictor for M={0} has been trained. Do you want to exit? (y/n)".format(M))
        train_again = input()
        
        if train_again == "y":
            break
            

def compute_X_matrix(M, X_train):
    X_matrix = np.array([])
    
    if M == 0:
        feature_vector = np.array([[1,1,1,1,1,1,1,1,1,1]])
#         print("\nFeature Vector for M={0}".format(M))
#         print(feature_vector)
        X_matrix = feature_vector #axis=0 creates new rows
    else:
        # Create the feature vector depending on the value of M
        for m in range(M+1):
            feature_list = []

            # Compute the feature vector by raising the training set to the appropriate value of M
            # If M=1, raise all training set values to the power of 1, if M=2, raise to power of 2, etc
            for features in X_train:
                feature_list.append(pow(features,m))

            feature_vector = np.array([feature_list])

#             print("\nFeature Vector for M={0}".format(m))
#             print(feature_vector)

            # After feature vector has been computed, concatenate to X_matrix
            if m == 0:
                X_matrix = feature_vector # X_matrix is vertical ones vector
            else:
                X_matrix = np.concatenate((X_matrix, feature_vector), axis=0)
    
    return X_matrix.transpose()
    
def compute_t_matrix(t_train):
    t_matrix = np.array(t_train)
    return t_matrix

# Compute the vector of parameters using least squares method
def compute_w_vector(X_matrix, t_matrix):
    
    # Compute (1) = inverse(X(transpose)*X)
    XT_X_matrix = np.matmul(X_matrix.transpose(), X_matrix)
#     print("\nXTX = \n", XT_X_matrix)
    XT_X_matrix_inv = np.linalg.inv(XT_X_matrix)
#     print("\nXTX inverse = \n", XT_X_matrix_inv)
    
    # Compute (2) = X(transpose)*t
    XT_t_matrix = np.matmul(X_matrix.transpose(), t_matrix)
#     print("\nXT_t = \n", XT_t_matrix)
    
    # Compute w = (1)*(2)
    w_vector = np.matmul(XT_X_matrix_inv, XT_t_matrix)
    
    return w_vector

def compute_fM_output(M, w_vector, X_train):
    predictor_results = []
    
    # Check if M=0, if so the result is a constant
    if M == 0:
        predictor_results.append(w_vector[0])
    else:
        for x in X_train:
            result = 0
            # Depending on the value of M raise the training set value of x to the respective M power
            for m in range(M+1):
#                 print("w_vector[{0}] = {1}".format(m, w_vector[m]))
#                 print("x = {0}".format(x))
                if m == 0:
                    result = w_vector[m]
                else:
                    result += w_vector[m]*(x**m)
        
            predictor_results.append(result)
    return predictor_results


def compute_error(predictor_output, true_output):
    error = 0
    for i in range(10):
        if len(predictor_output) == 1:
            # If M=1, then the there is only one element in w, hence predictor_output[0]
            error += (predictor_output[0] - true_output[i])**2
        else:
            error += (predictor_output[i] - true_output[i])**2
    return error/10
    
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
    plt.plot(x, fM, color = "red", label = "fM")
    plt.plot(x, np.sin(4*np.pi*x), color = "green", label = "ftrue")
    plt.plot(X_train, t_train, "o", color = "blue", label = "training set")
    plt.plot(X_valid, t_valid, "o", color = "violet", label = "validation set")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.show()
    
if __name__ == "__main__":
    main()
