from operator import lt
import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from SoftmaxRegression import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 
    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    plt.figure()

    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Digit 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Digit 2')

    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.title('Two Training Features for Digits 1 and 2')
    plt.legend()

    plt.savefig('train_features.png')
    plt.show()

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''

    plt.figure()

    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Digit 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Digit 2')

    x_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    y_values = -(W[0] + W[1] * x_values) / W[2]

    plt.plot(x_values, y_values, label='Decision Boundary')

    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()

    plt.ylim(np.min(X[:, 1]) - 0.05, np.max(X[:, 1]) + 0.05)
    plt.savefig('train_result_sigmoid.png')
    plt.show()

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    plt.figure()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Digit 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Digit 1')
    plt.scatter(X[y == 2, 0], X[y == 2, 1], label='Digit 2')

    x_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

    for i in range(3):
        for j in range(i + 1, 3):
            y_values = -((W[0, i] - W[0, j]) +
                         (W[1, i] - W[1, j]) * x_values) / \
                       (W[2, i] - W[2, j])

            plt.plot(x_values, y_values,
                     label='Boundary ' + str(i) + '-' + str(j))

    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.title('Softmax Logistic Regression Decision Boundaries')
    plt.legend()
    plt.savefig('train_result_softmax.png')
    plt.show()

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y == 2)] = -1
    valid_y[np.where(valid_y == 2)] = -1

    data_shape = train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    learning_rates = [0.05, 0.25, 0.75]
    max_iters = [50, 250, 750]

    best_accuracy = 0

    for learning_rate in learning_rates:
        for max_iter in max_iters:
            model = logistic_regression(learning_rate=learning_rate, max_iter=max_iter)
            model.fit_miniBGD(train_X, train_y, 10)

            accuracy = model.score(valid_X, valid_y)

            print(learning_rate, max_iter, accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_logisticR = model

    print("Best validation accuracy:", best_accuracy)
    print("Best weights:", best_logisticR.get_params())

	# Visualize the your 'best' model after training.
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())


    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    raw_data, labels = load_data(os.path.join(data_dir, test_filename))

    test_all_X = prepare_X(raw_data)
    test_all_Y, test_index = prepare_y(labels)

    test_X = test_all_X[test_index]
    test_y = test_all_Y[test_index]

    test_y[np.where(test_y == 2)] = -1

    print("Test accuracy:", best_logisticR.score(test_X, test_y))
    

    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    learning_rates = [0.05, 0.25, 0.75]
    max_iters = [50, 250, 750]

    best_accuracy = 0

    for learning_rate in learning_rates:
        for max_iter in max_iters:
            model = logistic_regression_multiclass(
                learning_rate=learning_rate,
                max_iter=max_iter,
                k=3
            )

            model.fit_miniBGD(train_X, train_y, 10)

            accuracy = model.score(valid_X, valid_y)

            print(learning_rate, max_iter, accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_logistic_multi_R = model

    print("Best validation accuracy:", best_accuracy)
    print("Best weights:", best_logistic_multi_R.get_params())

	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())

    # Use the 'best' model above to do testing.
    raw_data, labels = load_data(os.path.join(data_dir, test_filename))

    test_X = prepare_X(raw_data)
    test_y, _ = prepare_y(labels)

    print("Multiclass test accuracy:", best_logistic_multi_R.score(test_X, test_y))

    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    softmax_classifier = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000,k=2)

    softmax_classifier.fit_miniBGD(train_X, train_y, 10)

    print("Softmax training accuracy:", softmax_classifier.score(train_X, train_y))

    print("Softmax validation accuracy:", softmax_classifier.score(valid_X, valid_y))

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    ##### set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y == 2)] = -1
    valid_y[np.where(valid_y == 2)] = -1

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    sigmoid_classifier = logistic_regression(learning_rate=0.5, max_iter=10000)

    sigmoid_classifier.fit_miniBGD(train_X, train_y, 10)

    print("Sigmoid training accuracy:", sigmoid_classifier.score(train_X, train_y))

    print("Sigmoid validation accuracy:", sigmoid_classifier.score(valid_X, valid_y))


    ################Compare and report the observations/prediction accuracy


    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
