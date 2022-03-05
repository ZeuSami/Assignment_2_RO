import numpy as np
import pandas as pd
import mlrose_hiive
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import *
import matplotlib.pyplot as plt


#setup the randoms tate
RANDOM_STATE = 19920604


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
	return accuracy_score(Y_true, Y_pred)

#input: Name of classifier, predicted labels, actual labels
#output: print ACC, AUC, Prec, Recall and F1-Score of the Classifier
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print("______________________________________________")
	print("")

def main():
    #load training and testing data
    X = pd.read_csv("bldg_x.csv")
    Y = pd.read_csv("bldg_y.csv")
    X_train = X[:70]
    X_test = X[71:]
    Y_train = Y[:70]
    Y_test = Y[71:]
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    #change and select params here
    param_kfold = 10 #parameter k for kfold CV
    """
    #NN with SA
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))   
    
    
    title = "Learning Curves, SA, Geom"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.GeomDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, Arith"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, Exp"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ExpDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    plt.savefig('NN_SA_decay.png')

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))   
    
    
    title = "Learning Curves, SA, 100 iters"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 200 iters"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 200, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 500 iters"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 500, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 1000 iters"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 1000, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,3], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    plt.savefig('NN_SA_iters.png')

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))   
    
    
    title = "Learning Curves, SA, 10 attempts"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 10, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 20 attempts"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 50 attempts"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 50, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 100 attempts"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 100, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,3], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    plt.savefig('NN_SA_attempts.png')
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))   
    
    
    title = "Learning Curves, SA, 2 nodes"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 4 nodes"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 8 nodes"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [8], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, SA, 16 nodes"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [16], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,3], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    plt.savefig('NN_SA_nodes.png')

    #NN with GA
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))   
    
    
    title = "Learning Curves, GA, mutation prob= 0.1"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, GA, mutation prob= 0.2"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.2,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, GA, mutation prob= 0.5"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.5,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    plt.savefig('NN_GA_mutation prob.png')

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))       
    
    title = "Learning Curves, GA, pop = 100"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 100,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, GA, pop = 200"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 200,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, GA, pop = 500"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 500,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    plt.savefig('NN_GA_pop size.png')
    

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))       
    
    title = "Learning Curves, GA, nodes = 2"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 200,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, GA, nodes = 4"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 200,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, GA, nodes = 8"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [8], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 200,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    plt.savefig('NN_GA_nodes.png')
    

    #NN with RHC
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))       
    
    title = "Learning Curves, RHC, restart = 0"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 0,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, RHC, restart = 2"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 2,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, RHC, restart = 4"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 4,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    plt.savefig('NN_RHC_restarts.png')

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))       
    
    title = "Learning Curves, RHC, nodes = 2"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 0,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, RHC, nodes = 4"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 0,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, RHC, nodes = 8"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [8], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 0,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    plt.savefig('NN_RHC_nodes.png')
    """

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    title = "Learning Curves, Best SA"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='simulated_annealing', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, schedule = mlrose_hiive.ArithDecay(),
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, Best GA"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='genetic_alg', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, mutation_prob = 0.1, pop_size = 200,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    title = "Learning Curves, Best RHC"
    cv = KFold(n_splits=param_kfold)   
    estimator = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation ='relu', 
		                         	algorithm ='random_hill_climb', 
		                         	max_iters = 100, bias = True, is_classifier = True, 
		                         	learning_rate = 0.001, early_stopping = True, restarts = 0,
		                         	max_attempts = 20, random_state = RANDOM_STATE, curve = True)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )

    plt.savefig('NN_Comparison.png')    
	

if __name__ == "__main__":
	main()
	
