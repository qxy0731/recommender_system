import pandas as pd
import numpy as np
import random
import math

def devide_train_test(matrix,fraction = 0.1,seed = 11):
    test_data = {}  # {(row,col) : true_value}
    training_data = {}
    random.seed(seed)
    for row in range(1,matrix.shape[0]+1):
        for col in range(1,matrix.shape[1]+1):
            true_col = col -1
            if matrix[matrix.columns[true_col]][row]:
                if random.random() <= fraction:
                    test_data[(row,col)] = matrix[matrix.columns[true_col]][row]
                    matrix[matrix.columns[true_col]][row] = 0
    return test_data,matrix

def evaluate_rmse(test_data,estimated_data):
    sum = 0
    for key in test_data:
        sum += (test_data[key] - estimated_data[key])**2
    return math.sqrt(sum / len(test_data))
