
""" 
    Энд linear regression-г цэвэр кодоор хийх болно. 
"""
import matplotlib.pyplot as plt
import numpy as np


def main(number):
    print('main')
    x = [1,2,3,4,5,10,40,100,120,100]   # input
    y = [2,4,6,8,10,20,80,200,130,10]  # output


    iteration = 5000 #Хэдэн удаа сурах эсэх
    learning_rate = 0.1
    w = [0,0]
    # Томъёо нь y = ax + b энд w = [a,b]

    # Scale x
    x_scaled = []
    x_len = len(x)
    x_mean = sum(x)/x_len
    x_deviation = (sum([((l-x_mean)**2)/x_len for l in x]))**0.5
    # scaled_value = (original_value - mean) / standard_deviation
    for l in x:
        val = (l - x_mean)/x_deviation
        x_scaled += [val]

    x = x_scaled

    for i in range(iteration):
        # Алгоримтын хариуг олох
        predictions = []
        for l in x:
            result = w[0] * l + w[1]
            predictions += [result]

        error = [] 
        gradient = [0,0]
        n = len(y)
        for j in range(len(y)):
            # Алдаа нь зөв хариулт болон таамаглалын ялгавар байна
            err = predictions[j] - y[j]
            # print(f'{i} {err} ')
            error += [err]
            # Алдааг нь 
            gradient[0] += err * x[j]
            gradient[1] += err

        gradient[0] /= n
        gradient[1] /= n

        gradient[0] *= learning_rate
        gradient[1] *= learning_rate
        
        w[0] -= gradient[0]
        w[1] -= gradient[1]

    number = (number - x_mean)/x_deviation
    return number * w[0] + w[1]



inp = 145.58
res = main(inp)
print(f'Result: {inp} -> {res} {res/inp}')