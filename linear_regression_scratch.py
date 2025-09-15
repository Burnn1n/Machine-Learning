
""" 
    Энд linear regression-г цэвэр кодоор хийх болно. 
"""
import matplotlib.pyplot as plt
import numpy as np


def main(number):
    print('main')
    x = [1,2,3,4,5,10,40,100]   # input
    y = [2,4,6,8,10,20,80,200]  # output
    iteration = 500000 #Хэдэн удаа сурах эсэх

    learning_rate = 0.001
    w = [0,0]
    # Томъёо нь y = ax + b энд w = [a,b]



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
            print(f'{i} {err} ')
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
    print(w)
    print(number * w[0] + w[1])

    # plt.figure(figsize=(10, 6))
    
    # # Plot original data points
    # plt.scatter(x, y, color='red', s=50, label='Data points', zorder=3)
    
    # # Create line for plotting (extend range for better visualization)
    # x_line = np.linspace(0, 110, 100)
    # y_line = w[0] * x_line + w[1]
    
    # # Plot the fitted line
    # plt.plot(x_line, y_line, color='blue', linewidth=2, 
    #          label=f'Fitted line: y = {w[0]:.3f}x + {w[1]:.3f}')
    
    # # Plot the prediction point
    # pred_y = number * w[0] + w[1]
    # plt.scatter([number], [pred_y], color='green', s=100, 
    #             label=f'Prediction: ({number}, {pred_y:.1f})', zorder=3)
    
    # # Customize the plot
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Linear Regression: Fitted Line and Data Points')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # # Set axis limits for better view
    # plt.xlim(-5, 270)
    # plt.ylim(-10, 550)
    
    # plt.tight_layout()
    # plt.show()




	
# main(145.58)

# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Your data
# x = np.array([1, 2, 3, 4, 5, 10, 40, 100,]).reshape(-1, 1)  # feature must be 2D
# y = np.array([2, 4, 6, 8, 10, 20, 80, 200, ])                  # target

# # Create model
# model = LinearRegression()

# # Train model
# model.fit(x, y)

# # Learned weights
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"Learned line: y = {slope:.6f} * x + {intercept:.6f}")

# # Predict new value
# number = 145.58
# predicted = model.predict([[number]])[0]
# print(f"Prediction for x={number}: {predicted:.2f}")



# arr = [86,87,88,86,87,85,86]
# def _get_deviation(arr):
#     n = len(arr)
#     u = sum(arr) / n
#     a = 0
#     for i in range(n):
#         a += (arr[i] - u)**2
#     a = (a/n)**0.5
#     print(a)
#     x = np.std(arr)
#     print(x)
#     print(arr)

# _get_deviation(arr)

import numpy
import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
z = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# x = numpy.random.uniform(0.0, 5.0, 250)
# y = numpy.random.uniform(15, 500, 250)

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

# 2️⃣ Normalize feature
X_mean = x.mean()
X_std = x.std()
X_norm = (x - X_mean) / X_std

# Add bias term
X_bias = np.c_[np.ones(X_norm.shape[0]), X_norm]

# 3️⃣ Initialize weights
weights = np.zeros(2)

# 4️⃣ Gradient descent parameters
learning_rate = 0.1
iterations = 1000

# 5️⃣ Gradient descent loop
for i in range(iterations):
    predictions = X_bias.dot(weights)
    errors = predictions - y
    gradient = X_bias.T.dot(errors) / len(y)
    weights -= learning_rate * gradient

print(f'weights {weights}')

# 6️⃣ Prediction function
def predict(date_str):
    # date_num = (datetime.datetime.strptime(date_str, "%Y-%m-%d") - base_date).days
    date_num = date_str
    print(date_num)
    date_norm = (date_num - X_mean) / X_std
    X_new = np.array([1, date_norm])
    print(f'{X_new} {weights}')
    return X_new.dot(weights)

def myfunc(num):
    date_norm = (num - X_mean) / X_std
    X_new = np.array([1, date_norm])
    return X_new.dot(weights)

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)

plt.show()