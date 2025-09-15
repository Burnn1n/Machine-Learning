
""" 
    Энд linear regression-г цэвэр кодоор хийх болно. Linear regression from scratch
"""
import matplotlib.pyplot as plt
import numpy as np


def linear_scratch(number):
    print('main')
    x = [1,2,3,4,5,10,40,100,150]   # input
    y = [2,4,6,8,10,20,80,200,300]  # output


    iteration = 5000 #Хэдэн удаа сурах эсэх
    learning_rate = 0.1
    w = [0,0]
    # Томъёо нь y = ax + b энд w = [a,b]

    # Scale x
    x_scaled = []
    x_len = len(x)
    x_mean = sum(x)/x_len
    x_deviation = (sum([((l-x_mean)**2) for l in x])/x_len)**0.5
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

        gradient = [g / n * learning_rate for g in gradient]

        
        w[0] -= gradient[0]
        w[1] -= gradient[1]

    number = (number - x_mean)/x_deviation
    return number * w[0] + w[1]



inp = 145.58
res = linear_scratch(inp)
print(f'Result: {inp} -> {res} {res/inp}')


""" 
    Энд linear regression-г numpy сан ашиглан хийх болно. 
    Linear regression using numpy
    Multiple Input
"""

def linear_numpy():

    data = [
        [2020, 100, 300000],
        [2020, 150, 350000],
        [2020, 200, 400000],
        [2021, 100, 320000],
        [2021, 150, 380000],
        [2021, 200, 430000],
        [2022, 100, 340000],
        [2022, 150, 400000],
        [2022, 200, 460000],
        [2023, 100, 360000],
        [2023, 150, 420000],
        [2023, 200, 480000],
        [2024, 100, 380000],
        [2024, 150, 440000],
        [2024, 200, 500000],
        [2024, 250, 550000],
        [2024, 300, 600000],
    ]

    # 2️⃣ Split data into inputs (X) and output (y)
    X = np.array([[d[0], d[1]] for d in data], dtype=float)  # [year, m3]
    y = np.array([d[2] for d in data], dtype=float)          # price

    print("Original data shape:")
    print(f"X (inputs): {X.shape} - {X[:3]}")  # Show first 3 rows
    print(f"y (output): {y.shape} - {y[:3]}")

    # 3️⃣ Normalize features (very important for multiple inputs!)
    X_mean = X.mean(axis=0)  # Mean for each column [year_mean, m3_mean]
    X_std = X.std(axis=0)    # Std for each column [year_std, m3_std]
    X_norm = (X - X_mean) / X_std

    print(f"\nNormalization stats:")
    print(f"X_mean: {X_mean}")
    print(f"X_std: {X_std}")
    print(f"X_norm first 3 rows: {X_norm[:3]}")

    # 4️⃣ Add bias term (column of 1s)
    X_bias = np.c_[np.ones(X_norm.shape[0]), X_norm]
    print(f"\nX_bias shape: {X_bias.shape}")
    print(f"X_bias first 3 rows:\n{X_bias[:3]}")

    # 5️⃣ Initialize weights: [bias, weight_for_year, weight_for_m3]
    weights = np.zeros(3)  # Now we have 3 weights!
    print(f"\nInitial weights: {weights}")

    # 6️⃣ Gradient descent parameters
    learning_rate = 0.01  # Smaller for multiple features
    iterations = 2000     # More iterations for convergence

    # Track learning progress
    cost_history = []

    # 7️⃣ Gradient descent loop

    print("\nStarting gradient descent...")
    for i in range(iterations):
        # Forward pass: make predictions
        # y = ax + bz + c
        predictions = X_bias.dot(weights)
        
        # Calculate errors
        errors = predictions - y
        
        # Calculate cost (mean squared error)
        cost = np.mean(errors**2)
        cost_history.append(cost)
        
        # Calculate gradients for each weight
        gradient = X_bias.T.dot(errors) / len(y)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Print progress every 200 iterations
        if i % 200 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}, Weights = {weights}")

    print(f"\nFinal weights: {weights}")

    print(X_bias, weights)
    predictions = X_bias.dot(weights)
    print(predictions)
    # 8️⃣ Prediction function
    def predict_price(year, size_m3):
        """Predict house price given year and size in m3"""
        # Normalize inputs using same stats as training
        year_norm = (year - X_mean[0]) / X_std[0]
        size_norm = (size_m3 - X_mean[1]) / X_std[1]
        
        # Create input vector [1, year_norm, size_norm]
        X_new = np.array([1, year_norm, size_norm])
        
        # Make prediction
        return X_new.dot(weights)
    
    # 9️⃣ Test predictions
    print("\n🔮 Testing predictions:")
    test_cases = [
        [2025, 150],  # Future prediction
        [2023, 175],  # Interpolation
        [2024, 120],  # Recent data
    ]

    for year, size in test_cases:
        predicted = predict_price(year, size)
        print(f"Year {year}, Size {size}m³ → Predicted price: ${predicted:,.0f}")

    # 🔟 Evaluate model performance
    final_predictions = X_bias.dot(weights)
    final_errors = final_predictions - y
    mse = np.mean(final_errors**2)
    rmse = np.sqrt(mse)

    print(f"\n📊 Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.0f}")
    print(f"Average error: ${np.mean(np.abs(final_errors)):,.0f}")

    # 1️⃣1️⃣ Visualizations

    # Plot 1: Learning curve (how error decreases over time)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(cost_history)
    plt.title('Learning Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Error)')
    plt.grid(True)

    # Plot 2: Actual vs Predicted
    plt.subplot(1, 3, 2)
    plt.scatter(y, final_predictions, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted')
    plt.grid(True)

    # Plot 3: 3D visualization of the prediction surface
    ax = plt.subplot(1, 3, 3, projection='3d')

    # Create a grid for the prediction surface
    years_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    sizes_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    Years_grid, Sizes_grid = np.meshgrid(years_range, sizes_range)

    # Calculate predictions for the entire surface
    predictions_surface = np.zeros_like(Years_grid)
    for i in range(Years_grid.shape[0]):
        for j in range(Years_grid.shape[1]):
            pred = predict_price(Years_grid[i,j], Sizes_grid[i,j])
            predictions_surface[i,j] = pred

    # Plot the surface and actual data points
    ax.plot_surface(Years_grid, Sizes_grid, predictions_surface, alpha=0.6, color='red')
    ax.scatter(X[:, 0], X[:, 1], y, color='blue', s=50)
    ax.set_xlabel('Year')
    ax.set_ylabel('Size (m³)')
    ax.set_zlabel('Price')
    ax.set_title('3D Prediction Surface')

    plt.tight_layout()
    plt.show()

    # 1️⃣2️⃣ Advanced analysis: Feature importance
    print(f"\n🔍 Feature Analysis:")
    print(f"Bias (baseline price): ${weights[0]:,.0f}")
    print(f"Year effect: ${weights[1]:,.0f} per normalized year")
    print(f"Size effect: ${weights[2]:,.0f} per normalized m³")

    # Convert back to original scale for interpretation
    year_effect_per_year = weights[1] / X_std[0]
    size_effect_per_m3 = weights[2] / X_std[1]

    print(f"\nIn real terms:")
    print(f"Each year adds: ${year_effect_per_year:,.0f} to house price")
    print(f"Each m³ adds: ${size_effect_per_m3:,.0f} to house price")

    # 1️⃣3️⃣ Model equation in human terms
    print(f"\n📋 Final Model Equation:")
    print(f"Price = {weights[0]:,.0f}")
    print(f"      + {year_effect_per_year:,.0f} × (Year - {X_mean[0]:.0f})")
    print(f"      + {size_effect_per_m3:,.0f} × (Size - {X_mean[1]:.0f})")

    print(f"\n🎯 Example: A 180m³ house in 2025:")
    example_price = predict_price(2025, 180)
    print(f"Predicted price: ${example_price:,.0f}")

    # 1️⃣4️⃣ Residual analysis (check if our linear model is good)
    residuals = final_errors
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(final_predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals (Error)')
    plt.title('Residual Plot')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=10, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n✅ Analysis complete! The model has learned to predict house prices based on both year and size.")

linear_numpy()