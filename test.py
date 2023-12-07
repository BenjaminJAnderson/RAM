import matplotlib.pyplot as plt

# Replace these lists with your actual X & Y coordinates
x_coordinates = [1, 2, 2, 1]  # Replace with your X coordinates
y_coordinates = [2, 2, 3, 3]  # Replace with your Y coordinates

# Plot the points
plt.figure(figsize=(8, 6))
plt.plot(x_coordinates, y_coordinates, marker='o', linestyle='-')
plt.title('Outline of a Foot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()