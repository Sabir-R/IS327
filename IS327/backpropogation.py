import math

# Given values
x1 = 0.1
x2 = 0.37
y_desired = 0.76

w11 = 0.94
w12 = 0.84
w21 = 0.83
w22 = 0.76

b1 = 0.89
b2 = 0.79
b3 = 0.95

w1 = 0.68
w2 = 0.04

# Learning rate
alpha = 0.8

# Forward pass
input_h1 = w11 * x1 + w21 * x2 + b1
output_h1 = 1 / (1 + math.exp(-input_h1))

input_h2 = w12 * x1 + w22 * x2 + b2
output_h2 = 1 / (1 + math.exp(-input_h2))

input_o = w1 * output_h1 + w2 * output_h2 + b3
output_o = input_o  # For regression, no activation function at the output layer

# Calculate Mean Squared Error (MSE) loss
mse_loss = 0.5 * (output_o - y_desired) ** 2

# Backward pass

# Gradients for the output layer
d_output_o = output_o - y_desired
d_w1 = d_output_o * output_h1
d_w2 = d_output_o * output_h2
d_b3 = d_output_o

# Gradients for the hidden layer
d_output_h1 = d_output_o * w1
d_output_h2 = d_output_o * w2

d_input_h1 = d_output_h1 * output_h1 * (1 - output_h1)
d_input_h2 = d_output_h2 * output_h2 * (1 - output_h2)

d_w11 = d_input_h1 * x1
d_w21 = d_input_h1 * x2
d_b1 = d_input_h1

d_w12 = d_input_h2 * x1
d_w22 = d_input_h2 * x2
d_b2 = d_input_h2

# Update weights and biases
w11 -= alpha * d_w11
w21 -= alpha * d_w21
b1 -= alpha * d_b1

w12 -= alpha * d_w12
w22 -= alpha * d_w22
b2 -= alpha * d_b2

w1 -= alpha * d_w1
w2 -= alpha * d_w2
b3 -= alpha * d_b3

# Results
#print(f"Output of hidden layer neuron 1: {output_h1}")
#print("")
#print(f"Output of hidden layer neuron 2: {output_h2}")
#print("")
#print(f"Output of output layer: {output_o}")
#print("")
#print(f"MSE Loss: {mse_loss}")
#print("")
#print(f"Gradient of the loss with respect to the output neuron: {d_output_o}")
#print("")
#print(f"Gradients of the loss with respect to the weights from hidden layer to output layer: d_w1 = {d_w1}, d_w2 = {d_w2}, d_b3 = {d_b3}")
#print("")
#print(f"Gradients of the loss with respect to the weights from input layer to hidden layer: d_w11 = {d_w11}, d_w12 = {d_w12}, d_w21 = {d_w21}, d_w22 = {d_w22}, d_b1 = {d_b1}, d_b2 = {d_b2}")
#print("")
#print(f"Updated weights from input layer to hidden layer: w11 = {w11}, w12 = {w12}, w21 = {w21}, w22 = {w22}, b1 = {b1}, b2 = {b2}")

val1 = 27.811854
val2 = 2.511322
val3 = 1.453073
val4 = 1.026652
val5 = 0.769540
val6 = 0.495303
val7 = 0.337994
val8 = 0.112003
val9 = 0.051945
val10 = 0.017786

four_sum = val1 + val2 + val3 + val4
all_sum = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9 + val10
pca = four_sum/all_sum
print(pca)
