
input_dim = 123
middle_layer_size = 20
hidden_layers_num = 100

# go from 20 to 123 in 5 steps (or from 123 to 20)

delta_dim = int((input_dim - middle_layer_size) / hidden_layers_num)


print("Manual The number of neurons is ", input_dim)

for i in range(1, hidden_layers_num):
    neurons_num =  input_dim - i*delta_dim
    print("The number of neurons is ", neurons_num)

for i in range(hidden_layers_num):
    neurons_num =  middle_layer_size + i*delta_dim
    print("The number of neurons is ", neurons_num)

print("Manual The number of neurons is ", input_dim)