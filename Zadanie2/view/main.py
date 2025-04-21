from Zadanie2.logic.layer import Layer

if __name__ == "__main__":
    # test of first neuron
    #neuron = Neuron(3, "Sigmoid", False)

    X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

    layer1 = Layer(4, 5)
    layer2 = Layer(5, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)
