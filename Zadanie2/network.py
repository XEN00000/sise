from layer import Layer
import random


class Network:
    def __init__(self, layers_config, use_bias=True, learning_rate=0.1, momentum=0.0):
        self.layers = []
        for i in range(len(layers_config) - 1):
            self.layers.append(Layer(layers_config[i], layers_config[i + 1], use_bias))
        self.lr = learning_rate
        self.momentum = momentum
        self.prev_updates = [[[0] * layers_config[i] for _ in range(layers_config[i + 1])] for i in
                             range(len(self.layers))]

    def forward(self, input_vector):
        out = input_vector
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_epoch(self, training_data, shuffle=True):
        if shuffle:
            random.shuffle(training_data)
        global_error = 0
        for inputs, targets in training_data:
            outputs = self.forward(inputs)
            error = [t - o for t, o in zip(targets, outputs)]
            global_error += sum(e*e for e in error)
            next_error = error
            for idx in reversed(range(len(self.layers))):
                layer = self.layers[idx]
                prev = self.prev_updates[idx]
                next_error, updates = layer.backward(next_error, self.lr, self.momentum, prev)
                self.prev_updates[idx] = updates
        return global_error

    def train(self, training_data, max_epochs, target_errorm, log_every, log_file):
        with open(log_file, 'w') as f_log:
            for epoch in range(1, max_epochs + 1):
                err = self.train_epoch(training_data, shuffle=True)
                if epoch % log_every == 0:
                    f_log.write(f"{epoch},{err}\n")
                if err <= target_errorm:
                    break


    def test(self, test_data, output_file, record_details=False):
        with open(output_file, 'w') as f_out:
            for inputs, targets in test_data:
                outputs = self.forward(inputs)
                errors = [t - o for t, o in zip(targets, outputs)]
                total_error = sum(e*e for e in errors)
                line = f"{inputs},{total_error},{targets},{errors},{outputs}"
                if record_details:
                    pass
                f_out.write(line + "\n")