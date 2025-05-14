import dataLoader as dl
import mlp
import test

train_dataset, test_dataset = dl.loader()

net = mlp.MLP(input_number=4,
              output_number=3,
              hidden_layers=[5, 4],
              use_bias=True,
              learning_rate=0.1,
              momentum=0.9,
              max_epochs=500,
              target_error=0.001,
              log_rate=50,
              log_path='log.txt',
              dataset=train_dataset,
              save_path='irisNet.ntwrk')

net.train()

test.test_the_network(test_dataset, 'irisNet.ntwrk')
