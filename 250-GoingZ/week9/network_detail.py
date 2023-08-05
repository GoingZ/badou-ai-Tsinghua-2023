import numpy
import scipy.special

class network_detail:
    def __init__(self, inputnodes, hiddenodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddenodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid function

    # 根据输入数据计算并输出答案
    def query(self, inputs):
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)
        return final_outputs
    
    def train(self, input_list, output_list):

        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(output_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1 - final_outputs)),
                                numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                       numpy.transpose(inputs))

if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.1
    n = network_detail(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 100
    
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 归一化
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    test_data_file = open("dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print(label, "network's answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)