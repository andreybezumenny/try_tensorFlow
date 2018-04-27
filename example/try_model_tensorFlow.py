#! https://tproger.ru/translations/text-classification-tensorflow-neural-networks/
# https://tproger.ru/translations/learning-neural-networks/
# https://github.com/dmesquita/understanding_tensorflow_nn/blob/master/README.md

import tensorflow as tf
import numpy as np
import try_model
from collections import Counter


train_data = try_model.X_train
print('Elem in train collection: ', len(train_data))
train_intent = try_model.y_train
print('Intents : ', len(train_intent))
test_data = try_model.X_test
print('Elem in test collection: ', len(test_data))
test_intent = try_model.y_test
print('Intents : ', len(test_intent))


vocab = Counter()

for text in train_data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in test_data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)

print("Total words:", total_words)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i
    return word2index


word2index = get_word_2_index(vocab)

print('Index of the word \'the\'', word2index['the'])


def get_batch(data, intent, i, batch_size):
    batches = []
    results = []
    texts = data # [i * batch_size:i * batch_size + batch_size]  # a[start:end] items start through end-1
    categories = intent # [i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((27), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        elif category == 2:
            y[2] = 1.
        elif category == 3:
            y[3] = 1.
        elif category == 4:
            y[4] = 1.
        elif category == 5:
            y[5] = 1.
        elif category == 6:
            y[6] = 1.
        elif category == 7:
            y[7] = 1.
        elif category == 8:
            y[8] = 1.
        elif category == 9:
            y[9] = 1.
        elif category == 10:
            y[10] = 1.
        elif category == 11:
            y[11] = 1.
        elif category == 12:
            y[12] = 1.
        elif category == 13:
            y[13] = 1.
        elif category == 14:
            y[14] = 1.
        elif category == 15:
            y[15] = 1.
        elif category == 16:
            y[16] = 1.
        elif category == 17:
            y[17] = 1.
        elif category == 18:
            y[18] = 1.
        elif category == 19:
            y[19] = 1.
        elif category == 20:
            y[20] = 1.
        elif category == 21:
            y[21] = 1.
        elif category == 22:
            y[22] = 1.
        elif category == 23:
            y[23] = 1.
        elif category == 24:
            y[24] = 1.
        elif category == 25:
            y[25] = 1.
        else:
            y[26] = 1.
        results.append(y)
        # print('------------------', results)

    return np.array(batches), np.array(results)


print("Each batch has n_texts and each matrix has n_elements (words):", get_batch(train_data,train_intent,1,100)[0].shape)

print("Each batch has n_labels and each matrix has 27 elements (27 categories):", get_batch(train_data,train_intent ,1,100)[1].shape)


# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 50      # 1st layer number of features
n_hidden_2 = 50       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 27         # Categories: graphics, sci.space and baseball

input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output")
global_step = tf.Variable(0, trainable=False)


def multilayer_perceptron(input_tensor, weights, biases): # peceptron Rumelhard's
    # layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    # layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    # layer_1 = tf.nn.relu(layer_1_addition)
    #
    # # Hidden layer with RELU activation
    # layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    # layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    # layer_2 = tf.nn.relu(layer_2_addition)
    #
    # # Output layer
    # out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    # out_layer_addition = out_layer_multiplication + biases['out']

    layer_1 = tf.layers.dense(input_tensor, n_hidden_1, tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, tf.nn.relu)
    out_layer = tf.layers.dense(layer_2, n_classes)

    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss and optimizer
loss = tf.losses.softmax_cross_entropy(output_tensor, prediction)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(train_data, train_intent, i, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            step, c, _ = sess.run([global_step, loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor: batch_y})
            print(f"Step: {step} -> Loss: {c}")
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "loss=",
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(test_intent)
    batch_x_test, batch_y_test = get_batch(test_data, test_intent, 0, total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))



