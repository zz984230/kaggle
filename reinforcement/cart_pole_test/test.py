import tflearn as tl
import numpy as np
import matplotlib.pyplot as plt


BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1


def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape([BATCH_SIZE, TIME_STEPS]) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :], xs]


def build_net():
    net = tl.input_data([None, TIME_STEPS, 1])
    net = tl.lstm(net, 40, activation='linear', return_seq=True)
    net = tl.lstm(net, 40, activation='linear', return_seq=True)
    net = tl.lstm(net, 40, activation='linear')
    net = tl.fully_connected(net, TIME_STEPS)
    net = tl.regression(net, loss='mean_square', learning_rate=0.001)
    return net

if __name__ == '__main__':
    seq, res, xs = get_batch()
    print(np.shape(seq), np.shape(res), np.shape(xs))
    net = build_net()

    model = tl.DNN(net, tensorboard_verbose=2)
    model.fit(seq, res, n_epoch=300, show_metric=True)

    plt.figure()
    for i in range(50):
        s = res[i]
        r = model.predict(np.reshape(seq[i], [1, TIME_STEPS, 1]))
        t = np.arange(i * TIME_STEPS, (i + 1) * TIME_STEPS, 1)

        plt.plot(t, s, color='r')
        plt.plot(t, r[0], color='b')
    plt.show()
