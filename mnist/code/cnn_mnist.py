import tflearn as tl
import tflearn.datasets.mnist as mn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MnistData(object):
    def __init__(self):
        self.__train_img = None
        self.__train_label = None
        self.__test_img = None

    def load_data(self):
        self.__load_train_data()
        self.__load_test_data()

    def __load_test_data(self):
        test_path = "../data/test.csv"
        test_df = pd.read_csv(test_path)
        self.__test_img = np.reshape(np.array(test_df), [-1, 28 * 28])

    def __load_train_data(self):
        train_path = "../data/train.csv"
        train_df = pd.read_csv(train_path)
        train_img = train_df.iloc[:, 1:]
        self.__train_label = np.reshape(np.array(pd.get_dummies(train_df["label"])), [-1, 10])
        self.__train_img = np.reshape(np.array(train_img), [-1, 28 * 28])

    @property
    def train_img(self):
        return self.__train_img

    @property
    def train_label(self):
        return self.__train_label

    @property
    def test_img(self):
        return self.__test_img

    def show_pic(self):
        num = np.random.randint(0, np.shape(self.__train_img)[0])
        img = np.reshape(self.__train_img[num], [28, 28])
        plt.figure()
        plt.imshow(img, cmap='Greys_r')
        plt.show()


class Net(object):
    def __init__(self,
                 x_data,
                 y_data,
                 epoch=10,
                 learning_rate=0.01):
        self.x_data = x_data
        self.y_data = y_data
        self.epoch = epoch
        self.lr = learning_rate
        self.model = None
        self.sess = None
        self.net = None
        tl.init_graph(gpu_memory_fraction=0.6)

    def build_net(self):
        self.net = tl.input_data(shape=[None, 28, 28, 1])
        self.net = tl.conv_2d(incoming=self.net, nb_filter=32, filter_size=3, activation='relu')
        self.net = tl.max_pool_2d(incoming=self.net, kernel_size=3, strides=2)
        self.net = tl.conv_2d(incoming=self.net, nb_filter=64, filter_size=3, activation='relu')
        self.net = tl.max_pool_2d(incoming=self.net, kernel_size=3, strides=2)
        # self.net = tl.conv_2d(incoming=self.net, nb_filter=256, filter_size=3, activation='relu')
        # self.net = tl.max_pool_2d(incoming=self.net, kernel_size=3, strides=2)
        self.net = tl.fully_connected(incoming=self.net, n_units=7 * 7 * 64, activation='relu')
        self.net = tl.fully_connected(incoming=self.net, n_units=10, activation='softmax')
        self.net = tl.regression(incoming=self.net)

        self.model = tl.DNN(network=self.net, tensorboard_verbose=2, tensorboard_dir="D:/tmp/tflearn_logs/mnist")

    def train_model(self):
        self.model.fit(np.reshape(self.x_data, [-1, 28, 28, 1]), self.y_data, n_epoch=self.epoch, show_metric=True, batch_size=128)
        self.model.save("../model/cnn/cnn_model")

    def set_test_set(self, v_img):
        self.v_img = v_img

    def test_model(self):
        self.model.load("../model/cnn/cnn_model")
        submission_df = pd.DataFrame()
        batch_size = 5000
        n = 0
        l = np.shape(self.v_img)[0]
        pre = []
        while n + batch_size <= l:
            pre.append([np.argmax(i) for i in self.model.predict(X=np.reshape(self.v_img[n: n + batch_size], [-1, 28, 28, 1]))])
            n += batch_size
        pre = np.reshape(pre, [-1, 1])
        pre = np.concatenate((pre, np.reshape([np.argmax(i) for i in self.model.predict(X=np.reshape(self.v_img[n:], [-1, 28, 28, 1]))], [-1, 1])))

        submission_df["ImageId"] = [i for i in range(len(pre) + 1) if i != 0]
        submission_df["Label"] = pre
        submission_df.to_csv("../data/sample_submission.csv", index=False)

    def validate_model(self, val_img, val_label):
        self.model.load("../model/cnn/cnn_model")
        batch_size = 5000
        n = 0
        l = np.shape(val_img)[0]
        pre = []
        while n + batch_size <= l:
            pre.append([np.argmax(i) for i in
                        self.model.predict(X=np.reshape(val_img[n: n + batch_size], [-1, 28, 28, 1]))])
            n += batch_size
        pre = np.reshape(pre, [-1, 1])
        pre = np.concatenate((pre, np.reshape(
            [np.argmax(i) for i in self.model.predict(X=np.reshape(val_img[n:], [-1, 28, 28, 1]))], [-1, 1])))

        pre = np.ravel(pre)
        label = np.ravel([np.argmax(i) for i in val_label])
        diff_index = np.ravel(np.argwhere(pre != label))
        self.__draw(pre, val_img, diff_index)

    def __draw(self, pre, val_img, diff_index):
        for i in diff_index:
            plt.figure()
            plt.title(pre[i])
            plt.imshow(np.reshape(val_img[i], [28, 28]), cmap='Greys_r')
            plt.savefig("../check/%s" % i)
            plt.close()

if __name__ == "__main__":
    data = MnistData()
    data.load_data()
    train_img = data.train_img / 255
    test_img = data.test_img / 255
    model = Net(x_data=train_img, y_data=data.train_label, epoch=5, learning_rate=0.001)
    model.set_test_set(test_img)
    model.build_net()
    # model.train_model()
    # model.test_model()
    model.validate_model(train_img, data.train_label)
