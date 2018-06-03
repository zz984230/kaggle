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
    """
        Attributes:
            x_data:          x坐标特征数据
            y_data:          y坐标目标数据
            epoch:           训练次数
            lr:              学习率
            layer_num:       神经层数
            activation_type: 激活函数类型  0: linear  线性激活函数
                                         1: sigmoid S型激活函数
                                         2: tanh    双曲正切激活函数
                                         3: relu    修正线性单元激活函数
            cell_num:        神经元个数
            optimizer:       优化器       0: SGD     随机梯度下降优化器
                                         1: Adam    Adam优化器
            opt:             优化器
        """

    def __init__(self,
                 x_data,
                 y_data,
                 epoch=10,
                 learning_rate=0.01,
                 layer_num=3,
                 cell_num=3,
                 activation_type=2,
                 optimizer=1):
        """
        类初始化
        :param x_data:        x坐标特征数据
        :param y_data:        y坐标目标数据
        :param epoch:         训练次数
        :param learning_rate: 学习率
        """
        self.x_data = x_data
        self.y_data = y_data
        self.epoch = epoch
        self.lr = learning_rate
        self.layer_num = layer_num
        self.activation_type = activation_type
        self.cell_num = cell_num
        self.optimizer = optimizer
        self.model = None
        self.sess = None
        self.opt = {0: 'sgd',
                    1: 'adam'}
        self.func = {0: 'linear',
                     1: 'sigmoid',
                     2: 'tanh',
                     3: 'relu'}
        self.net = None
        tl.init_graph(gpu_memory_fraction=0.6)

    def build_net(self):
        self.net = tl.input_data(shape=[None, np.shape(self.x_data)[1]])
        for i in range(self.layer_num):
            activation = self.func[self.activation_type]
            if i == self.layer_num - 1:
                self.cell_num = 10
                activation = 'softmax'
            self.net = tl.fully_connected(incoming=self.net,
                                          n_units=self.cell_num,
                                          activation=activation)

        self.net = tl.regression(incoming=self.net,
                                 loss="categorical_crossentropy",
                                 optimizer=self.opt[self.optimizer],
                                 learning_rate=self.lr,)

        self.model = tl.DNN(network=self.net, tensorboard_verbose=2, tensorboard_dir="D:/tmp/tflearn_logs/mnist")

    def train_model(self):
        self.model.fit(self.x_data, self.y_data, n_epoch=self.epoch, show_metric=True)
        self.model.save("../model/fully/fully_model")

    def set_validate_set(self, v_img):
        self.v_img = v_img

    def test_model(self):
        self.model.load("../model/fully/fully_model")
        submission_df = pd.DataFrame()
        pre = [np.argmax(i) for i in self.model.predict(X=self.v_img)]
        submission_df["ImageId"] = [i for i in range(len(pre) + 1) if i != 0]
        submission_df["Label"] = pre
        submission_df.to_csv("../data/sample_submission.csv", index=False)
        # self.__draw(pre)

    # def __draw(self, pre):
    #     fig = plt.figure(figsize=(9, 6))
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.xlim((-2, 2))
    #     plt.ylim((0, 2))
    #     ax.scatter(self.x_data, self.y_data, s=5)
    #     ax.plot(self.x_data, pre, color='gold', linewidth=3)
    #     plt.show()

if __name__ == "__main__":
    data = MnistData()
    data.load_data()
    model = Net(x_data=data.train_img, y_data=data.train_label, epoch=5, learning_rate=0.001, layer_num=5, cell_num=64, activation_type=3)
    model.set_validate_set(data.test_img)
    print(np.shape(data.test_img))
    model.build_net()
    # model.train_model()
    model.test_model()
