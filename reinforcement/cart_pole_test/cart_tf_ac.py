import tensorflow as tf
import gym
import numpy as np


class ActorNet(object):
    def __init__(self, learning_rate=0.01, obs_num=4, act_num=2):
        self.lr = learning_rate
        self.obs_num = obs_num
        self.act_num = act_num
        self.obs_input = None
        self.action_input = None
        self.td_err = None

        self.sess = tf.Session()

    def build_net(self):
        self.obs_input = tf.placeholder(tf.float32, [1, self.obs_num])
        self.action_input = tf.placeholder(tf.int32, [1, ])
        self.td_err = tf.placeholder(tf.float32, [1, 1])

        net = tf.layers.dense(inputs=self.obs_input,
                              units=10,
                              activation=tf.nn.relu)
        net = tf.layers.dense(inputs=net,
                              units=self.act_num)

        self.result = tf.nn.softmax(net)

        neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=self.action_input)
        loss = tf.reduce_mean(neg_loss * self.td_err)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.sess.run(tf.global_variables_initializer())

    def learn(self, train_features, train_actions, td_err):
        self.sess.run(self.opt, feed_dict={self.obs_input: train_features,
                                           self.action_input: train_actions,
                                           self.td_err: td_err})

    def run(self, cur_obs):
        return self.sess.run(self.result, feed_dict={self.obs_input: cur_obs})


class CriticNet(object):
    def __init__(self, learning_rate=0.003, obs_num=4, decay=0.9):
        self.lr = learning_rate
        self.obs_num = obs_num
        self.decay = decay
        self.obs_input = None
        self.r = None
        self.v_ = None
        self.td_err = None

        self.sess = tf.Session()

    def build_net(self):
        self.obs_input = tf.placeholder(tf.float32, [1, self.obs_num])
        self.r = tf.placeholder(tf.float32, [1, 1])
        self.v_ = tf.placeholder(tf.float32, [1, 1])

        net = tf.layers.dense(inputs=self.obs_input,
                              units=10,
                              activation=tf.nn.relu)
        self.v = tf.layers.dense(inputs=net,
                                 units=1)

        self.td_err = self.r + self.decay * self.v_ - self.v
        loss = tf.square(self.td_err)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.sess.run(tf.global_variables_initializer())

    def learn(self, state, state_, r):
        v_ = self.sess.run(self.v, feed_dict={self.obs_input: state_})
        td_err, _ = self.sess.run([self.td_err, self.opt], feed_dict={self.obs_input: state,
                                                                      self.r: r,
                                                                      self.v_: v_})
        return td_err


def action_choice(acts):
    return np.random.choice(range(len(acts[0])), p=acts[0])

if __name__ == '__main__':
    obs_num = 4
    act_num = 2

    an = ActorNet(obs_num=obs_num, act_num=act_num)
    an.build_net()

    cn = CriticNet(obs_num=obs_num)
    cn.build_net()

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    flag = False
    for i in range(10000):
        obs = env.reset()
        count = 0

        while True:
            count += 1
            acts = an.run(np.reshape(obs, [1, 4]))
            action = action_choice(acts)
            obs_, reward, done, _ = env.step(action)
            if done:
                reward = -20
            td_err = cn.learn(np.reshape(obs, [1, 4]), np.reshape(obs_, [1, 4]), np.reshape(reward, [1, 1]))
            an.learn(np.reshape(obs, [1, 4]), [action], td_err)

            if done:
                break

            if flag:
                env.render()

            obs = obs_

        if count >= 1000:
            flag = True

        print("cur round is: %s, reward is: %s" % (i, count))
