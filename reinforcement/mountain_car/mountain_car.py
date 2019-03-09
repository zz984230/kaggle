import tensorflow as tf
import gym
import numpy as np


class Policy(object):
    def __init__(self, learning_rate=0.01, obs_num=2, act_num=3):
        self.lr = learning_rate
        self.obs_num = obs_num
        self.act_num = act_num
        self.obs_input = None
        self.action_input = None
        self.reward_input = None

        self.sess = tf.Session()

    def build_net(self):
        self.obs_input = tf.placeholder(tf.float32, [None, self.obs_num])
        self.action_input = tf.placeholder(tf.int32, [None, ])
        self.reward_input = tf.placeholder(tf.float32, [None, ])

        net = tf.layers.dense(inputs=self.obs_input,
                              units=10,
                              activation='tanh')
        net = tf.layers.dense(inputs=net,
                              units=self.act_num)

        self.result = tf.nn.softmax(net)

        neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=self.action_input)
        loss = tf.reduce_mean(neg_loss * self.reward_input)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.sess.run(tf.global_variables_initializer())

    def learn(self, train_features, train_actions, rewards):
        print(train_actions)
        self.sess.run(self.opt, feed_dict={self.obs_input: train_features,
                                           self.action_input: train_actions,
                                           self.reward_input: rewards})

    def run(self, cur_obs):
        return self.sess.run(self.result, feed_dict={self.obs_input: cur_obs})

obs_num = 2
act_num = 3
features = []
actions = []
rewards = []

p = Policy(obs_num=obs_num, act_num=act_num)
p.build_net()


def action_choice(acts):
    return np.random.choice(range(len(acts[0])), p=acts[0])


def data_accumulation(obs, reward, action):
    features.append(obs)
    rewards.append(reward)
    actions.append(action)


def reward_decay(rewards, decay=0.995):
    c = np.zeros_like(rewards)
    s = 0
    for i in reversed(range(len(rewards))):
        s = s * decay + rewards[i]
        c[i] = s

    c -= np.mean(c)
    c /= np.std(c)

    return c


def reward_process(rewards):
    rewards -= np.mean(rewards)
    rewards /= np.std(rewards)

    return rewards

env = gym.make('MountainCar-v0')
# env.seed(1)
env = env.unwrapped
flag = False
max_len = 10000

for i in range(100000):
    obs = env.reset()
    features = []
    actions = []
    rewards = []

    while True:
        acts = p.run(np.reshape(obs, [1, 2]))
        action = action_choice(acts)
        obs_, reward, done, _ = env.step(action)
        data_accumulation(obs, reward, action)

        if len(rewards) > max_len:
            s = np.sum(rewards)
            print("cur round is: %s, length is larger than max" % i)
            break

        if done:
            s = np.sum(rewards)
            if len(rewards) <= max_len:
                rewards = reward_decay(rewards)
                p.learn(features, actions, rewards)
            break

        if flag:
            env.render()

        obs = obs_

    if i % 50 == 0 and i > 0 and len(rewards) <= 500:
        flag = True
    else:
        flag = False

    print("cur round is: %s, reward is: %s, len is: %s" % (i, s, len(rewards)))
