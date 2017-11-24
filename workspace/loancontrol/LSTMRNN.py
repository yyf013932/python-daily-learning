import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

n_input = 1
n_output = 1
# LSTM隐层单元数量
cell_size = 64
# 序列长度
time_step = 24
batch_start = 0
# batch大小
train_size = 720
test_size = 240
# 学习率
lr = 0.6
# 迭代次数
iter_times = 200
prefix = 't40b200r0_6_'


def weight_variable(shape, name):
    data = tf.truncated_normal(stddev=0.01, shape=shape)
    return tf.Variable(data, name=name)


# 初始化偏置单元
def bais_variable(shape, name):
    data = tf.constant(0.01, shape=shape)
    return tf.Variable(data, name=name)


def get_format_data(d, time_steps):
    leng = len(d)
    x = []
    y = []
    for i in range(leng - time_steps - 1):
        x.append(d[i:i + time_steps])
        y.append(d[i + time_steps])
    return x, y


with tf.name_scope('input_layer'):
    w_in = weight_variable([n_input, cell_size], name='W_in')
    b_in = bais_variable([cell_size], name="b_in")

with tf.name_scope('output_layer'):
    w_out = weight_variable([cell_size, n_output], "W_out")
    b_out = bais_variable([n_output], "b_out")

with tf.name_scope('cell_layer'):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=1.0, state_is_tuple=True)

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, time_step, n_input], name="xs")
    ys = tf.placeholder(tf.float32, [None, n_output], name="ys")


def lstm(batch_size):
    with tf.name_scope('input_layer'):
        i = tf.reshape(xs, [-1, n_input])
        i_rnn = tf.matmul(i, w_in) + b_in
        c_in = tf.reshape(i_rnn, [-1, time_step, cell_size])
    with tf.name_scope('cell_layer'):
        with tf.name_scope('initial_state'):
            cell_initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            # cell_out [batch,time_step,cell_size]   cell_final_state [cell_size]
            cell_out, cell_final_state = tf.nn.dynamic_rnn(lstm_cell, c_in,
                                                           initial_state=cell_initial_state,
                                                           time_major=False)
    with tf.name_scope('output_layer'):
        out_x = tf.reshape(cell_out[:, -1, :], [-1, cell_size])
        out = tf.matmul(out_x, w_out) + b_out
    return out, cell_final_state


out, cfs = lstm(train_size)

with tf.name_scope('loss'):
    losses = tf.reduce_mean(tf.square(out - ys))

with tf.name_scope('train_step'):
    train_step = tf.train.AdadeltaOptimizer(lr).minimize(losses)

with tf.name_scope('summary'):
    tf.summary.histogram(prefix + "w_in", w_in)
    tf.summary.histogram(prefix + "b_in", b_in)
    tf.summary.histogram(prefix + "W_out", w_out)
    tf.summary.histogram(prefix + "b_out", b_out)
    tf.summary.scalar(prefix + "loss", losses)
    merged = tf.summary.merge_all()

data = pd.read_csv('D:\\resources\\loan-control\\trade-8-9-10m.csv', header=None)

data.columns = ['time', 'value']

data['date'] = data['time'].apply(lambda x: x.split()[0])
data['month'] = data['date'].apply(lambda x: int(x.split('-')[1]))
data['day'] = data['date'].apply(lambda x: int(x.split('-')[2]))
data['hour'] = data['time'].apply(lambda x: int(x.split()[1][0:2]))

data_gb = data.groupby(by=['month', 'day', 'hour']).agg(np.sum).reset_index()

hour_data = data_gb['value'].values
day_data = data_gb.groupby(by=['month', 'day']).value.agg('sum').values
hour_cross_day_data = []
for i in range(24):
    t = data_gb[data_gb.hour == i].value.values
    hour_cross_day_data.append(t)

day_data_norm = (day_data - np.mean(day_data)) / np.std(day_data)
hour_data_norm = (hour_data - np.mean(hour_data)) / np.std(hour_data)

data_input = day_data_norm[:, np.newaxis]

tx, ty = get_format_data(hour_data_norm[-(train_size + test_size):, np.newaxis], time_step)

x_train = tx[:train_size]
y_train = ty[:train_size]
x_test = tx[train_size:]
y_test = ty[train_size:]

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
fileWriter = tf.summary.FileWriter('logs/lstm', sess.graph)
for i in range(iter_times):
    _, __, summaried = sess.run([train_step, losses, merged], feed_dict={xs: x_train, ys: y_train})
    fileWriter.add_summary(summaried, i)
    if i % 50 == 0:
        print(i, " step")
fileWriter.close()

leng = len(tx)
x = np.arange(0, leng)

plt.plot(x, ty, 'r')

out1 = lstm(len(x_train))
y_pre = sess.run([out1], feed_dict={xs: x_train})
plt.plot(x[:train_size], y_pre[0][0], 'g')

out1 = lstm(len(x_test))
y_pre = sess.run([out1], feed_dict={xs: x_test})
plt.plot(x[train_size:], y_pre[0][0], 'b')
