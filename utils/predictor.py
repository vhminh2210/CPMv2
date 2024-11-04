import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tf_slim as slim
# import tensorflow.contrib.layers as slim
# from tensorflow.contrib.framework import slim.arg_scope


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def resBlock(
    x,
    num_outputs,
    kernel_size=4,
    stride=1,
    activation_fn=tf.nn.relu,
    normalizer_fn=slim.batch_norm,
    scope=None,
):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
    with tf.compat.v1.variable_scope(scope, "resBlock"):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = slim.conv2d(
                shortcut,
                num_outputs,
                kernel_size=1,
                stride=stride,
                activation_fn=None,
                normalizer_fn=None,
                scope="shortcut",
            )
        x = slim.conv2d(x, num_outputs / 2, kernel_size=1, stride=1, padding="SAME")
        x = slim.conv2d(x, num_outputs / 2, kernel_size=kernel_size, stride=stride, padding="SAME")
        x = slim.conv2d(
            x,
            num_outputs,
            kernel_size=1,
            stride=1,
            activation_fn=None,
            padding="SAME",
            normalizer_fn=None,
        )

        x += shortcut
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name="resfcn256"):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training=True):
        with tf.compat.v1.variable_scope(self.name) as scope:
            with slim.arg_scope([slim.batch_norm], is_training=is_training, scale=True):
                with slim.arg_scope(
                    [slim.conv2d, slim.conv2d_transpose],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    biases_initializer=None,
                    padding="SAME",
                    weights_regularizer=slim.l2_regularizer(0.0002),
                ):
                    size = 16
                    # x: s x s x 3
                    se = slim.conv2d(x, num_outputs=size, kernel_size=4, stride=1)  # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512

                    pd = slim.conv2d_transpose(se, size * 32, 4, stride=1)  # 8 x 8 x 512
                    pd = slim.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
                    pd = slim.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = slim.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = slim.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
                    pd = slim.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = slim.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = slim.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
                    pd = slim.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
                    pd = slim.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64

                    pd = slim.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
                    pd = slim.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
                    pd = slim.conv2d_transpose(pd, size, 4, stride=2)  # 256 x 256 x 16
                    pd = slim.conv2d_transpose(pd, size, 4, stride=1)  # 256 x 256 x 16

                    pd = slim.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pd = slim.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pos = slim.conv2d_transpose(
                        pd, 3, 4, stride=1, activation_fn=tf.nn.sigmoid
                    )  # , padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

                    return pos

    @property
    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]


class PosPrediction:
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = resfcn256(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        self.x_op = self.network(self.x, is_training=False)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

    def restore(self, model_path):
        tf.compat.v1.train.Saver(self.network.vars).restore(self.sess, model_path)

    def predict(self, image):
        pos = self.sess.run(self.x_op, feed_dict={self.x: image[np.newaxis, :, :, :]})
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, feed_dict={self.x: images})
        return pos * self.MaxPos
