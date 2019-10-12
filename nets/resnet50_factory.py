import tensorflow as tf

def conv_layer(inputs, out_channels, kernel_size=3,stride=1, bias=True,bn=True,relu=True,training=True,name=''):
    net = tf.layers.conv2d(inputs=inputs, filters=out_channels,kernel_size=kernel_size,
                         strides=stride, use_bias=bias, padding='SAME',name=name)

    if bn:
        net = tf.layers.batch_normalization(net, training=training)
    if relu:
        net = tf.nn.relu(net)

    return net


def max_pool(inputs, kernel_size, stride, name, padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding,
                          name=name)


def res_block_3_layers(inputs, channel_list, stride=1,is_training=True,name=''):
    block_conv_1 = conv_layer(inputs,channel_list[0],kernel_size=1,stride=stride,bn=False,training=is_training,name=name)
    block_conv_2 = conv_layer(block_conv_1,channel_list[1],kernel_size=3,stride=1,bn=False,training=is_training,name=name)
    block_conv_3 = conv_layer(block_conv_2,channel_list[2],kernel_size=1,stride=1,bn=False,relu=False,training=is_training,name=name)
    return block_conv_3
def concat_layer(input1,input2):
    output = tf.add(input1, input2)
    output = tf.nn.relu(output)
    return output

def transpose_layer(inputs,channels=256,kernel_size=4,stride=2,name=''):
    '''
    out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                                trainable=trainable, weights_initializer=normal_initializer,
                                padding='SAME', activation_fn=tf.nn.relu,
                                scope='up1')
    '''
    outputs = tf.layers.conv2d_transpose(inputs,
                                         channels,
                                         [kernel_size,kernel_size],
                                         strides=(stride, stride),
                                         padding='same',
                                         activation=None,  ## no activation
                                         use_bias=True,  ## use bias
                                         )

    return outputs

def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)
