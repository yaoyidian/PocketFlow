import tensorflow as tf
from nets.resnet50_factory import conv_layer,max_pool,res_block_3_layers,concat_layer,transpose_layer,upsample

def resnet_openpose_build(inputs,num_joints, num_pafs,is_training=True):
    with tf.variable_scope('BackBone', reuse=tf.AUTO_REUSE):
        
        #image_max = tf.reduce_max(inputs, name='image_max')
        #image_min = tf.reduce_min(inputs, name='image_min')
        #3 inputs  320x320   net0 160x160
        net0 = conv_layer(inputs, out_channels=64 , kernel_size=7, stride=2,bn=False,training=is_training)
        #net1 80x80
        net1 = max_pool(net0, kernel_size=3,stride=2, name='pool_1')
        #net10 80x80
        net10 = conv_layer(net1, out_channels=256, kernel_size=1, stride=1, relu=False,bn=False,training=is_training)
        net2=res_block_3_layers(net1,[64,64,256],stride=1,is_training=True)  #net2 46x46
        net21=concat_layer(net10,net2)

        net2_1 = res_block_3_layers(net21, [64, 64, 256],stride=1, is_training=is_training)
        net22 = concat_layer(net21, net2_1)

        net2_2 = res_block_3_layers(net22, [64, 64, 256],stride=1, is_training=is_training)
        net23 = concat_layer(net22, net2_2)

        #net3 net30 40x40
        net3 = res_block_3_layers(net23, [128, 128, 512],stride=2, is_training=True)
        net30=conv_layer(net23, out_channels=512, kernel_size=1, stride=2, relu=False,bn=False,training=is_training)
        net31 = concat_layer(net3, net30)

        net3_1 = res_block_3_layers(net31, [128, 128, 512],stride=1, is_training=is_training)
        net32 = concat_layer(net31, net3_1)

        net3_2 = res_block_3_layers(net32, [128, 128, 512], stride=1,is_training=is_training)
        net33 = concat_layer(net32, net3_2)

        net3_3 = res_block_3_layers(net33, [128, 128, 512],stride=1, is_training=is_training)
        net34 = concat_layer(net33, net3_3)   #512

        #net4 net40 20x20
        net4 = res_block_3_layers(net34, [256, 256, 1024],stride=2, is_training=True)
        net40 = conv_layer(net34, out_channels=1024, kernel_size=1, stride=2, relu=False, bn=False,training=is_training)
        net41 = concat_layer(net4, net40)

        net4_1 = res_block_3_layers(net41, [256, 256, 1024],stride=1, is_training=is_training)
        net42 = concat_layer(net41, net4_1)

        net4_2 = res_block_3_layers(net42, [256, 256, 1024],stride=1, is_training=is_training)
        net43 = concat_layer(net42, net4_2)

        net4_3 = res_block_3_layers(net43, [256, 256, 1024],stride=1, is_training=is_training)
        net44 = concat_layer(net43, net4_3)

        net4_4 = res_block_3_layers(net44, [256, 256, 1024],stride=1, is_training=is_training)
        net45 = concat_layer(net44, net4_4)  #1024

        #net5 net50 10x10
        net5 = res_block_3_layers(net44, [512, 512, 2048],stride=2, is_training=True)
        net50 = conv_layer(net45, out_channels=2048, kernel_size=1, stride=2, relu=False, bn=False,training=is_training)
        net51 = concat_layer(net5, net50)

        net5_1 = res_block_3_layers(net51, [512, 512, 2048],stride=1, is_training=is_training)
        net52 = concat_layer(net51, net5_1)

        net5_2 = res_block_3_layers(net52, [512, 512, 2048],stride=1, is_training=is_training)
        net53 = concat_layer(net52, net5_2)

        #net53  256x1048x1x1
        net60 = conv_layer(net53, out_channels=256, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)
        net60 = upsample(net60, 2, 'upsample_1')
        net60 = upsample(net60, 2, 'upsample_2')
        output_5= conv_layer(net60, out_channels=256, kernel_size=3, stride=1,bn=False, training=is_training)
        #output_5 40x40

        #net45 1024x256x1x1
        net61 = conv_layer(net45, out_channels=256, kernel_size=1, stride=1, relu=False, bn=False,training=is_training)
        net61 = upsample(net61, 2, 'upsample_3')
        output_4 = conv_layer(net61, out_channels=256, kernel_size=3, stride=1,bn=False, training=is_training)


        net62 = conv_layer(net34, out_channels=256, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)
        output_3 = conv_layer(net62, out_channels=256, kernel_size=3, stride=1, bn=False,training=is_training)

        net63 = conv_layer(net23, out_channels=256, kernel_size=1, stride=2, relu=False,bn=False, training=is_training)
        net63 = conv_layer(net63, out_channels=256, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)
        output_2 = conv_layer(net63, out_channels=256, kernel_size=3, stride=1,bn=False,  training=is_training)

        net64 = conv_layer(net1, out_channels=256, kernel_size=1, stride=2, relu=False, bn=False, training=is_training)
        net64 = conv_layer(net64, out_channels=256, kernel_size=1, stride=1, relu=False,bn=False,  training=is_training)
        net64 = conv_layer(net64, out_channels=256, kernel_size=1, stride=1, relu=False,bn=False,  training=is_training)
        output_1 = conv_layer(net64, out_channels=256, kernel_size=3, stride=1, bn=False,  training=is_training)

        cancat_resnet = tf.concat(
            [
                output_1,
                output_2,
                output_3,
                output_4,
                output_5
            ]
            , axis=3)

    with tf.variable_scope('Cpm', reuse=tf.AUTO_REUSE):
        net_cpm= conv_layer(cancat_resnet, out_channels=512, kernel_size=3, stride=1, bn=False,training=is_training)
        net_cpm = conv_layer(net_cpm, out_channels=512, kernel_size=3, stride=1,bn=False,training=is_training)
    with tf.variable_scope('RefinementStage_1', reuse=tf.AUTO_REUSE):
        net_stage1 = conv_layer(net_cpm, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage1 = conv_layer(net_stage1, out_channels=128, kernel_size=3, stride=1,bn=False,  training=is_training)
        net_stage1 = conv_layer(net_stage1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage1 = conv_layer(net_stage1, out_channels=128, kernel_size=3, stride=1, bn=False, training=is_training)
        net_stage1 = conv_layer(net_stage1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage1 = conv_layer(net_stage1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        paf1 = conv_layer(net_stage1, out_channels=128, kernel_size=1, stride=1,bn=False,  training=is_training)
        paf1 = conv_layer(paf1, out_channels=num_pafs, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)

        net_stage1_1 = conv_layer(net_cpm, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage1_1 = conv_layer(net_stage1_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage1_1 = conv_layer(net_stage1_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage1_1 = conv_layer(net_stage1_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage1_1 = conv_layer(net_stage1_1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage1_1 = conv_layer(net_stage1_1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        heatmap1 = conv_layer(net_stage1_1, out_channels=128, kernel_size=1, stride=1, bn=False,training=is_training)
        heatmap1 = conv_layer(heatmap1, out_channels=num_joints, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)
        stages_output = [[heatmap1, paf1]]
        outs1 = tf.concat([net_cpm, heatmap1, paf1], axis=-1)
    with tf.variable_scope('RefinementStage_2', reuse=tf.AUTO_REUSE):
        net_stage2 = conv_layer(outs1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage2 = conv_layer(net_stage2, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage2 = conv_layer(net_stage2, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage2 = conv_layer(net_stage2, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage2 = conv_layer(net_stage2, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage2 = conv_layer(net_stage2, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        paf2 = conv_layer(net_stage2, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        paf2 = conv_layer(paf2, out_channels=num_pafs, kernel_size=1, stride=1, relu=False, bn=False,training=is_training)

        net_stage2_1 = conv_layer(outs1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage2_1 = conv_layer(net_stage2_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage2_1 = conv_layer(net_stage2_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage2_1 = conv_layer(net_stage2_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage2_1 = conv_layer(net_stage2_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage2_1 = conv_layer(net_stage2_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        heatmap2 = conv_layer(net_stage2_1, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        heatmap2 = conv_layer(heatmap2, out_channels=num_joints, kernel_size=1, stride=1, relu=False, bn=False,training=is_training)
        stages_output.append([heatmap2, paf2])
        outs2 = tf.concat([net_cpm, heatmap2, paf2], axis=-1)
    with tf.variable_scope('RefinementStage_3', reuse=tf.AUTO_REUSE):
        net_stage3 = conv_layer(outs2, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3 = conv_layer(net_stage3, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3 = conv_layer(net_stage3, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3 = conv_layer(net_stage3, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3 = conv_layer(net_stage3, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3 = conv_layer(net_stage3, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        paf3 = conv_layer(net_stage3, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        paf3 = conv_layer(paf3, out_channels=num_pafs, kernel_size=1, stride=1, relu=False, bn=False,training=is_training)

        net_stage3_1 = conv_layer(outs2, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3_1 = conv_layer(net_stage3_1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3_1 = conv_layer(net_stage3_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage3_1 = conv_layer(net_stage3_1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3_1 = conv_layer(net_stage3_1, out_channels=128, kernel_size=3, stride=1, bn=False,training=is_training)
        net_stage3_1 = conv_layer(net_stage3_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        heatmap3 = conv_layer(net_stage3_1, out_channels=128, kernel_size=1, stride=1, bn=False,training=is_training)
        heatmap3 = conv_layer(heatmap3, out_channels=num_joints, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)
        stages_output.append([heatmap3, paf3])
        outs3 = tf.concat([net_cpm, heatmap3, paf3], axis=-1)
    with tf.variable_scope('RefinementStage_4', reuse=tf.AUTO_REUSE):
        net_stage4 = conv_layer(outs3, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4 = conv_layer(net_stage4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4 = conv_layer(net_stage4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4 = conv_layer(net_stage4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4 = conv_layer(net_stage4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4 = conv_layer(net_stage4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        paf4 = conv_layer(net_stage4, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        paf4 = conv_layer(paf4, out_channels=num_pafs, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)

        net_stage4_1 = conv_layer(outs3, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4_1 = conv_layer(net_stage4_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4_1 = conv_layer(net_stage4_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4_1 = conv_layer(net_stage4_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4_1 = conv_layer(net_stage4_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage4_1 = conv_layer(net_stage4_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        heatmap4 = conv_layer(net_stage4_1, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        heatmap4 = conv_layer(heatmap4, out_channels=num_joints, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)
        stages_output.append([heatmap4, paf4])
        outs4 = tf.concat([net_cpm, heatmap4, paf4], axis=-1)
    with tf.variable_scope('RefinementStage_5', reuse=tf.AUTO_REUSE):
        net_stage5 = conv_layer(outs4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5 = conv_layer(net_stage5, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5 = conv_layer(net_stage5, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5 = conv_layer(net_stage5, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5 = conv_layer(net_stage5, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5 = conv_layer(net_stage5, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        paf5 = conv_layer(net_stage5, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        paf5 = conv_layer(paf5, out_channels=num_pafs, kernel_size=1, stride=1, relu=False,bn=False, training=is_training)

        net_stage5_1 = conv_layer(outs4, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5_1 = conv_layer(net_stage5_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5_1 = conv_layer(net_stage5_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5_1 = conv_layer(net_stage5_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5_1 = conv_layer(net_stage5_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        net_stage5_1 = conv_layer(net_stage5_1, out_channels=128, kernel_size=3, stride=1,bn=False, training=is_training)
        heatmap5 = conv_layer(net_stage5_1, out_channels=128, kernel_size=1, stride=1,bn=False, training=is_training)
        heatmap5 = conv_layer(heatmap5, out_channels=num_joints, kernel_size=1, stride=1, relu=False, bn=False,training=is_training)
        stages_output.append([heatmap5, paf5])

    return stages_output

if __name__ == '__main__':
    import os

    modelpath='F:/progrogram/ResNet50-Tensorflow-Face-Recognition-master/models/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    inputs = tf.placeholder(tf.float32, shape=(1, 320, 320, 3))
    net = resnet_openpose_build(inputs, 21, 42, True)

    saver = tf.train.Saver()
    global_step = tf.Variable(0, trainable=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.Graph().as_default() as tf_graph:
            saver.save(sess, os.path.join(modelpath, 'model'), global_step=global_step)