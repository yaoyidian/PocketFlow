import tensorflow as tf
import tensorlayer as tl
from nets.abstract_model_helper import AbstractModelHelper
from datasets.pose_dataset import Dataset
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from nets.resnet50_openpose import resnet_openpose_build

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\' ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 16, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')

n_pos = 21
def tf_repeat(tensor, repeats):
    """
    Args:
    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    Returns:
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor
    
def forward_fn(inputs,is_train=True):
    inputs = resnet_openpose_build(inputs,21,42,is_train)
    if not is_train:
        inputs = inputs[-1]
    return inputs

def calc_loss_fn(outputs,objects):
    results = objects['resultmap']
    mask = objects['mask']
    
    confs = results[:, :, :, :n_pos]
    pafs = results[:, :, :, n_pos:]
    m1 = tf_repeat(mask, [1, 1, 1, n_pos])
    m2 = tf_repeat(mask, [1, 1, 1, n_pos * 2])
    b1_list = [ outputs[i][0] for i in range(len(outputs))]
    b2_list = [ outputs[i][1] for i in range(len(outputs))]
    # define loss
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    stage_losses = []
    
    for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
        loss_l1 = tf.nn.l2_loss((l1 - confs) * m1)
        loss_l2 = tf.nn.l2_loss((l2 - pafs) * m2)

        losses.append(tf.reduce_mean([loss_l1, loss_l2]))
        stage_losses.append(loss_l1 / FLAGS.batch_size)
        stage_losses.append(loss_l2 / FLAGS.batch_size)

    last_conf = b1_list[-1]
    last_paf = b2_list[-1]
    last_losses_l1.append(loss_l1)
    last_losses_l2.append(loss_l2)
    l2_loss = 0.0

    for p in tl.layers.get_variables_with_name('kernel', True, True):
        l2_loss += tf.contrib.layers.l2_regularizer(FLAGS.loss_w_dcy)(p)
    total_loss = tf.reduce_sum(losses) / FLAGS.batch_size + l2_loss
    return total_loss

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a ConvNet model for the Fashion-MNIST dataset."""

  def __init__(self):
    """Constructor function."""
    data_format='channel_last'
    # class-independent initialization
    super(ModelHelper, self).__init__(data_format)

    # initialize training & evaluation subsets
    self.dataset_train = Dataset(is_train=True)
    self.dataset_eval = Dataset(is_train=False)

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build(enbl_trn_val_split)

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs):
    """Forward computation at training."""

    return forward_fn(inputs)

  def forward_eval(self, inputs):
    """Forward computation at evaluation."""

    return forward_fn(inputs,False)

  def calc_loss(self,  objects, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    loss = calc_loss_fn(outputs,objects)
    
    # metrics = {'accuracy': accuracy}

    return loss

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    nb_epochs = 160
    idxs_epoch = [40, 80, 120]
    decay_rates = [1.0, 0.1, 0.01, 0.001]
    batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
    nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

    return lrn_rate, nb_iters

  @property
  def model_name(self):
    """Model's name."""

    return 'resnet50_pose'

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'coco_pose'