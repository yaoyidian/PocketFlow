import os
import gzip
import numpy as np
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset
from datasets.pose_utils import PoseInfo, draw_results, get_heatmap, get_vectormap, tf_repeat,get_pose_data_list,_data_aug_fn,_map_fn,_mock_map_fn
from pycocotools.coco import maskUtils
import _pickle as cPickle
import tensorlayer as tl
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_smpls_train', 50000, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 5000, '# of samples for validation')
tf.app.flags.DEFINE_integer('batch_size', 8, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 8, 'batch size for evaluation')
#tf.app.flags.DEFINE_integer('prefetch_size', 80, 'batch size for prefetch')
# Fashion-MNIST specifications
IMAGE_HEI = 320
IMAGE_WID = 384
IMAGE_CHN = 3

def parse_fn(img_list, annos, is_train):
  if is_train:
    image, resultmap, mask = _map_fn(img_list, annos)
  else:
    image, resultmap, mask = _mock_map_fn(img_list, annos)
  Objects = {'resultmap':resultmap, 'mask':mask}
  return image, Objects

class Dataset(AbstractDataset):
  '''Dataset dataset.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """
    
    # initialize the base class
    super(Dataset, self).__init__(is_train)
    path = FLAGS.data_dir_local
    if is_train:
        images_path = os.path.join(path, "train2017")
        annotations_file_path = os.path.join(path, "annotations_foot_body", "person_keypoints_train2017.json")
        imgs_file_list, objs_info_list, mask_list, targets = \
            get_pose_data_list(images_path, annotations_file_path)
    else:
        images_path = os.path.join(path, "train2017")
        annotations_file_path = os.path.join(path, "annotations_foot_body", "person_keypoints_train2017.json")
        imgs_file_list, objs_info_list, mask_list, targets = \
            get_pose_data_list(images_path, annotations_file_path)
    def generator():
        """TF Dataset generator."""
        assert len(imgs_file_list) == len(targets)
        for _input, _target in zip(imgs_file_list, targets):
            yield _input.encode('utf-8'), cPickle.dumps(_target)
    self.dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string))
    self.parse_fn = lambda x,y : parse_fn(x, y ,is_train)
  def build(self, enbl_trn_val_split=False):
    """Build iterator(s) for tf.data.Dataset() object.
    
    Args:
    * enbl_trn_val_split: whether to split into training & validation subsets

    Returns:
    * iterator_trn: iterator for the training subset
    * iterator_val: iterator for the validation subset
      OR
    * iterator: iterator for the chosen subset (training OR testing)
    """

    # create a tf.data.Dataset() object from NumPy arrays
    
    dataset = self.dataset
    dataset = dataset.map(self.parse_fn, num_parallel_calls=FLAGS.nb_threads)
    # create iterators for training & validation subsets separately
    if self.is_train and enbl_trn_val_split:
      iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val))
      iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_val))
      return iterator_trn, iterator_val
    dataset = dataset.shuffle(buffer_size = FLAGS.buffer_size)
    dataset = dataset.repeat(16)
    return self.__make_iterator(dataset)
  def __make_iterator(self, dataset):
    """Make an iterator from tf.data.Dataset.

    Args:
    * dataset: tf.data.Dataset object

    Returns:
    * iterator: iterator for the dataset
    """

    #dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size))
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator
