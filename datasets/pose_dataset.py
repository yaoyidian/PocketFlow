import os
import gzip
import numpy as np
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset
from pose_utils import PoseInfo, draw_results, get_heatmap, get_vectormap, load_mscoco_dataset, tf_repeat,get_pose_data_list,_data_aug_fn,_map_fn,_mock_map_fn
from pycocotools.coco import maskUtils
import _pickle as cPickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 10, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 60000, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 5000, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 10000, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 100, 'batch size for evaluation')

# Fashion-MNIST specifications
IMAGE_HEI = 320
IMAGE_WID = 384
IMAGE_CHN = 3

def parse_fn(image, label, is_train):
  """Parse an (image, label) pair and apply data augmentation if needed.

  Args:
  * image: image tensor
  * label: label tensor
  * is_train: whether data augmentation should be applied

  Returns:
  * image: image tensor
  * label: one-hot label tensor
  """

  # data parsing
  label = tf.one_hot(tf.reshape(label, []), FLAGS.nb_classes)
  image = tf.cast(tf.reshape(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN]), tf.float32)
  image = tf.image.per_image_standardization(image)

  # data augmentation
  if is_train:
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEI + 8, IMAGE_WID + 8)
    image = tf.random_crop(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN])
    image = tf.image.random_flip_left_right(image)

  return image, label

class Dataset(AbstractDataset):
  '''Dataset dataset.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """
	
    # initialize the base class
    super(FMnistDataset, self).__init__(is_train)
	if is_train:
		images_path = os.path.join(path, "train2017")
		annotations_file_path = os.path.join(path, "annotations_foot_body", "person_keypoints_train2017.json")
		imgs_file_list, objs_info_list, mask_list, targets = \
            get_pose_data_list(train_im_path, train_ann_path)
	else:
		images_path = os.path.join(path, "annotations_foot_body", "person_keypoints_val2017.json")
		annotations_file_path = os.path.join(path, "val2017")
		imgs_file_list, objs_info_list, mask_list, targets = \
            get_pose_data_list(val_im_path, val_ann_path)
	def generator():
        """TF Dataset generator."""
        assert len(imgs_file_list) == len(train_targets)
        for _input, _target in zip(imgs_file_list, train_targets):
            yield _input.encode('utf-8'), cPickle.dumps(_target)
	self.generator=generator()
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
    dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string))

    # create iterators for training & validation subsets separately
    if self.is_train and enbl_trn_val_split:
      iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val))
      iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_val))
      return iterator_trn, iterator_val

    return self.__make_iterator(dataset)

  def __make_iterator(self, dataset):
    """Make an iterator from tf.data.Dataset.

    Args:
    * dataset: tf.data.Dataset object

    Returns:
    * iterator: iterator for the dataset
    """

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size))
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator