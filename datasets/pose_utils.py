import math
import os
from distutils.dir_util import mkpath

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import imresize
import tensorflow as tf

from pycocotools.coco import COCO, maskUtils
from tensorlayer import logging
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)

n_pos = 21
hin = 320
win = 384
hout = int(hin/8)
wout = int(win/8)
def get_pose_data_list(im_path, ann_path):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    print("[x] Get pose data from {}".format(im_path))
    data = PoseInfo(im_path, ann_path, False)
    imgs_file_list = data.get_image_list()
    objs_info_list = data.get_joint_list()
    mask_list = data.get_mask()
    targets = list(zip(objs_info_list, mask_list))
    if len(imgs_file_list) != len(objs_info_list):
        raise Exception("number of images and annotations do not match")
    else:
        print("{} has {} images".format(im_path, len(imgs_file_list)))
    return imgs_file_list, objs_info_list, mask_list, targets
def _data_aug_fn(image, ground_truth):
    """Data augmentation function."""
    ground_truth = cPickle.loads(ground_truth)
    ground_truth = list(ground_truth)

    annos = ground_truth[0]
    mask = ground_truth[1]
    h_mask, w_mask, _ = np.shape(image)
    # mask
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)

    for seg in mask:
        bin_mask = maskUtils.decode(seg)
        bin_mask = np.logical_not(bin_mask)
        mask_miss = np.bitwise_and(mask_miss, bin_mask)

    ## image data augmentation
    # # randomly resize height and width independently, scale is changed
    # image, annos, mask_miss = tl.prepro.keypoint_random_resize(image, annos, mask_miss, zoom_range=(0.8, 1.2))# removed hao
    # # random rotate
    # image, annos, mask_miss = tl.prepro.keypoint_random_rotate(image, annos, mask_miss, rg=15.0)# removed hao
    # # random left-right flipping
    # image, annos, mask_miss = tl.prepro.keypoint_random_flip(image, annos, mask_miss, prob=0.5)# removed hao

    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
    # M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5) # hao removed: bug, keypoints will have error
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
    # M_shear = tl.prepro.affine_shear_matrix(x_shear=(-0.1, 0.1), y_shear=(-0.1, 0.1))
    M_combined = M_rotate.dot(M_zoom)
    # M_combined = M_rotate.dot(M_flip).dot(M_zoom)#.dot(M_shear)
    # M_combined = tl.prepro.affine_zoom_matrix(zoom_range=0.9) # for debug
    h, w, _ = image.shape
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    image = tl.prepro.affine_transform_cv2(image, transform_matrix)
    mask_miss = tl.prepro.affine_transform_cv2(mask_miss, transform_matrix, border_mode='replicate')
    annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)

    # random resize height and width together
    # image, annos, mask_miss = tl.prepro.keypoint_random_resize_shortestedge(
    #     image, annos, mask_miss, min_size=(hin, win), zoom_range=(0.95, 1.6)) # removed hao
    # random crop
    # image, annos, mask_miss = tl.prepro.keypoint_random_crop(image, annos, mask_miss, size=(hin, win))  # with padding # removed hao

    image, annos, mask_miss = tl.prepro.keypoint_random_flip(image, annos, mask_miss, prob=0.5)
    image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss, size=(hin, win)) # hao add

    # generate result maps including keypoints heatmap, pafs and mask
    h, w, _ = np.shape(image)
    height, width, _ = np.shape(image)
    heatmap = get_heatmap(annos, height, width)
    vectormap = get_vectormap(annos, height, width)
    resultmap = np.concatenate((heatmap, vectormap), axis=2)

    image = np.array(image, dtype=np.float32)

    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)

    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = imresize(mask_miss, (hout, wout))
    mask_miss = np.array(mask_miss, dtype=np.float32)
    return image, resultmap, mask_miss


def _map_fn(img_list, annos):
    """TF Dataset pipeline."""
    image = tf.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Affine transform and get paf maps
    image, resultmap, mask = tf.py_func(_data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32])

    image = tf.reshape(image, [hin, win, 3])
    resultmap = tf.reshape(resultmap, [hout, wout, n_pos * 3])
    mask = tf.reshape(mask, [hout, wout, 1])

    image = tf.image.random_brightness(image, max_delta=45./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image, resultmap, mask
def _mock_map_fn(img_list, annos):
    """TF Dataset pipeline."""
    image = tf.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = np.ones((hin, win, 3), dtype=np.float32)
    resultmap = np.ones((hout, wout, 57), dtype=np.float32)
    mask = np.ones((hout, wout, 1), dtype=np.float32)

    return image, resultmap, mask
## read coco data
class CocoMeta:
    """ Be used in PoseInfo. """
    limb = list(
        zip(            
		[1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4,18,19],
		[1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4,18,19]))

    def __init__(self, idx, img_url, img_meta, annotations, masks):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])
        self.masks = masks
        joint_list = []

        for anno in annotations:
            if anno.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(anno['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]
            # if joint is marked
            joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        # 对原 COCO 数据集的转换 其中第二位之所以不一样是为了计算 Neck 等于左右 shoulder 的中点
        transform = list(
            zip([1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4,18,19],
                [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4,18,19]))
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

            # for background
            new_joint.append((-1000, -1000))
            if len(new_joint) != n_pos:
                print('The Length of joints list should be 0 or 19 but actually:', len(new_joint))
            self.joint_list.append(new_joint)


class PoseInfo:
    """ Use COCO for pose estimation, returns images with people only. """

    def __init__(self, image_base_dir, anno_path, with_mask):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.with_mask = with_mask
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        self.image_list = os.listdir(self.image_base_dir)

    @staticmethod
    def get_keypoints(annos_info):
        annolist = []
        for anno in annos_info:
            adjust_anno = {'keypoints': anno['keypoints'], 'num_keypoints': anno['num_keypoints']}
            annolist.append(adjust_anno)
        return annolist

    def get_image_annos(self):
        """Read JSON file, and get and check the image list.
        Skip missing images.
        """
        images_ids = self.coco.getImgIds()
        len_imgs = len(images_ids)
        for idx in range(len_imgs):

            images_info = self.coco.loadImgs(images_ids[idx])
            image_path = os.path.join(self.image_base_dir, images_info[0]['file_name'])
            # filter that some images might not in the list
            if not os.path.exists(image_path):
                print("[skip] json annotation found, but cannot found image: {}".format(image_path))
                continue

            annos_ids = self.coco.getAnnIds(imgIds=images_ids[idx])
            annos_info = self.coco.loadAnns(annos_ids)
            keypoints = self.get_keypoints(annos_info)

            #############################################################################
            anns = annos_info
            prev_center = []
            masks = []

            # sort from the biggest person to the smallest one
            if self.with_mask:
                persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

                for p_id in list(persons_ids):
                    person_meta = anns[p_id]

                    if person_meta["iscrowd"]:
                        masks.append(self.coco.annToRLE(person_meta))
                        continue

                    # skip this person if parts number is too low or if
                    # segmentation area is too small
                    if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                        masks.append(self.coco.annToRLE(person_meta))
                        continue

                    person_center = [
                        person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                        person_meta["bbox"][1] + person_meta["bbox"][3] / 2
                    ]

                    # skip this person if the distance to existing person is too small
                    too_close = False
                    for pc in prev_center:
                        a = np.expand_dims(pc[:2], axis=0)
                        b = np.expand_dims(person_center, axis=0)
                        dist = cdist(a, b)[0]
                        if dist < pc[2] * 0.3:
                            too_close = True
                            break

                    if too_close:
                        # add mask of this person. we don't want to show the network
                        # unlabeled people
                        masks.append(self.coco.annToRLE(person_meta))
                        continue

            ############################################################################
            total_keypoints = sum([ann.get('num_keypoints', 0) for ann in annos_info])
            if total_keypoints > 0:
                meta = CocoMeta(images_ids[idx], image_path, images_info[0], keypoints, masks)
                self.metas.append(meta)

        print("Overall get {} valid pose images from {} and {}".format(
            len(self.metas), self.image_base_dir, self.anno_path))

    def load_images(self):
        pass

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_joint_list(self):
        joint_list = []
        for meta in self.metas:
            joint_list.append(meta.joint_list)
        return joint_list

    def get_mask(self):
        mask_list = []
        for meta in self.metas:
            mask_list.append(meta.masks)
        return mask_list


def get_heatmap(annos, height, width):
    """
    Parameters
    -----------
    Returns
    --------
    """

    # 19 for coco, 15 for MPII
    num_joints = n_pos

    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((num_joints, height, width), dtype=np.float32)

    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum

        for i, points in enumerate(joint):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap = put_heatmap(joints_heatmap, i, points, 8.0)

    # 0: joint index, 1:y, 2:x
    joints_heatmap = joints_heatmap.transpose((1, 2, 0))

    # background
    joints_heatmap[:, :, -1] = np.clip(1 - np.amax(joints_heatmap, axis=2), 0.0, 1.0)

    mapholder = []
    for i in range(0, num_joints):
        a = imresize(np.array(joints_heatmap[:, :, i]), (hout, wout))
        mapholder.append(a)
    mapholder = np.array(mapholder)
    joints_heatmap = mapholder.transpose(1, 2, 0)

    return joints_heatmap.astype(np.float16)


def put_heatmap(heatmap, plane_idx, center, sigma):
    """
    Parameters
    -----------
    Returns
    --------
    """
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap


def get_vectormap(annos, height, width):
    """
    Parameters
    -----------
    Returns
    --------
    """
    num_joints = n_pos

    limb = list(
        zip(        
		[2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16, 11,14],
        [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18, 20,19]))

    vectormap = np.zeros((num_joints * 2, height, width), dtype=np.float32)
    counter = np.zeros((num_joints, height, width), dtype=np.int16)

    for joint in annos:
        if len(joint) != 19:
            print('THE LENGTH IS NOT 19 ERROR:', len(joint))
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            vectormap = cal_vectormap(vectormap, counter, i, v_start, v_end)

    vectormap = vectormap.transpose((1, 2, 0))
    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    nonzero_vector = np.nonzero(counter)

    for i, y, x in zip(nonzero_vector[0], nonzero_vector[1], nonzero_vector[2]):

        if counter[i][y][x] <= 0:
            continue
        vectormap[y][x][i * 2 + 0] /= counter[i][y][x]
        vectormap[y][x][i * 2 + 1] /= counter[i][y][x]

    mapholder = []
    for i in range(0, n_pos * 2):
        a = cv2.resize(np.array(vectormap[:, :, i]), (hout, wout), interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    mapholder = np.array(mapholder)
    vectormap = mapholder.transpose(1, 2, 0)

    return vectormap.astype(np.float16)


def cal_vectormap(vectormap, countmap, i, v_start, v_end):
    """
    Parameters
    -----------
    Returns
    --------
    """
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)

            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            countmap[i][y][x] += 1
            vectormap[i * 2 + 0][y][x] = norm_x
            vectormap[i * 2 + 1][y][x] = norm_y

    return vectormap


def fast_vectormap(vectormap, countmap, i, v_start, v_end):
    """
    Parameters
    -----------
    Returns
    --------
    """
    _, height, width = vectormap.shape[:3]
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]

    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    x_vec = (np.arange(min_x, max_x) - v_start[0]) * norm_y
    y_vec = (np.arange(min_y, max_y) - v_start[1]) * norm_x

    xv, yv = np.meshgrid(x_vec, y_vec)

    dist_matrix = abs(xv - yv)
    filter_matrix = np.where(dist_matrix > threshold, 0, 1)
    countmap[i, min_y:max_y, min_x:max_x] += filter_matrix
    for y in range(max_y - min_y):
        for x in range(max_x - min_x):
            if filter_matrix[y, x] != 0:
                vectormap[i * 2 + 0, min_y + y, min_x + x] = norm_x
                vectormap[i * 2 + 1, min_y + y, min_x + x] = norm_y
    return vectormap


def draw_results(images, heats_ground, heats_result, pafs_ground, pafs_result, masks, name=''):
    """Save results for debugging.
    Parameters
    -----------
    images : a list of RGB images
    heats_ground : a list of keypoint heat maps or None
    heats_result : a list of keypoint heat maps or None
    pafs_ground : a list of paf vector maps or None
    pafs_result : a list of paf vector maps or None
    masks : a list of mask for people
    """
    # interval = len(images)
    for i in range(len(images)):
        if heats_ground is not None:
            heat_ground = heats_ground[i]
        if heats_result is not None:
            heat_result = heats_result[i]
        if pafs_ground is not None:
            paf_ground = pafs_ground[i]
        if pafs_result is not None:
            paf_result = pafs_result[i]
        if masks is not None:
            # print(masks.shape)
            mask = masks[i, :, :, 0]
            # print(mask.shape)
            mask = mask[:, :, np.newaxis]
            # mask = masks[:,:,:,0]
            # mask = mask.reshape(hout, wout, 1)
            mask1 = np.repeat(mask, n_pos, 2)
            mask2 = np.repeat(mask, n_pos * 2, 2)
            # print(mask1.shape, mask2.shape)
        image = images[i]

        fig = plt.figure(figsize=(8, 8))
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(image)

        if pafs_ground is not None:
            a = fig.add_subplot(2, 3, 2)
            a.set_title('Vectormap_ground')
            vectormap = paf_ground * mask2
            tmp2 = vectormap.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            # tmp2_odd = tmp2_odd * 255
            # tmp2_odd = tmp2_odd.astype(np.int)
            plt.imshow(tmp2_odd, alpha=0.3)

            # tmp2_even = tmp2_even * 255
            # tmp2_even = tmp2_even.astype(np.int)
            plt.colorbar()
            plt.imshow(tmp2_even, alpha=0.3)

        if pafs_result is not None:
            a = fig.add_subplot(2, 3, 3)
            a.set_title('Vectormap result')
            if masks is not None:
                vectormap = paf_result * mask2
            else:
                vectormap = paf_result
            tmp2 = vectormap.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
            plt.imshow(tmp2_odd, alpha=0.3)

            plt.colorbar()
            plt.imshow(tmp2_even, alpha=0.3)

        if heats_result is not None:
            a = fig.add_subplot(2, 3, 4)
            a.set_title('Heatmap result')
            if masks is not None:
                heatmap = heat_result * mask1
            else:
                heatmap = heat_result
            tmp = heatmap
            tmp = np.amax(heatmap[:, :, :-1], axis=2)

            plt.colorbar()
            plt.imshow(tmp, alpha=0.3)

        if heats_ground is not None:
            a = fig.add_subplot(2, 3, 5)
            a.set_title('Heatmap ground truth')
            if masks is not None:
                heatmap = heat_ground * mask1
            else:
                heatmap = heat_ground
            tmp = heatmap
            tmp = np.amax(heatmap[:, :, :-1], axis=2)

            plt.colorbar()
            plt.imshow(tmp, alpha=0.3)

        if masks is not None:
            a = fig.add_subplot(2, 3, 6)
            a.set_title('Mask')
            # print(mask.shape, tmp.shape)
            plt.colorbar()
            plt.imshow(mask[:, :, 0], alpha=0.3)
        # plt.savefig(str(i)+'.png',dpi=300)
        # plt.show()

        mkpath(config.LOG.vis_path)
        plt.savefig(os.path.join(config.LOG.vis_path, '%s%d.png' % (name, i)), dpi=300)


def vis_annos(image, annos, name=''):
    """Save results for debugging.
    Parameters
    -----------
    images : single RGB image
    annos  : annotation, list of lists
    """

    fig = plt.figure(figsize=(8, 8))
    a = fig.add_subplot(1, 1, 1)

    plt.imshow(image)
    for people in annos:
        for idx, jo in enumerate(people):
            if jo[0] > 0 and jo[1] > 0:
                plt.plot(jo[0], jo[1], '*')

    mkpath(config.LOG.vis_path)
    plt.savefig(os.path.join(config.LOG.vis_path, 'keypoints%s%d.png' % (name, i)), dpi=300)


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