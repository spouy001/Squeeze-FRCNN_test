import pickle
import time
import cv2
import numpy as np
from keras import backend as K
from scipy.io import loadmat
import os
from xml.etree import ElementTree


class TestConfig:
    def __init__(self):
        #
        # Parameters used for testing
        #
        # Directory of test images
        self.test_image_path = "/home/bear-c/users/spouy001/MyResearch/TenesorFlow/Faster-RCNN_TF/data/LPIRC/"
        # Path to hdf5 model file
        self.model_path = "/home/mitch-b/dmis-research/Samira/2017/LPIRC2017/code2/Squeeze-FRCNN_test/model/model_frcnn.hdf5"
        # Path to model configuration file
        config_path = "/home/mitch-b/dmis-research/Samira/2017/LPIRC2017/code2/Squeeze-FRCNN_test/model/config.pickle"
        # Path to mat file for mapping WordNet ID to ImageNet DET ID and human-understandable name
        meta_path = "/home/mitch-b/dmis-research/Samira/2017/LPIRC2017/code2/Squeeze-FRCNN_test/other_resources/meta_det.mat"
        # Number of proposal regions for RPN (num_pr % num_roi == 0)
        self.num_pr = 30
        # Number of regions of interest for SPP (num_pr % num_roi == 0)
        self.num_roi = self.num_pr
        # Threshold of confidence of classification for bounding boxes to submit
        self.bbox_threshold = 0.4
        # Batch size
        self.batch_size = 30

        #
        # Parameters derived from training
        #
        # Load parameters for training from configuration file
        with open(config_path, 'rb') as config_file:
            train_config = pickle.load(config_file)
        # Anchor box scales
        self.anchor_box_scales = train_config.anchor_box_scales
        self.anchor_box_scales[0] /= 2
        self.anchor_box_scales[1] /= 2
        self.anchor_box_scales[2] /= 2
        # Anchor box ratios
        self.anchor_box_ratios = train_config.anchor_box_ratios
        # Stride at RPN
        self.rpn_stride = train_config.rpn_stride
        # The smallest side of the image for image resize
        self.image_size = train_config.im_size / 2
        # Image channel-wise mean
        self.img_channel_mean = train_config.img_channel_mean
        # Standard scaling factor (for rpn2roi)
        self.std_scaling = train_config.std_scaling
        # Standard scaling factor (for bounding box regression)
        self.classifier_regr_std = train_config.classifier_regr_std
        # Concept mapping (Model Concept ID -> (ImageNet DET ID, Name))
        self.concept_mapping = []
        meta_det = loadmat(meta_path)
        synset_info = []
        for i in xrange(200):
            synset = meta_det['synsets'][0, i]
            det_id = int(synset[0][0, 0])
            wnid = str(synset[1][0])
            assert synset[2].shape == (1,)
            name = str(synset[2][0])
            synset_info.append({'det_id': det_id, 'wnid': wnid, 'name': name})
            self.concept_mapping.append({'det_id': -1, 'name': ''})
        for wnid, modelID in train_config.class_mapping.items():
            for synset in synset_info:
                if synset['wnid'] == wnid:
                    self.concept_mapping[modelID] = {'det_id': synset['det_id'], 'name': synset['name'], 'class':synset['wnid'] }
        self.concept_mapping.append({'det_id': 0, 'name': 'bg', 'class': ''})

    def num_anchors(self):
        return len(self.anchor_box_scales) * len(self.anchor_box_ratios)

    def preprocess_image(self, img):
        # Resize image (Keep a fixed size for batch training)
        (height, width, _) = img.shape
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        scale_ratio = (self.image_size / float(height), self.image_size / float(width))

        # Format image
        img = img[:, :, (2, 1, 0)]
        img = img.astype(K.floatx())
        img[:, :, 0] -= self.img_channel_mean[0]
        img[:, :, 1] -= self.img_channel_mean[1]
        img[:, :, 2] -= self.img_channel_mean[2]
        # img = np.transpose(img, (2, 0, 1))
        return img, scale_ratio
