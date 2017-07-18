import time
import math
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from squeeze_frcnn.squeezenet import squeezenet
from squeeze_frcnn.region_proposal_network import rpn
from squeeze_frcnn.classifier import cls_net, roi_pooling_net

assert (K.backend() == 'tensorflow')
assert (K.image_dim_ordering() == 'tf')
assert (K.image_data_format() == 'channels_last')


def non_max_suppression_fast(boxes, probs, overlap_threshold, max_boxes):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    # TODO: Check the most efficient data type
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    # TODO: Check the most efficient data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def apply_regr_np(X, T):
    try:
        # TODO: Simplify exp op
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw) * w
        h1 = np.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print("Exception in apply_regr_np: ")
        print(e)
        return X


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        # TODO: Simplify exp op
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        print("ValueError in apply_regr")
        return x, y, w, h
    except OverflowError:
        print("Overflow in apply_regr")
        return x, y, w, h
    except Exception as e:
        print("Exception in apply_regr:")
        print(e)
        return x, y, w, h


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    ratio_x = ratio[1]
    ratio_y = ratio[0]
    real_x1 = int(round(x1 // ratio_x))
    real_y1 = int(round(y1 // ratio_y))
    real_x2 = int(round(x2 // ratio_x))
    real_y2 = int(round(y2 // ratio_y))

    return real_x1, real_y1, real_x2 ,real_y2


class SqueezeModel:
    def __init__(self, cfg):
        assert(cfg.__class__.__name__ in ['TestConfig', 'TrainConfig'])
        self.cfg = cfg
        if cfg.__class__.__name__ == 'TestConfig':
            # Define the layers of model
            img_input = Input(shape=(None, None, 3))
            shared_layers = squeezenet(img_input, trainable=False)
            rpn_layers = rpn(shared_layers, cfg.num_anchors())

            roi_input = Input(shape=(cfg.num_roi, 4))
            feature_input = Input(shape=(None, None, shared_layers.shape[3].value))
            out_roi_pool = roi_pooling_net(feature_input, roi_input, cfg.num_roi)
            cls_input = Input(shape=(cfg.num_roi, 14, 14, 512))
            classifier = cls_net(cls_input, cfg.num_roi, nb_classes=len(cfg.concept_mapping))
            # Define the model
            self.rpn_model = Model(img_input, rpn_layers)
            self.ssp_model = Model([feature_input, roi_input], out_roi_pool)
            self.cls_model = Model(cls_input, classifier)
            # Load weights
            self.rpn_model.load_weights(cfg.model_path, by_name=True)
            self.ssp_model.load_weights(cfg.model_path, by_name=True)
            self.cls_model.load_weights(cfg.model_path, by_name=True)

    def apply_rpn_model(self, img):
        # Generate proposal regions
        [rpn_layer, regr_layer, features] = self.rpn_model.predict(img)
        return rpn_layer, regr_layer, features

    def post_rpn_model(self, rpn_layer, regr_layer):
        # Convert proposal regions to rois
        rois = self.rpn_to_roi(rpn_layer, regr_layer, 0.7)
        # Convert from (x1,y1,x2,y2) to (x,y,w,h)
        rois[:, :, 2] -= rois[:, :, 0]
        rois[:, :, 3] -= rois[:, :, 1]

        return rois

    def apply_ssp_model(self, features, rois):
        all_out_roi_pool = []
        # TODO: Can this part be processed in parallel
        for image_id in xrange(features.shape[0]):
            this_features = features[image_id, :, :, :]
            this_rois = rois[image_id, :, :]
            this_features = np.expand_dims(this_features, axis=0)
            this_rois = np.expand_dims(this_rois, axis=0)
            out_roi_pool = self.ssp_model.predict([this_features, this_rois])
            all_out_roi_pool.append(out_roi_pool[0, :, :, :, :])

        return np.asarray(all_out_roi_pool)

    def apply_cls_model(self, rois):
        [prob_cls, prob_regr] = self.cls_model.predict(rois)
        return prob_cls, prob_regr

    def post_cls_model(self, rois, prob_cls, prob_regr, scale_ratio):
        # filter bbox and apply regression
        all_bboxes = []
        for image_id in xrange(prob_cls.shape[0]):
            bboxes = {}
            probs = {}
            # Find all bounding boxed with acceptable confidence
            for bbox_id in xrange(prob_cls.shape[1]):
                # Find top-1 concept
                mid = np.argmax(prob_cls[image_id, bbox_id, :])
                # Verify confidence and outliers
                if mid == (prob_cls.shape[2] - 1) or prob_cls[image_id, bbox_id, mid] < self.cfg.bbox_threshold:
                    continue
                # Use Model ID as key for NMS
                if mid not in bboxes:
                    bboxes[mid] = []
                    probs[mid] = []
                (x, y, w, h) = rois[image_id, bbox_id, :]
                try:
                    (tx, ty, tw, th) = prob_regr[image_id, bbox_id, 4*mid:4*(mid+1)]
                    tx /= self.cfg.classifier_regr_std[0]
                    ty /= self.cfg.classifier_regr_std[1]
                    tw /= self.cfg.classifier_regr_std[2]
                    th /= self.cfg.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                # Update result
                bboxes[mid].append([self.cfg.rpn_stride * x, self.cfg.rpn_stride * y,
                                    self.cfg.rpn_stride * (x + w), self.cfg.rpn_stride * (y + h)])
                probs[mid].append(prob_cls[image_id, bbox_id, mid])
            # Apply NMS based on concept type
            results = {'image_id': [], 'concept_id': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'conf': []}
            for key in bboxes:
                bbox = np.array(bboxes[key])
                new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                overlap_threshold=0.5, max_boxes=self.cfg.num_pr)
                # Generate final result
                for bbox_id in xrange(new_boxes.shape[0]):
                    x1, y1, x2, y2 = new_boxes[bbox_id, :]
                    real_x1, real_y1, real_x2, real_y2 = get_real_coordinates(scale_ratio[image_id], x1, y1, x2, y2)
                    results['image_id'].append(None)
                    results['concept_id'].append(self.cfg.concept_mapping[key])
                    results['x1'].append(real_x1)
                    results['y1'].append(real_y1)
                    results['x2'].append(real_x2)
                    results['y2'].append(real_y2)
                    results['conf'].append(new_probs[bbox_id])

            all_bboxes.append(results)

        return all_bboxes

    def rpn_to_roi(self, rpn_layer, regr_layer, overlap_threshold):
        # Normalization
        regr_layer /= self.cfg.std_scaling
        # Calculate ROIs
        (rows, cols, _) = rpn_layer.shape[1:]
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
        # TODO: Check the most efficient data type
        results = np.ndarray(shape=(0, self.cfg.num_pr, 4), dtype='int')
        # TODO: Process each image in parallel
        for curr_image in xrange(len(regr_layer)):
            curr_layer = 0
            for anchor_size in self.cfg.anchor_box_scales:
                for anchor_ratio in self.cfg.anchor_box_ratios:
                    anchor_x = (anchor_size * anchor_ratio[0])/self.cfg.rpn_stride
                    anchor_y = (anchor_size * anchor_ratio[1]) / self.cfg.rpn_stride
                    regr = regr_layer[curr_image, :, :, 4*curr_layer: 4*curr_layer+4]
                    regr = np.transpose(regr, (2, 0, 1))
                    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
                    A[0, :, :, curr_layer] = X - anchor_x / 2
                    A[1, :, :, curr_layer] = Y - anchor_y / 2
                    A[2, :, :, curr_layer] = anchor_x
                    A[3, :, :, curr_layer] = anchor_y
                    # Regression
                    A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

                    A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
                    A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
                    A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
                    A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

                    A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
                    A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
                    A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
                    A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

                    curr_layer += 1

            all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
            all_probs = rpn_layer[curr_image, :, :, :].transpose((2, 0, 1)).reshape((-1))

            x1 = all_boxes[:, 0]
            y1 = all_boxes[:, 1]
            x2 = all_boxes[:, 2]
            y2 = all_boxes[:, 3]

            idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

            all_boxes = np.delete(all_boxes, idxs, 0)
            all_probs = np.delete(all_probs, idxs, 0)

            result = non_max_suppression_fast(all_boxes, all_probs, overlap_threshold=overlap_threshold,
                                              max_boxes=self.cfg.num_pr)[0]

            # Pad if number of proposal regions is not sufficient
            if len(result) != self.cfg.num_pr:
                padded_result = np.ndarray(shape=(self.cfg.num_pr, 4))
                padded_result[:result.shape[0], :] = result
                padded_result[result.shape[0]:, :] = result[0, :]
                result = padded_result
            results = np.append(results, np.expand_dims(result, axis=0), axis=0)

        return results
