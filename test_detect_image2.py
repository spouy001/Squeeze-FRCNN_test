import time
import os
import cv2
import numpy as np
from test_config import TestConfig
from test_model import SqueezeModel
from imagenet_det_parser import get_training_images
from sklearn.metrics import average_precision_score

def initialization():
    _cfg = TestConfig()
    _model = SqueezeModel(_cfg)
    return _cfg, _model
def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union
def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h
def isNaN(num):
    return num != num
def cal_iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)

def get_map(pred, gt, f):
	T = {}
	P = {}
	fx, fy = (1,1)

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
                print('pred_box:', pred_box)
		pred_class_list = pred_box['class']
                pred_class = pred_class_list['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False

		for gt_box in gt:
                        print('gt_box:', gt_box)
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx
			gt_x2 = gt_box['x2']/fx
			gt_y1 = gt_box['y1']/fy
			gt_y2 = gt_box['y2']/fy
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou = cal_iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                        print('IOU:', iou)
			if iou >= 0.5:
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:
		if not gt_box['bbox_matched']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()
	print('T=', T)
	print('P=', P)
	return T, P



# Main Entrance
cfg, model = initialization()
(val_imgs, classes_count, class_mapping) = get_training_images(cfg.test_image_path)
#data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')
print('Num val samples {}'.format(len(val_imgs)))
time_proc_start = time.time()
all_bboxes = []  # store all the results for MAP
#image_list = os.listdir(cfg.test_image_path)
batch = []
scale_ratio = []
time_batch_start = time.time()
T = {}
P = {}
for idx, image_name in enumerate(val_imgs):
    # Verify file type
    #if not image_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        #continue

    # Generate Batch
    #image_path = os.path.join(cfg.test_image_path, image_name)
    image_path = image_name['filepath']
    print("Image path: '{}'".format(image_path))
    img, img_scale_ratio = cfg.preprocess_image(cv2.imread(image_path))
    batch.append(img)
    scale_ratio.append(img_scale_ratio)
    #print("Detecting Image '{}'".format(image_name))

    # Process Batch
    if len(batch) == cfg.batch_size:
        # Finalize Batch
        batch = np.asarray(batch)
        print("Time Load Batch: {}".format(time.time() - time_batch_start))
        # RPN
        time_rpn_start = time.time()
        rpn_layer, regr_layer, features = model.apply_rpn_model(batch)
        print("Time RPN Model: {}".format(time.time() - time_rpn_start))
        time_rpn2roi_start = time.time()
        rois = model.post_rpn_model(rpn_layer, regr_layer)
        print("Time Generate ROIs: {}".format(time.time() - time_rpn2roi_start))
        # SSP
        time_ssp_start = time.time()
        ssp_rois = model.apply_ssp_model(features, rois)
        print("Time SSP Model: {}".format(time.time() - time_ssp_start))
        # Classification
        time_cls_start = time.time()
        prob_cls, prob_regr = model.apply_cls_model(ssp_rois)
        print("Time CLS Model: {}".format(time.time() - time_cls_start))
        time_regr_start = time.time()
        bboxes = model.post_cls_model(rois, prob_cls, prob_regr, scale_ratio)
        print("Time Generate Results: {}".format(time.time() - time_regr_start))
        print('All Bounding box: ', bboxes)
        #calculate MAP
        for idb, bbox in enumerate(bboxes):
            #bbox = np.array(bboxes[key])
            print('One Bounding box: ', bbox)
            all_dets = []
            print('Length of bbox list:', len(bbox['x1']))
            for index in range(len(bbox['x1'])):
                det = {'x1': bbox['x1'][index],'x2': bbox['x2'][index], 'y1': bbox['y1'][index], 'y2': bbox['y2'][index], 'class': bbox['concept_id'][index], 'prob': bbox['conf'][index] }
                all_dets.append(det)
            t, p = get_map(all_dets, image_name['bboxes'], img_scale_ratio)
            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])
            all_aps = []
            for key in T.keys():
                ap = average_precision_score(T[key], P[key])
                print('{} AP: {}'.format(key, ap))
		if isNaN(ap):
			all_aps.append(0)
		else:
                	all_aps.append(ap)
	    		
            print('mAP = {}'.format(np.mean(np.array(all_aps))))
            #nans=np.isnan(all_aps)
            #all_aps[nans] = 0
            #print('mAP2 = {}'.format(np.mean(np.array(all_aps))))				
        all_bboxes.extend(bboxes)
        # TODO: Upload Results





        batch = []
        scale_ratio = []
        print("Total Time Batch: {}".format(time.time() - time_batch_start))
        print("Average Time Batch: {}".format((time.time() - time_batch_start)/cfg.batch_size))
        time_batch_start = time.time()

print("Total Time: {}".format(time.time() - time_proc_start))

if len(batch) != 0:
    results = model.apply_rpn_model(batch)
    # TODO: Upload Results



