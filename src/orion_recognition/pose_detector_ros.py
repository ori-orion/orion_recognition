#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import pdb
import torch.nn as nn
import torch.utils.data
import numpy as np

#import matplotlib.pyplot as plt
import cv2
import os
import sys
from tqdm import tqdm
import time
from yolo.darknet import Darknet
from yolo.util import dynamic_write_results
from PIL import Image
import torch.utils.data as data
from SPPE.src.models.FastPose import createModel
import torch.nn as nn
import torch
from SPPE.src.utils.img import im_to_torch,cropBox
from SPPE.src.utils.img import findPeak, processPeaks
from matching import candidate_reselect as matching
import math

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = 320
        self.inputResW = 256
        self.outputResH = 80
        self.outputResW = 64
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet_fast, self).__init__()

        model = createModel().cuda()
        print('Loading pose model from {}'.format('/home/ori/code/recognition/pose_estimation/models/sppe/duc_se.pth'))
        model.load_state_dict(torch.load('/home/ori/code/recognition/pose_estimation/models/sppe/duc_se.pth'))
        model.eval()
        self.pyranet = model
        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out

class PoseDetector(object):
    def __init__(self):
        self.det_model = Darknet("/home/ori/code/recognition/pose_estimation/yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('/home/ori/code/recognition/pose_estimation/models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = 608
        self.det_inp_dim = int(self.det_model.net_info['height'])
        self.det_model.cuda()
        self.det_model.eval()
        self.pose_dataset = Mscoco()
        self.pose_model = InferenNet_fast(4 * 1 + 1, self.pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()
        self.semantic_list = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'Neck']
       
    def detect(self, orig_im):
        output_dict={}
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (self.letterbox_image(orig_im, (608, 608)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        with torch.no_grad():
            img_cu = img_.cuda()
            prediction = self.det_model(img_cu, CUDA=True)
            dets = dynamic_write_results(prediction, 0.05,80, nms=True, nms_conf=0.6)
            if isinstance(dets, int):
                return dets,dets
            im_dim_list=[dim]
            im_dim_list=torch.FloatTensor(im_dim_list).repeat(1,2)
            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
            scaling_factor = torch.min(608 / im_dim_list, 1)[0].view(-1, 1)
            dets[:, [1, 3]] -= (608 - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (608 - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            boxes_k = boxes[dets[:,0]==0]
            inps = torch.zeros(boxes_k.size(0), 3, 320, 256)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)
            scores_k=scores[dets[:,0]==0]
            inp = im_to_torch(cv2.cvtColor(orig_im, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = self.crop_from_dets(inp, boxes_k, inps, pt1, pt2)
            hm_j = self.pose_model(inps.cuda())
            hm = hm_j.cpu()
            preds = self.getMultiPeakPrediction(hm, pt1.numpy(), pt2.numpy(), 320, 256, 80, 64)
            result = matching(boxes_k, scores_k.numpy(), preds)
            result = {'imgname': "ori_robot",'result': result}
            img = self.vis_frame(orig_im, result)
            #publish human pose information
            for i,human in enumerate(result['result']):
                #print(result['result'])
                upLeft=pt1[i] 
                bottomRight=pt2[i]
                kp_preds = human['keypoints']
                kp_scores = human['kp_score']
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
                kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
                # Draw keypoints
                key_points = {}
                for n in range(kp_scores.shape[0]):
                    if kp_scores[n] <= 0.05:
                        cor_x, cor_y = (-1, -1)
                    cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                    key_points[self.semantic_list[n]]=(cor_x, cor_y)
                output_dict[str(i)] = (key_points, upLeft, bottomRight)
            print('{} humans detected.'.format(len(result['result'])))
        return img, output_dict

    def letterbox_image(self, img, inp_dim):
        '''resize image with unchanged aspect ratio using padding'''
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
        return canvas
   
    def crop_from_dets(self,img, boxes, inps, pt1, pt2):
        imght = img.size(1)
        imgwidth = img.size(2)
        tmp_img = img
        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)
        for i, box in enumerate(boxes):
            upLeft = torch.Tensor(
                (float(box[0]), float(box[1])))
            bottomRight = torch.Tensor(
                (float(box[2]), float(box[3])))

            ht = bottomRight[1] - upLeft[1]
            width = bottomRight[0] - upLeft[0]

            scaleRate = 0.3

            upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
            upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
            bottomRight[0] = max(
                min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
            bottomRight[1] = max(
                min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

            try:
                inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, 320, 256)
            except IndexError:
                print(tmp_img.shape)
                print(upLeft)
                print(bottomRight)
                print('===')
            pt1[i] = upLeft
            pt2[i] = bottomRight

        return inps, pt1, pt2
     
    def getMultiPeakPrediction(self, hms, pt1, pt2, inpH, inpW, resH, resW):

        assert hms.dim() == 4, 'Score maps should be 4-dim'

        preds_img = {}
        hms = hms.numpy()
        for n in range(hms.shape[0]):        # Number of samples
            preds_img[n] = {}           # Result of sample: n
            for k in range(hms.shape[1]):    # Number of keypoints
                preds_img[n][k] = []    # Result of keypoint: k
                hm = hms[n][k]

                candidate_points = findPeak(hm)

                res_pt = processPeaks(candidate_points, hm,
                                      pt1[n], pt2[n], inpH, inpW, resH, resW)

                preds_img[n][k] = res_pt

        return preds_img
    
    def vis_frame(self, frame, im_res, format='coco'):
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]

            p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                        (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                        (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                        (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                        (77,222,255), (255,156,127), 
                        (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
            line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError

        im_name = im_res['imgname'].split('/')[-1]
        img = frame
        #pdb.set_trace()
        height,width = img.shape[:2]
        img = cv2.resize(img,(int(width/2), int(height/2)))
        for human in im_res['result']:
            part_line = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
            # Draw keypoints
            #print('LShoulder:',int(kp_preds[5, 0]), int(kp_preds[5, 1]))
            #print('RShoulder:',int(kp_preds[6, 0]), int(kp_preds[6, 1]))
            #print('LHip:',int(kp_preds[11, 0]), int(kp_preds[11, 1]))
            #print('Neck:',int(kp_preds[-1, 0]), int(kp_preds[-1, 1]))
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= 0.05:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_line[n] = (int(cor_x/2), int(cor_y/2))
                bg = img.copy()
                cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 2, p_color[n], -1)
                # Now create a mask of logo and create its inverse mask also
                transparency = max(0, min(1, kp_scores[n]))
                img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    bg = img.copy()

                    X = (start_xy[0], end_xy[0])
                    Y = (start_xy[1], end_xy[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                    polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(bg, polygon, line_color[i])
                    #cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                    transparency = max(0, min(1, 0.5*(kp_scores[start_p] + kp_scores[end_p])))
                    img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
        return img
