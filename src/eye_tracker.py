import math
import os

import numpy as np
import torch
import cv2

import models.MaskRCNN as MRCNN
from src.geometry import batch_rodrigues


class EyeTracker():
    def __init__(self, model_path, shape: (int, int)=(512,512), device='cuda'):
        self.shape = shape
        self.height = shape[0]
        self.width = shape[1]
        self.focal = 3940

        self.model_path = model_path
        self.device = device

        Rx_90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        Ry_90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        Rz_90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

        self.default2standard = Ry_90 @ (Rz_90 @ Rz_90 @ Rz_90).T  # transform fish mesh from default position to standard position (parallel to screen plane)
        reshape_4x4 = torch.zeros([4,4])
        reshape_4x4[-1,-1] = 1
        reshape_4x4[:3,:3] = self.default2standard
        self.default2standard = reshape_4x4.cpu().numpy()

    def detect_blob(self, params, img):
        detector = cv2.SimpleBlobDetector_create(params)
        eye_to_detect = img.astype(np.uint8)
        keypoints = detector.detect(eye_to_detect)

        im_with_keypoints = cv2.drawKeypoints(eye_to_detect, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return im_with_keypoints, keypoints

    def img_enhance(self, img, contrast=1.5, brightness=0):
        # contrast and brightness
        new_img = cv2.addWeighted(img, contrast, img, 0, brightness)

        return new_img

    def eye_img_pyramid(self, eye_cropped, mask_cropped, start_size=512, end_size=128):
        img_size = start_size
        contrast_list = [0.5, 0.7, 1.5, 1.7, 2.0, 2.3, 2.5, 2.7]
        contrast_list = contrast_list[::-1]

        while img_size >= end_size:
            # blob detection params
            scale = img_size / 2  # 4 * 10
            params_pupil = cv2.SimpleBlobDetector_Params()

            params_pupil.minThreshold = 10
            params_pupil.maxThreshold = 100
            params_pupil.blobColor = 0
            params_pupil.filterByArea = True
            params_pupil.maxArea = int(scale ** 2)  # filter abnormal detection based on size
            params_pupil.minArea = int((scale / 4) ** 2)
            params_pupil.filterByInertia = True
            params_pupil.minInertiaRatio = 0.5  # 0.5
            params_pupil.filterByCircularity = True
            params_pupil.minCircularity = 0.8  # 0.8

            params_sclera = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params_sclera.minThreshold = 150
            params_sclera.maxThreshold = 250
            params_sclera.blobColor = 255
            params_sclera.maxArea = int((scale * 3) ** 2)  # 20000 * (scale ** 2)  # change for different img size
            params_sclera.filterByInertia = True
            params_sclera.minInertiaRatio = 0.5
            params_sclera.filterByCircularity = True
            params_sclera.minCircularity = 0.8

            blob_detected = False
            eye_cropped = cv2.resize(eye_cropped, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            mask_cropped = cv2.resize(mask_cropped, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

            for contrast in contrast_list:
                eye_cropped_enhance = self.img_enhance(eye_cropped, contrast)
                pupil_blob, pupil_keypoint = self.detect_blob(params_pupil, eye_cropped_enhance)
                if len(pupil_keypoint) > 0:
                    blob_detected = True
                    break

            if blob_detected:
                eye_blob, eye_keypoint = self.detect_blob(params_sclera, mask_cropped)

                return eye_cropped, mask_cropped, pupil_blob, pupil_keypoint, eye_blob, eye_keypoint

            img_size = int(img_size / 2)

        raise Exception(f"no blob detected for input image")

    def find_eye_pos_2(self, image, body_pose, bbox, kpt, threshold=90, eval=False, base_dir='../data/output/eye_detect_test/debug/', scale=5):
        # perform eye area detection using mrcnn
        num_classes = 2
        model = MRCNN.get_model_instance_segmentation(num_classes)
        model.to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        if not eval:
            # image shape: (c, H, W)
            padded_img, padded_kpt = self.get_cropped_kpt(image, bbox, kpt)
            image = self.revert_pose(body_pose, padded_kpt, padded_img).astype(np.float32)  # reverted fish
            cv2.imwrite(os.path.join(base_dir, 'reverted.png'), image)

        image = image.astype(np.float32)
        predictions = model(
            [torch.from_numpy(image / 255.).to(self.device).permute(2, 0, 1)])  # image[:,:int(0.125 * image.shape[1])]

        for i in range(predictions[0]['boxes'].size()[0]):
            # take the first result as we only have one instance in the image
            if i > 0:
                break

            # find bounding box of mask
            pos = np.where(predictions[0]['masks'][i, 0].detach().cpu().numpy())
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bounding_box = np.array([xmin, ymin, xmax, ymax])

            # bounding_box = predictions[0]['boxes'][i].data.int().cpu().numpy()  # [w_low, h_low, w_up, h_up]
            eye_area = image[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10,
                       int(bounding_box[0]) - 10:int(bounding_box[2]) + 10]
            cv2.imwrite(os.path.join(base_dir, 'cropped_eye.png'), eye_area)

            mask = predictions[0]['masks'][i, 0].mul(255)
            cropped_mask = mask[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10,
                           int(bounding_box[0]) - 10:int(bounding_box[2]) + 10]

            cropped_mask = cropped_mask.byte().cpu().numpy()
            cv2.imwrite(os.path.join(base_dir, 'cropped_mask.png'), cropped_mask)

        eye_area, cropped_mask, pupil_blob, pupil_keypoint, eye_blob, eye_keypoint = self.eye_img_pyramid(eye_area, cropped_mask)

        pupil_circle = [pupil_keypoint[0].pt, pupil_keypoint[0].size]

        eye_circle = [eye_keypoint[0].pt, eye_keypoint[0].size]

        eye_with_keypoints = cv2.drawKeypoints(eye_area.astype(np.uint8), eye_keypoint, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        eye_with_keypoints = cv2.drawKeypoints(eye_with_keypoints.astype(np.uint8), (pupil_keypoint[0],), np.array([]),
                                               (255, 0, 0),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        r = (eye_keypoint[0].size / 2)
        pad = 1
        eye_image = eye_with_keypoints
        cv2.imwrite(os.path.join(base_dir, 'central_eye.png'), eye_image)

        # information
        normalized_pupil_dir = np.array(
            [pupil_circle[0][0] - eye_circle[0][0], -(pupil_circle[0][1] - eye_circle[0][1])]) \
                               / (eye_circle[1] / 2)  # radius

        eye_info = {}
        eye_info['pupil_size'] = pupil_keypoint[0].size
        eye_info['eye_size'] = eye_keypoint[0].size
        eye_info['vector'] = normalized_pupil_dir
        eye_info['dist'] = np.linalg.norm(normalized_pupil_dir)
        eye_info['angle'] = np.arctan2(normalized_pupil_dir[0], normalized_pupil_dir[1]) / np.pi * 180
        eye_info['eye_position'] = (int(bounding_box[0]) - 10 + eye_keypoint[0].pt[0],
                                    int(bounding_box[1]) - 10 + eye_keypoint[0].pt[1],)
        eye_info['pupil_position'] = (int(bounding_box[0]) - 10 + pupil_keypoint[0].pt[0],
                                      int(bounding_box[1]) - 10 + pupil_keypoint[0].pt[1])

        # eye_info['eye_position'] =

        # draw eye plot
        scaling = 3

        blank = np.ones((eye_area.shape[0] * scaling, eye_area.shape[1] * scaling, 3)) * 255
        middle = (int(eye_area.shape[0] * scaling / 2), int(eye_area.shape[1] * scaling / 2))
        cv2.circle(blank, middle, int(eye_circle[1] * scaling / 2), (0, 0, 255))

        cv2.circle(blank, (middle[0] + int(scaling * (pupil_circle[0][0] - eye_circle[0][0])),
                           middle[1] + int(scaling * (pupil_circle[0][1] - eye_circle[0][1]))),
                   int(pupil_circle[1] * scaling / 2), (255, 0, 0))

        cv2.imwrite(os.path.join(base_dir, 'eye_absolute.png'), blank)

        eye_image = cv2.resize(eye_image, (130, 130))

        image[int(bounding_box[1]) - 10, int(bounding_box[0]) - 10:int(bounding_box[2]) + 10] = np.array([255, 255, 0])
        image[int(bounding_box[3]) + 10, int(bounding_box[0]) - 10:int(bounding_box[2]) + 10] = np.array([255, 255, 0])
        image[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10, int(bounding_box[2]) + 10] = np.array([255, 255, 0])
        image[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10, int(bounding_box[0]) - 10] = np.array([255, 255, 0])

        return image, eye_image, eye_area, eye_info

    def find_eye_pos(self, image, body_pose, bbox, kpt, threshold=90, eval=False, base_dir='../data/output/eye_detect_test/debug/', scale=1):
        # perform eye area detection using mrcnn
        num_classes = 2
        model = MRCNN.get_model_instance_segmentation(num_classes)
        model.to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        if not eval:
            # image shape: (c, H, W)
            padded_img, padded_kpt = self.get_cropped_kpt(image, bbox, kpt)
            image = self.revert_pose(body_pose, padded_kpt, padded_img).astype(np.float32)  # reverted fish
            cv2.imwrite(os.path.join(base_dir, 'reverted.png'), image)
            #images = list(torch.unsqueeze(img.to(device), 2).permute(2, 0, 1) for img in images[0])

        image = image.astype(np.float32)
        predictions = model([torch.from_numpy(image / 255.).to(self.device).permute(2, 0, 1)]) # image[:,:int(0.125 * image.shape[1])]

        for i in range(predictions[0]['boxes'].size()[0]):
            # take the first result as we only have one instance in the image
            if i > 0:
                break

            # find bounding box of mask
            pos = np.where(predictions[0]['masks'][i, 0].detach().cpu().numpy())
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bounding_box = np.array([xmin, ymin, xmax, ymax])

            #bounding_box = predictions[0]['boxes'][i].data.int().cpu().numpy()  # [w_low, h_low, w_up, h_up]
            eye_area = image[int(bounding_box[1]) - 10 :int(bounding_box[3]) + 10,
                      int(bounding_box[0]) - 10:int(bounding_box[2]) + 10]
            cv2.imwrite(os.path.join(base_dir, 'cropped_eye.png'), eye_area)

            mask = predictions[0]['masks'][i, 0].mul(255)
            cropped_mask = mask[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10,
                           int(bounding_box[0]) - 10:int(bounding_box[2]) + 10]

            cropped_mask = cropped_mask.byte().cpu().numpy()
            cv2.imwrite(os.path.join(base_dir, 'cropped_mask.png'), cropped_mask)

        # pupil detection
        params2 = cv2.SimpleBlobDetector_Params()

        scale = 4

        # Change blob thresholds
        params2.minThreshold = 10
        params2.maxThreshold = 100
        params2.blobColor = 0
        params2.filterByArea = True
        params2.maxArea = 2000 * (scale ** 2) # change for different img size
        params2.filterByInertia = True
        params2.minInertiaRatio = 0.5
        params2.filterByCircularity = True
        params2.minCircularity = 0.8

        pupil_blob, pupil_keypoint = self.detect_blob(params2, eye_area)
        pupil_circle = [pupil_keypoint[0].pt, pupil_keypoint[0].size]
        cv2.imwrite(os.path.join(base_dir, 'eye_pupil.png'), pupil_blob)

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 150
        params.maxThreshold = 250
        params.blobColor = 255
        params.maxArea = 20000 * (scale ** 2)  # change for different img size
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        params.filterByCircularity = True
        params.minCircularity = 0.8

        eye_blob, eye_keypoint = self.detect_blob(params, cropped_mask)
        cv2.imwrite(os.path.join(base_dir, 'eye_area.png'), eye_blob)

        eye_circle = [eye_keypoint[0].pt, eye_keypoint[0].size]
        cv2.imwrite(os.path.join(base_dir, 'eye_area.png'), eye_blob)

        eye_with_keypoints = cv2.drawKeypoints(eye_area.astype(np.uint8), eye_keypoint, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        eye_with_keypoints = cv2.drawKeypoints(eye_with_keypoints.astype(np.uint8), (pupil_keypoint[0],), np.array([]), (255, 0, 0),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        r = (eye_keypoint[0].size / 2)
        pad = 1
        eye_image = eye_with_keypoints
        cv2.imwrite(os.path.join(base_dir, 'central_eye.png'), eye_image)

        # information
        normalized_pupil_dir = np.array([pupil_circle[0][0] - eye_circle[0][0], -(pupil_circle[0][1] - eye_circle[0][1])]) \
                               / (eye_circle[1] / 2)  # radius
        eye_info = {}
        eye_info['pupil_size'] = pupil_keypoint[0].size
        eye_info['eye_size'] = eye_keypoint[0].size
        eye_info['vector'] = normalized_pupil_dir
        eye_info['dist'] = np.linalg.norm(normalized_pupil_dir)
        eye_info['angle'] = np.arctan2(normalized_pupil_dir[0], normalized_pupil_dir[1]) / np.pi * 180
        eye_info['eye_position'] = (int(bounding_box[0]) - 10 + eye_keypoint[0].pt[0],
                                    int(bounding_box[1]) - 10 + eye_keypoint[0].pt[1],)
        eye_info['pupil_position'] = (int(bounding_box[0]) - 10 + pupil_keypoint[0].pt[0],
                                      int(bounding_box[1]) - 10 + pupil_keypoint[0].pt[1])

        # eye_info['eye_position'] =

        # draw eye plot
        scaling = 3

        blank = np.ones((eye_area.shape[0] * scaling, eye_area.shape[1] * scaling, 3)) * 255
        middle = (int(eye_area.shape[0] * scaling / 2), int(eye_area.shape[1] * scaling / 2))
        cv2.circle(blank, middle, int(eye_circle[1] * scaling / 2), (0, 0, 255))

        cv2.circle(blank, (middle[0] + int(scaling * (pupil_circle[0][0] - eye_circle[0][0])),
                           middle[1] + int(scaling * (pupil_circle[0][1] - eye_circle[0][1]))), int(pupil_circle[1] * scaling / 2), (255, 0, 0))

        cv2.imwrite(os.path.join(base_dir, 'eye_absolute.png'), blank)
        eye_image = cv2.resize(eye_image, (130, 130))

        image[int(bounding_box[1]) - 10, int(bounding_box[0]) - 10:int(bounding_box[2]) + 10] = np.array([255, 255, 0])
        image[int(bounding_box[3]) + 10, int(bounding_box[0]) - 10:int(bounding_box[2]) + 10] = np.array([255, 255, 0])
        image[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10, int(bounding_box[2]) + 10] = np.array([255, 255, 0])
        image[int(bounding_box[1]) - 10:int(bounding_box[3]) + 10, int(bounding_box[0]) - 10] = np.array([255, 255, 0])

        return image, eye_image, eye_area, eye_info

    def detect_blob(self, params, img):
        detector = cv2.SimpleBlobDetector_create(params)
        #eye_to_detect = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        eye_to_detect = img.astype(np.uint8)
        keypoints = detector.detect(eye_to_detect)

        im_with_keypoints = cv2.drawKeypoints(eye_to_detect, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #cv2.imwrite('data/output/eye_detect_test/debug/eye_pupil.png', im_with_keypoints)

        return im_with_keypoints, keypoints

    def revert_pose_kpt(self, kpt_padded, img):
        w = img.shape[1]
        h = img.shape[0]

        pts1 = np.float32(kpt_padded.detach().cpu().numpy()[[0,1,2,4]])
        pts2 = np.float32([[3,165], [85,168], [264,170], [141,216]])

        M = cv2.getPerspectiveTransform(pts1, pts2)  # map from transformed position to satandard position
        reverted = cv2.warpPerspective(img, M, (h, w))

        reverted = cv2.resize(reverted, self.shape)
        return reverted

    def revert_pose(self, body_pose, kpt, img):
        pose_mat = batch_rodrigues(body_pose.view(-1, 3))

        target_kpt = np.array([[ 729.9990,  584.6780],
                                [ 809.9070,  585.5550],
                                [1000.6560,  559.4640],
                                [1061.4780,  602.5130],
                                [ 886.9320,  612.2850],
                                [1040.3100,  548.0620]])

        w = img.shape[1]
        h = img.shape[0]
        f = self.focal

        dx = 0
        dy = 0
        dz = f

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2],
                           [0, 1, -h / 2],
                           [0, 0, 1],
                           [0, 0, 1]])

        # Translation matrix
        T = np.array([[1, 0, 0, dx],
                          [0, 1, 0, dy],
                          [0, 0, 1, dz],
                          [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0],
                           [0, f, h / 2, 0],
                           [0, 0, 1, 0]])

        rot = np.zeros([4,4])
        rot[-1,-1] = 1
        rot[:3,:3] = pose_mat[0].detach().cpu().numpy()  # global pose

        rot_head = np.zeros([4, 4])
        rot_head[-1, -1] = 1
        rot_head[:3, :3] = pose_mat[1].detach().cpu().numpy()  # head pose

        #revert_mat = self.default2standard @ torch.inverse(rot) @ torch.inverse(rot_head)  # convert from fitted pose to standard position
        forward_mat = rot @ np.linalg.inv(self.default2standard)  #rot_head @ rot @ np.linalg.inv(self.default2standard)  # from standard position to fitted pose (inverse of revert_mat)

        trans_mat = A2 @ T @ forward_mat @ A1

        # calculate corner mapping
        center = np.float32([h/2, w/2, 0])

        v1 = [-h/2, -w/2, 1] + center
        v1r = trans_mat @ v1
        v1r = v1r / v1r[2]

        v2 = [-h/2, w/2, 1] + center
        v2r = trans_mat @ v2
        v2r = v2r / v2r[2]

        v3 = [h/2, -w/2, 1] + center
        v3r = trans_mat @ v3
        v3r = v3r / v3r[2]

        v4 = [h/2, w/2, 1] + center
        v4r = trans_mat @ v4
        v4r = v4r / v4r[2]

        pts2 = np.float32([v1r[0:2], v2r[0:2], v3r[0:2], v4r[0:2]])
        pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])

        # only kpt mapping
        # pts1 = kpt[[0,1,2,4]].numpy()[:,[1,0]]
        # pts1 = pts1 - pts1[[0]] + np.array([[100, 70]])
        # pts1 = pts1.astype(np.float32)
        # pts2 = target_kpt[[0,1,2,4]][:,[1,0]]
        # pts2 = pts2 - pts2[[0]] + np.array([[100, 70]])
        # pts2 = pts2.astype(np.float32)

        M = cv2.getPerspectiveTransform(pts1, pts2)  # map from transformed position to satandard position
        reverted = cv2.warpPerspective(img, M, (h, w))
        # kpt_0 = M @ np.array([[kpt[1, 0]], [kpt[1, 1]], [1]])
        # kpt_0 = kpt_0 / kpt_0[2]
        #
        # kpt_1 = M @ np.array([[kpt[2, 0]], [kpt[2, 1]], [1]])
        # kpt_1 = kpt_1 / kpt_1[2]
        #
        # angle = np.arctan(-(kpt_1[1] - kpt_0[1]) / (kpt_1[0] - kpt_0[0])) / np.pi * 180
        # rot_mat = cv2.getRotationMatrix2D((h/2, w/2), -angle[0], 1.0)
        # result = cv2.warpAffine(reverted, rot_mat, reverted.shape[1::-1], flags=cv2.INTER_LINEAR)
        result = reverted

        result = cv2.resize(result, self.shape)

        return result

    def get_cropped_kpt(self, image, bbox, kpt):
        cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # origin_size = np.max([bbox[3] - bbox[1], bbox[2] - bbox[0]])

        # equalize hist
        # img_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
        #
        # # equalize the histogram of the Y channel
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        #
        # # convert the YUV image back to RGB format
        # cropped = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        kpt_cropped = kpt - bbox[:2]

        (a, b, d) = cropped.shape
        if a > b:
            padding = ((0, 0), (math.floor((a-b)/2.), math.ceil((a-b)/2.)), (0, 0))
        else:
            padding = ((math.floor((b-a)/2.), math.ceil((b-a)/2.)), (0, 0), (0, 0))
        padded = np.pad(cropped, padding, mode='constant', constant_values=0)

        kpt_padded = kpt_cropped + torch.tensor([padding[1][0], padding[0][0]])

        return padded, kpt_padded

    def get_cropped(self, image, bbox):
        cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # origin_size = np.max([bbox[3] - bbox[1], bbox[2] - bbox[0]])

        (a, b, d) = cropped.shape
        if a > b:
            padding = ((0, 0), (math.floor((a-b)/2.), math.ceil((a-b)/2.)), (0, 0))
        else:
            padding = ((math.floor((b-a)/2.), math.ceil((b-a)/2.)), (0, 0), (0, 0))
        padded = np.pad(cropped, padding, mode='constant', constant_values=0)

        return cv2.resize(padded, self.shape)
