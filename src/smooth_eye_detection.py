import argparse
import csv
import os
import pickle

import cv2
import numpy as np
import torch

from src.dataloaders import Multiview_Dataset
from src.eye_tracker import EyeTracker
from src.interpolate_pose import denoise_sequence
from tqdm import trange
import matplotlib.pyplot as plt


def output_reverted(eye_model, image, body_pose, bbox, keypoints):
    et = EyeTracker(eye_model, (512, 512))
    padded_img, padded_kpt = et.get_cropped_kpt(image, bbox, keypoints)
    image = et.revert_pose(body_pose, padded_kpt, padded_img)

    return image

def detect_eye(args):
    multiview_data = Multiview_Dataset(root=args.datadir)

    fish_place = args.fish_place

    pose_dic = pickle.load(open(os.path.join(args.dir, f"pose_result_{args.index_range}_({args.fish_place}).pickle"), 'rb'))
    individual_fit_parameters = pose_dic['individual_fit_parameters']
    indices = pose_dic['indices']
    # image_data = pose_dic['sample_data']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sequence = []

    for i in range(10):
        sequence.append(individual_fit_parameters[0].unsqueeze(0))

    for i in range(len(indices)):
        sequence.append(individual_fit_parameters[4 * i].unsqueeze(0))

    for i in range(10):
        sequence.append(individual_fit_parameters[-4].unsqueeze(0))

    sequence = torch.cat(sequence).to(device)  # [10:-10]
    sequence = denoise_sequence(args, sequence, device)[10:-10]

    # create output folder
    # if not os.path.exists(args.dir + '/images_eye/eye_frames'):
    os.makedirs(os.path.join(args.dir, f'../images_eye/fish_{args.fish_place}/eye_frames'), exist_ok=True)
    os.makedirs(os.path.join(args.dir, f'../eye_poses_csv/fish_{args.fish_place}'), exist_ok=True)

    eye_info_list = []
    pbar = trange(len(indices) - 1, desc="outputting frames")
    # for i in range(15,len(indices)):
    i = 0
    for i in indices:
        # if not db_util.load_reconstruct_record(col_name, fish_place, i):
        #     continue

        try:
            sample_data = multiview_data[i]
            if not sample_data['full_kpts']:
                raise Exception('no enough keypoints')
        except:
            print(f'frame {i} data load failed')
            if i - indices[0] == 0:
                eye_info_list.append({
                    'pupil_size': 'none',
                    'eye_size': 'none',
                    'vector': 'none',
                    'dist': 'none',
                    'angle': 'none',
                })
            else:
                eye_info_list.append(eye_info_list[-1])
            continue

        # if i < 0:
        #     continue
        # image = plt.imread('../' + image_data[i][1][0]) * 255
        image = cv2.imread(sample_data['imgpaths'][0], 1)
        #image = cv2.imread(image_data[i][1][0], 1)
        if fish_place == 2:
            bbox = sample_data["bboxes2"][0]
        else:
            bbox = sample_data["bboxes"][0]
        bbox[0] = bbox[0] - 40

        if args.fish_place == 1:
            keypoints = sample_data["keypoints"][0][:, :2]
        elif args.fish_place == 2:
            keypoints = sample_data["keypoints2"][0][:, :2]

        # keypoints = image_data[i][2][0][:, :2]
        # keypoints = torch.cat([keypoints, multiview_data[image_data[i][-1]]["detected_eyes"][[1],:2]])  # select the correct instance (0 left 1 right)

        # todo load last keypoint in loader
        for j in range(keypoints.size(0)):
            image[int(keypoints[j, 1]), int(keypoints[j, 0])] = np.array([255, 0, 0])


        et = EyeTracker(args.eye_model, (512, 512))
        # try:
        reverted_fish, eye_image, eye_area, eye_info = et.find_eye_pos_2(image, sequence[i - indices[0]], bbox, keypoints)
        # except:
        #     print(f'frame {i} detection failed')
        #     if i - indices[0] == 0:
        #         eye_info_list.append({
        #             'pupil_size': 'none',
        #             'eye_size': 'none',
        #             'vector': 'none',
        #             'dist': 'none',
        #             'angle': 'none',
        #         })
        #     else:
        #         eye_info_list.append(eye_info_list[-1])
        #     continue
        # # reverted_fish = reverted_fish.astype(np.uint8)

        # put detected frame to original image
        image[-512:, 100:612] = reverted_fish
        image[-512, 100:612] = np.array([0, 255, 0])
        image[-1, 100:612] = np.array([0, 255, 0])
        image[-512:, 612] = np.array([0, 255, 0])
        image[-512:, 100] = np.array([0, 255, 0])

        image[bbox[1], bbox[0]:bbox[2]] = np.array([0, 255, 0])
        image[bbox[3], bbox[0]:bbox[2]] = np.array([0, 255, 0])
        image[bbox[1]:bbox[3], bbox[0]] = np.array([0, 255, 0])
        image[bbox[1]:bbox[3], bbox[2]] = np.array([0, 255, 0])

        plt.imsave(os.path.join(args.dir, f'../images_eye/fish_{args.fish_place}/{i}_reverted.png'),
                   reverted_fish.astype(np.uint8))
        plt.imsave(os.path.join(args.dir, f'../images_eye/fish_{args.fish_place}/eye_frames/frame_{i}.png'),
                   eye_image.astype(np.uint8))
        eye_info_list.append(eye_info)

        # output image for training
        # image = output_reverted(args.eye_model, image, sequence[i], bbox, keypoints)
        # plt.imsave(args.dir + '/images_eye/training_set/{}_reverted.png'.format(image_data[i][-1]),
        #            image.astype(np.uint8))

        pbar.update(1)
        i += 1

    # plot result
    xs = np.array(range(len(eye_info_list)))
    ys = np.array([e['angle'] for e in eye_info_list])
    ys2 = np.array([e['dist'] for e in eye_info_list])

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), gridspec_kw={'height_ratios': [1, 1, 3]})
    axs[0].plot(xs, ys)
    axs[0].set_title('angle-time')
    axs[1].plot(xs, ys2)
    axs[1].set_title('dist-time')

    # heat map
    # vecs = np.array([e['vector'] for e in eye_info_list])
    # vecs = np.int8(vecs * 150) + np.array([49, 49])
    # # data = np.zeros((100, 100))
    # # for v in vecs:
    # #     data[v[1]][v[0]] += 1
    # #
    # # data /= np.max(data)
    # # axs[2].pcolormesh(data, cmap='summer')
    # alphas = np.linspace(0.3, 1.0, num=vecs.shape[0] - 1)
    # rs = np.linspace(0.1, 1.0, num=vecs.shape[0] - 1)
    # gs = np.linspace(0.1, 0.6, num=vecs.shape[0] - 1)
    #
    # for i in range(vecs.shape[0]):
    #     if i == 0:
    #         continue
    #     start = vecs[i - 1]
    #     end = vecs[i]
    #
    #     axs[2].plot([start[0], end[0]], [start[1], end[1]], 'r', linestyle="-",
    #                 alpha=alphas[i - 1])  # (rs[i-1], 0.0, 0.0, alphas[i-1])
    #
    # axs[2].set_xlim([0, 100])
    # axs[2].set_ylim([0, 100])
    # axs[2].set_title('HeatMap')

    #fig.savefig('../data/angle_plot.png')
    # if not os.path.exists(os.path.join(args.dir, 'eye_poses_csv')):
    #     os.mkdir(os.path.join(args.dir, 'eye_poses_csv'))

    with open(os.path.join(args.dir, '../eye_poses_csv/fish_{}/eye_pos_{}-{}.csv'.format(args.fish_place, indices[0], indices[-1])), 'w') as eye_file:
        csv_writer = csv.writer(eye_file, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['frame', 'x', 'y', 'pupil_size', 'eye_size', 'dist', 'angle'])
        count = 0
        for ei in eye_info_list:
            csv_writer.writerow([400 + count, ei['vector'][0], ei['vector'][1],
                                 ei['pupil_size'], ei['eye_size'], ei['dist'], ei['angle']])
            count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json', help='file of template fish')
    parser.add_argument('--dir', type=str, default='../data/output/pose_pickle', help='Folder contains pickle file')
    parser.add_argument('--index_range', type=str, default='63-142', help='clip index range')
    #parser.add_argument('-f', '--file', type=str, default='pose_pickle/pose_result_0-50_(2).pickle', help='pickle file name')
    parser.add_argument('--datadir', type=str, default='../data/input_frames/video_frames',
                        help='Folder for input dataset')
    parser.add_argument('--eye_model', type=str, default='../models/trained_models/eye_detection_model_2022-06-21',
                        help='path to mask-rcnn model for eye area detection')

    parser.add_argument('-fp', '--fish_place', type=int, default=2, help='which fish to reconstruct')

    parser.add_argument('-e', '--epoches', type=int, default=150, help='number of epoches for denoising')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, )


    args = parser.parse_args()

    detect_eye(args)