import argparse
import csv
import json
import math
import operator

import h5py
import os
import cv2
import numpy as np

def process_frames(frames, min_count=200):
    start = frames[0]
    count = 1
    count_sum = 0

    frame_cut = []
    frame_split = []

    for i in range(len(frames)):
        if i == 0:
            continue
        if frames[i] == frames[i-1] + 1:
            count += 1
        else:
            if count < min_count:
                start = frames[i]
                count = 1
                continue

            frame_cut.append((start, frames[i-1]))
            frame_split.append((count_sum, count_sum + count))

            count_sum += count
            start = frames[i]
            count = 1

    if count >= min_count:
        frame_cut.append((start,frames[i]))
        frame_split.append((count_sum, count_sum + count))

    return frame_cut, frame_split

def output_frames(videos, video_code, frame_cut, frame_split, shift=0, dist='../data/input/GoldFish20171216_BL320'):
    if not os.path.exists(dist):
        raise Exception('Output dir do not exist')

    dist = os.path.join(dist, 'video_frames_' + video_code)
    if not os.path.exists(dist):
        os.mkdir(dist)

    json_out_file = open(os.path.join(dist, 'index.json'), "w")
    json_index = {}
    json_index['frame_folders'] = []
    csv_list = {}

    satisfied_frames = []
    for t in frame_cut:
        satisfied_frames += list(range(t[0], t[1] + 1))

    for video_path in videos:
        if not os.path.exists(video_path):
            raise Exception('Input video do not exist')

        print("processing video: {}, front shift: {}".format(video_path, shift))

        video_name = video_path.split('/')[-1]
        ext_len = len(video_name.split('.')[-1])
        if video_name[-12:] == '21797353.mp4':
            video_name = 'sample-bottom'
        elif video_name[-12:] == '21990451.mp4':
            video_name = 'sample-front'
        save_path = os.path.join(dist, video_name, 'origin')
        if not os.path.exists(os.path.join(dist, video_name)):
            os.mkdir(os.path.join(dist, video_name))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        csv_out_file = open(os.path.join(dist, video_name, 'files.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder'])

        # cv2 extract frames
        capture = cv2.VideoCapture(video_path)

        image_count = 0
        frame_number = 0
        success = True
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        while capture.isOpened() and success:
            success, frame = capture.read()
            print('frame in process: {}, total image: {}'.format(frame_number, image_count), end='\r')

            if video_name == 'sample-bottom':
                if frame_number not in satisfied_frames:
                    frame_number += 1
                    continue
                if (frame_number + shift) < 0 or (frame_number + shift) > (total_frames - 1):
                    frame_number += 1
                    continue

            if video_name == 'sample-front':
                if frame_number - shift not in satisfied_frames:
                    frame_number += 1
                    continue

            if success:
                file_loc = os.path.join(save_path, "{}_{}.png".format(video_name, image_count))
                cv2.imwrite(file_loc, frame)

                csvwriter.writerow([image_count,
                                    os.path.join(video_name, "origin/{}_{}.png".format(video_name, image_count)),
                                    'origin',
                                    0,
                                    video_name])
                image_count += 1

            frame_number += 1

            # if image_count > 5:
            #     break

        print('total image: {}, done                  '.format(image_count))

        json_index['frame_folders'].append(video_name)
        csv_list[video_name] = os.path.join(dist, video_name, 'files.csv')

        csv_out_file.close()
        capture.release()

    json_index['status'] = 'origin'
    json_index['index_files'] = csv_list
    json_index['image_count'] = image_count
    json_index['frame_split'] = frame_split

    json.dump(json_index, json_out_file)
    json_out_file.close()

def output_coord(frame_cut, xy_coord, xz_coord, dist):
    frames = []
    for c in frame_cut:
        frames += list(range(c[0], c[1] + 1))

    out_coord_xy = xy_coord[:,frames,:]
    out_coord_xz = xz_coord[:, frames,:]

    if not os.path.exists(dist):
        os.makedirs(dist)

    with open(os.path.join(dist, 'xy_xz_coords.csv'), 'w') as of:
        csvwriter = csv.writer(of, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'fish1_xy_x', 'fish1_xy_y', 'fish2_xy_x', 'fish2_xy_y', 'fish1_xz_x', 'fish1_xz_z', 'fish2_xz_x', 'fish2_xz_z'])

        for i in range(len(frames)):
            csvwriter.writerow([i, 2048 - out_coord_xy[1,i,0], out_coord_xy[1,i,1], 2048 - out_coord_xy[0,i,0], out_coord_xy[0,i,1],  # xy plane flipped horizontally
                                   out_coord_xz[0,i,0], out_coord_xz[0,i,1], out_coord_xz[1,i,0], out_coord_xz[1,i,1]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str,
                        default='/home/liang/Documents/ruiheng/extra_videos/GoldFish20171215_BL330',
                        help='Folder for input videos')
    parser.add_argument('--outdir', type=str,
                        default="/media/liang/Samsung_T5/Ruiheng/reconstructions/filtered_videos/GoldFish20171215_BL330_long",
                        help='Folder where output locates')


    pos_dir = "/media/liang/recordings/recnode_Liang/MPIO_videos/MPIO_videos_kl/GoldFish20171215_BL330/"

    args = parser.parse_args()

    video_folders = [
        "goldfish_speed12_bls_20171215_102727",
        "goldfish_speed12_bls_20171215_110550",
        "goldfish_speed12_bls_20171215_113430",
        "goldfish_speed12_bls_20171215_120138",
        "goldfish_speed12_bls_20171215_132817",
        "goldfish_speed12_bls_20171215_145115",
        "goldfish_speed13_bls_20171215_133839",
        "goldfish_speed13_bls_20171215_144051",
        "goldfish_speed13_bls_20171215_151156",
        "goldfish_speed14_bls_20171215_125802",
        "goldfish_speed14_bls_20171215_130834",
        "goldfish_speed14_bls_20171215_134901",
        "goldfish_speed14_bls_20171215_142007",
        "goldfish_speed14_bls_20171215_152217",
        "goldfish_speed15_bls_20171215_135924",
        "goldfish_speed15_bls_20171215_140945",
        "goldfish_speed15_bls_20171215_150134",
        "goldfish_speed16_bls_20171215_101550",
        "goldfish_speed16_bls_20171215_103829",
        "goldfish_speed16_bls_20171215_105330",
        "goldfish_speed16_bls_20171215_112002",
        "goldfish_speed16_bls_20171215_114808",
        "goldfish_speed16_bls_20171215_131754",
        "goldfish_speed16_bls_20171215_143029",
        "goldfish_speed16_bls_20171215_153242",
    ]

    for v in video_folders:
        args.video_name = v

        bottom_folder = args.video_name + '.21797353'
        front_folder = args.video_name + '.21990451'

        position_file = args.video_name + '.21797353_position_xyz.h5'

        if os.path.isfile(os.path.join(pos_dir, bottom_folder, position_file)):
            print(f'filtering video {v} with criteria')
            # 1,3: raw detection; 2,4: apply Kalman filter
            with h5py.File(os.path.join(pos_dir, bottom_folder, position_file), "r") as posfile:
                xy_coord = posfile[list(posfile.keys())[2]][()]
                xz_coord = posfile[list(posfile.keys())[4]][()]

            frame_count = min(xy_coord.shape[1], xz_coord.shape[1])

            satisfied_frames = []

            for i in range(frame_count):
                satisfied_frames.append(i)

        else:
            print(f'output video {v} all frame')
            cap = cv2.VideoCapture(os.path.join(args.datadir, bottom_folder, bottom_folder + '.mp4'))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            satisfied_frames = list(range(frame_count))

        # try:
        if len(satisfied_frames) == 0:
            print(f'video {v} contains no satisfying sequence')
            continue
        else:
            frame_cut, frame_split = process_frames(satisfied_frames)
        # except:
        #     print("video {} failed".format(args.video_name))
        #     continue

        if os.path.isfile(os.path.join(pos_dir, bottom_folder, position_file)):
            shifts = []
            search_range = 200
            for i in range(2):
                x_front = xz_coord[i,:,0]
                x_bottom = 2048 - xy_coord[i,:,0]

                cors = windowed_cor_shift(x_front, x_bottom, search_range)
                shifts.append(np.argmax(cors) - search_range)

            avg_shift = int(np.mean(np.array(shifts)))
        else:
            avg_shift = 0

        # frame_cut, frame_split = process_frames(list(range(15, 50)) + list(range(60, 90)) + list(range(92, 98)) + list(range(100, 150)))
        if len(frame_cut) == 0:
            print(f'video {v} has no sequence long enough')
            continue
        print(frame_cut)
        # print(frame_split)
        # print(len(list(range(15, 50)) + list(range(60, 90)) + list(range(92, 98)) + list(range(100, 150))))

        videos = [os.path.join(args.datadir, bottom_folder, bottom_folder + '.mp4'),
                  os.path.join(args.datadir, front_folder, front_folder + '.mp4')]

        video_name = '_'.join(args.video_name.split('_')[-2:])

        #output_frames(videos, video_name, frame_cut, frame_split, dist=args.outdir)
        try:
            output_frames(videos, video_name, frame_cut, frame_split, shift=avg_shift, dist=args.outdir)
        except:
            print("video {} failed".format(args.video_name))

        if os.path.isfile(os.path.join(pos_dir, bottom_folder, position_file)):
            output_coord(frame_cut, xy_coord, xz_coord, os.path.join(args.outdir, 'video_frames_' + video_name))