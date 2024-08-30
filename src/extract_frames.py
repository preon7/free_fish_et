import cv2
import csv
import json
import os
import numpy as np
import torch
import deeplabcut

from src.dataloaders import UniLabDataset
from PIL import Image
from tqdm import trange

import torchvision.transforms as T
import torch.nn.functional as F
import models.MaskRCNN as MRCNN

def extract_from_video(videos, out_dir, out_folder='video_frames'):

    if not os.path.exists(out_dir):
        raise Exception('Output dir do not exist')

    dist = os.path.join(out_dir, out_folder)
    if not os.path.exists(dist):
        os.mkdir(dist)

    json_out_file = open(os.path.join(dist, 'index.json'), "w")
    json_index = {}
    json_index['frame_folders'] = []
    csv_list = {}

    for video_path in videos:
        if not os.path.exists(video_path):
            raise Exception('Input video do not exist')

        print("processing video: {}".format(video_path))

        video_name = video_path.split('/')[-1]
        ext_len = len(video_name.split('.')[-1])
        save_path = os.path.join(dist, video_name[:-ext_len-1], 'origin')
        if not os.path.exists(os.path.join(dist, video_name[:-ext_len-1])):
            os.mkdir(os.path.join(dist, video_name[:-ext_len - 1]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        csv_out_file = open(os.path.join(dist, video_name[:-ext_len-1], 'files.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder'])

        # cv2 extract frames
        capture = cv2.VideoCapture(video_path)

        image_count = 0
        frame_number = 0
        success = True

        while capture.isOpened() and success:
            success, frame = capture.read()

            if success:
                file_loc = os.path.join(save_path, "{}_{}.png".format(video_name[:-ext_len-1], frame_number))
                cv2.imwrite(file_loc, frame)
                image_count += 1
                csvwriter.writerow([frame_number,
                                    os.path.join(video_name[:-ext_len-1], "origin/{}_{}.png".format(video_name[:-ext_len-1], frame_number)),
                                    'origin',
                                    0,
                                    video_name[:-ext_len-1]])

            frame_number += 1
            print('frame out: {}, total image: {}'.format(frame_number, image_count), end='\r')

            # if image_count > 5:
            #     break

        print('total image: {}, done                  '.format(image_count))

        json_index['frame_folders'].append(video_name[:-ext_len-1])
        csv_list[video_name[:-ext_len-1]] = os.path.join(dist, video_name[:-ext_len-1], 'files.csv')

        csv_out_file.close()
        capture.release()

    json_index['status'] = 'origin'
    json_index['index_files'] = csv_list
    json_index['image_count'] = image_count

    json.dump(json_index, json_out_file)
    json_out_file.close()


def process_input_folder(data_folder):
    with open(os.path.join(data_folder, 'index.json')) as jf:
        video_meta = json.load(jf)

    n_frames = video_meta['image_count']

    # process bottom images
    with open(os.path.join(data_folder, 'bottom', 'frame2video_1.csv'), 'w') as csv_out_file:
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['origin_frame', 'new_frame'])

        for i in range(n_frames):
            csvwriter.writerow([i, i])

    # create bottom video
    if not os.path.exists(os.path.join(data_folder, 'bottom', 'dlc_results')):
        os.mkdir(os.path.join(data_folder, 'bottom', 'dlc_results'))

    out = cv2.VideoWriter(os.path.join(data_folder, 'bottom', 'dlc_results', 'full_size.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (2048, 1040))
    frame_count = 0

    for i in range(n_frames):
        img1 = cv2.imread(os.path.join(data_folder, 'bottom', 'origin', 'bottom_{}.png'.format(i)))
        out.write(img1)
        print('bottom video frame out: {}'.format(frame_count), end='\r')
        frame_count += 1

    out.release()

    # flip bottom images
    for i in range(n_frames):
        img = cv2.imread(os.path.join(data_folder, 'bottom', 'origin', 'bottom_{}.png'.format(i)))
        img_flip_h = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(data_folder, 'bottom', 'origin', 'bottom_{}.png'.format(i)), img_flip_h)
        print('flipping frame: {}'.format(i), end='\r')

    # process front images
    with open(os.path.join(data_folder, 'front', 'frame2video_1.csv'), 'w') as csv_out_file:
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['origin_frame', 'new_frame'])

        for i in range(n_frames):
            csvwriter.writerow([i, i])

    # undistort front images
    img_list = os.listdir(os.path.join(data_folder, 'front', 'origin'))
    mtx = np.array([[3946, 0, 1080], [0, 3934, 520], [0, 0, 1]])
    dist = np.array([-0.568857779226978, 0.151730496415158, 0, 0, 0])
    w = 2048
    h = 1040
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    iid = 1
    for img in img_list:
        if img[-4:] != '.png':
            continue

        dst = cv2.undistort(cv2.imread(os.path.join(data_folder, 'front', 'origin', img)), mtx, dist, None, newcameramtx)
        cv2.imwrite(os.path.join(data_folder, 'front', 'origin', img), dst)
        print('undistorting image: {}'.format(iid), end='\r')
        iid += 1

    # create front video
    if not os.path.exists(os.path.join(data_folder, 'front', 'dlc_results')):
        os.mkdir(os.path.join(data_folder, 'front', 'dlc_results'))

    out = cv2.VideoWriter(os.path.join(data_folder, 'front', 'dlc_results', 'full_size.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 20, (2048, 1040))
    frame_count = 0

    for i in range(n_frames):
        img1 = cv2.imread(os.path.join(data_folder, 'front', 'origin', 'front_{}.png'.format(i)))
        out.write(img1)
        print('front video frame out: {}'.format(frame_count), end='\r')
        frame_count += 1

    out.release()


def predict(dataset_path, model_path, device, num_classes=2):
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def collate_fn(batch):
        return tuple(zip(*batch))

    # get the model using our helper function
    model = MRCNN.get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    dataset = UniLabDataset(dataset_path, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    # create csv index files in each video folder
    jf = open(os.path.join(dataset_path, 'index.json'))
    index_json = json.load(jf)
    image_folders = index_json['frame_folders']

    csvwriters = {}
    for folder in image_folders:
        if not os.path.exists(os.path.join(dataset_path, folder, 'cropped')):
            os.mkdir(os.path.join(dataset_path, folder, 'cropped'))

        if not os.path.exists(os.path.join(dataset_path, folder, 'mask')):
            os.mkdir(os.path.join(dataset_path, folder, 'mask'))

        if not os.path.exists(os.path.join(dataset_path, folder, 'mask_full')):
            os.mkdir(os.path.join(dataset_path, folder, 'mask_full'))

        csv_out_file = open(os.path.join(dataset_path, folder, 'files_crop.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox'])

        csvwriters[folder] = csvwriter

    target_frames = list(range(index_json['image_count']))

    model.eval()
    with torch.no_grad():
        k = 0
        pbar = trange(len(target_frames), desc="detect from frames")
        for image, label in data_loader_test:
            # only use non-overlap frames
            label = label[0]
            if int(label['frame']) not in target_frames:
                continue

            pbar.set_description('detecting frame {}'.format(label['frame']))
            images = torch.from_numpy(np.array(Image.open(image[0]).convert("RGB"))) #image[0]
            csvwriter = csvwriters[label['folder']]

            crop_image = images.permute(1,0,2)

            images = [images.to(device).permute(2, 0, 1) / 255.]
            predictions = model(images)

            for i in range(predictions[0]['boxes'].size()[0]):
                # only 2 fishes in the scene
                if i > 1:
                    break
                mask = predictions[0]['masks'][i, 0].cpu().numpy()

                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                bounding_box = [xmin, ymin, xmax, ymax]

                cropped = crop_image[int(bounding_box[0]):int(bounding_box[2]),
                          int(bounding_box[1]):int(bounding_box[3])]

                crop_mask = predictions[0]['masks'][i, 0].mul(255).permute(1, 0)
                cropped_mask = crop_mask[int(bounding_box[0]):int(bounding_box[2]),
                               int(bounding_box[1]):int(bounding_box[3])]

                diff = abs(bounding_box[2] - bounding_box[0] - (bounding_box[3] - bounding_box[1]))

                output = cropped.permute(2,0,1)
                out_mask = cropped_mask

                if bounding_box[2] - bounding_box[0] < bounding_box[3] - bounding_box[1]:
                    # padding height
                    output = F.pad(input=output,
                                   pad=(0, 0, int(diff / 2.0),
                                        int(diff / 2.0)),
                                   mode='constant', value=0)
                    out_mask = F.pad(input=cropped_mask,
                                     pad=(0, 0, int(diff / 2.0),
                                          int(diff / 2.0)),
                                     mode='constant', value=0)

                if bounding_box[2] - bounding_box[0] > bounding_box[3] - bounding_box[1]:
                    # padding height
                    output = F.pad(input=output,
                                   pad=(int(diff / 2.0),
                                        int(diff / 2.0), 0, 0),
                                   mode='constant', value=0)
                    out_mask = F.pad(input=cropped_mask,
                                     pad=(int(diff / 2.0),
                                          int(diff / 2.0), 0, 0),
                                     mode='constant', value=0)

                crop_out_dir = os.path.join(dataset_path, label['folder'], 'cropped')
                mask_out_dir = os.path.join(dataset_path, label['folder'], 'mask')
                mask_full_dir = os.path.join(dataset_path, label['folder'], 'mask_full')

                output = Image.fromarray(output.permute(2, 1, 0).cpu().byte().numpy())
                output.save(os.path.join(crop_out_dir, 'image_{}_{}.png'.format(k, i)))
                predicted_mask = Image.fromarray(out_mask.permute(1, 0).byte().cpu().numpy())
                predicted_mask.save(os.path.join(mask_out_dir, 'image_{}_{}_mask.png'.format(k, i)))
                full_mask = Image.fromarray(predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                full_mask.save(os.path.join(mask_full_dir, 'image_{}_{}_mask.png'.format(k, i)))

                crop_out_dir = os.path.join(crop_out_dir, 'image_{}_{}.png'.format(k, i))
                mask_out_dir = os.path.join(mask_out_dir, 'image_{}_{}_mask.png'.format(k, i))

                csvwriter.writerow([label['frame'],
                                    '/'.join(crop_out_dir.split('/')[-3:]),
                                    'cropped',
                                    str(i),
                                    label['folder'],
                                    str(bounding_box)])

                csvwriter.writerow([label['frame'],
                                    '/'.join(mask_out_dir.split('/')[-3:]),
                                    'mask',
                                    str(i),
                                    label['folder'],
                                    str(bounding_box)])

            pbar.update(1)
            k += 1
    print('\n finish')

def detect_dlc(data_folder,
               front_config_path='/home/lab/Documents/fish_mesh_eye_public/models/trained_models/master2021demo_front-Ruiheng Wu-2021-06-02/config.yaml',
               bottom_config_path='/home/lab/Documents/fish_mesh_eye_public/models/trained_models/master2021demo_bottom-Ruiheng Wu-2021-06-01/config.yaml'):

    deeplabcut.analyze_videos(front_config_path,
                              [os.path.join(data_folder, 'front', 'dlc_results', 'full_size.mp4')],
                              videotype='.mp4',
                              engine=deeplabcut.core.engine.Engine.TF)

    deeplabcut.analyze_videos(bottom_config_path,
                              [os.path.join(data_folder, 'bottom', 'dlc_results', 'full_size.mp4')],
                              videotype='.mp4',
                              engine=deeplabcut.core.engine.Engine.TF)