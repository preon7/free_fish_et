import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch


import src.multiview as multiview
import src.multiview_utils as mutil

from tqdm import tqdm
from src.fish_model import fish_model
from src.pose_optimizer import OptimizeMV
from src.Silhouette_Renderer import Silhouette_Renderer
from src.dataloaders import Multiview_Dataset

def reconstruct(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Load model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    fish = fish_model(mesh=args.mesh)
    optimizer = OptimizeMV(num_iters=100, lim_weight=200, prior_weight=30,
                           bone_weight=200, mask_weight=200, smooth_weights=[100, 100, 20],
                           device=device, mesh=args.mesh)
    silhouette_renderer = Silhouette_Renderer(device=device)

    multiview_data = Multiview_Dataset(root=args.datadir)

    count_start = multiview_data[args.index[0]]['full_kpts']
    start_index = args.index[0]

    individual_fit_parameters = []
    # individual_fit_parameters_2 = []  # for the second fish
    sample_data = []
    # sample_data_2 = []
    prev_keypoints = None
    eye_info_list = []

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if not os.path.exists(os.path.join(args.outdir, f'images_fish_{args.fish_place}')):
        os.mkdir(os.path.join(args.outdir, f'images_fish_{args.fish_place}'))

    pbar = tqdm(total=len(args.index))
    pbar.set_description(f'video {args.datadir.split("/")[-1]}')

    for sample_index in args.index:
        # print("processing sample: {}/{}".format(sample_index, args.index[-1]), end='\r')
        if sample_index > start_index:
            prev_sample = sample
        else:
            prev_sample = None
        try:
            sample = multiview_data[sample_index]
        except:
            print(f'sample {sample_index} failed, using previous one')
            sample = prev_sample


        if not sample['full_kpts']:
            if not count_start:
                print("frame {} has missing keypoints".format(sample_index))
                individual_fit_parameters += [0,0,0,0]
                sample_data.append([0])
                count_start = multiview_data[sample_index + 1]['full_kpts']
                start_index = sample_index + 1
            else:
                individual_fit_parameters += [individual_fit_parameters[-4], individual_fit_parameters[-3],
                                              individual_fit_parameters[-2], individual_fit_parameters[-1]]
                sample_data.append(sample_data[-1])
                print("frame {} has missing keypoints".format(sample_index))
            continue

        # if prev_keypoints is not None:
        #     # distance of head
        #     dist1 = torch.nn.MSELoss(reduction='sum')(prev_keypoints[1, 0, 0:2], sample["keypoints"][1, 0, 0:2]).item()
        #     dist2 = torch.nn.MSELoss(reduction='sum')(prev_keypoints[1, 0, 0:2], sample["keypoints2"][1, 0, 0:2]).item()
        #
        # if prev_keypoints is not None and dist2 < dist1:
        #     frames = sample["frames"]
        #     img_filenames = sample["imgpaths"]
        #     keypoints = torch.cat([sample['keypoints'][[0]], sample["keypoints2"][[1]]])
        #     masks = sample["masks2"].to(device) / 255.
        #     bboxs = sample["bboxes2"]
        #
        #     keypoints_2 = sample["keypoints"]
        #     masks_2 = sample["masks"].to(device) / 255.
        #     bboxs_2 = sample["bboxes"]
        # else:
        frames = sample["frames"]
        img_filenames = sample["imgpaths"]
        if args.fish_place == 1:
            keypoints = sample["keypoints"]
            masks = sample["masks_full"].to(device) / 255.
            bboxs = sample["bboxes"]
        elif args.fish_place == 2:
            keypoints = sample["keypoints2"]
            masks = sample["masks_full2"].to(device) / 255.
            bboxs = sample["bboxes2"]
        else:
            raise Exception('invalid fish place: {}'.format(args.fish_place))

        # keypoints_2 = torch.cat([sample['keypoints'][[0]], sample["keypoints2"][[1]]])
        # masks_2 = sample["masks2"].to(device) / 255.
        # bboxs_2 = sample["bboxes2"]
        #
        # prev_keypoints = keypoints.clone()

        # sample_data_2.append([frames, img_filenames, keypoints_2, masks_2, bboxs_2, sample_index])

        # for i in range(len(img_filenames)):
        #     frame = [frames[i]]
        #     img = plt.imread(img_filenames[i]) * 255
        #     img_pose, img_model = mutil.render_vertex_on_frame(img, [vertex_posed], fish, frame, [keypoints], bboxs[i])
        #     #img_pose, img_model = mutil.render_vertex_on_frame(img, [debug_vert], fish, frame, [keypoints], bboxs[i])
        #
        #     plt.imsave(args.outdir +'/debug_view_{}.png'.format(i), img_pose)

        prev_index = sample_index - args.index[0] - 1
        if (prev_index >= 0) and (len(sample_data[-1]) != 1):
            init_ori = individual_fit_parameters[4 * prev_index][:, 0:3]
            init_pose = individual_fit_parameters[4 * prev_index][:, 3:]
            init_bone = individual_fit_parameters[4 * (start_index - args.index[0]) + 1] # start_index to fix bone length
            init_s = individual_fit_parameters[4 * prev_index + 2]
            init_t = individual_fit_parameters[4 * prev_index + 3]

            # init_ori_2 = individual_fit_parameters_2[4 * prev_index][:, 0:3]
            # init_pose_2 = individual_fit_parameters_2[4 * prev_index][:, 3:]
            # init_bone_2 = individual_fit_parameters_2[4 * prev_index + 1]
            # init_s_2 = individual_fit_parameters_2[4 * prev_index + 2]
            # init_t_2 = individual_fit_parameters_2[4 * prev_index + 3]
        else:
            init_ori = None
            init_pose = None
            init_bone = None
            init_s = None
            init_t = None

            # init_ori_2 = None
            # init_pose_2 = None
            # init_bone_2 = None
            # init_s_2 = None
            # init_t_2 = None

        sample_data.append([frames, img_filenames, keypoints, bboxs, sample_index])

        vertex_posed, mesh_keypoints, t, body_pose, bone, scale, losses \
            = multiview.fit_mesh(fish, optimizer, keypoints, frames, masks, silhouette_renderer, device,
                                 init_ori, init_t, init_s, init_pose, init_bone, img_filenames, sample_index, bboxs)

        # vertex_posed_2, mesh_keypoints_2, t_2, body_pose_2, bone_2, scale_2, losses_2 \
        #     = multiview.fit_mesh(fish, optimizer, keypoints_2, frames, masks_2, silhouette_renderer, device,
        #                          init_ori_2, init_t_2, init_s_2, init_pose_2, init_bone_2)

        individual_fit_parameters += [body_pose, bone, scale, t]

        # fish_posed = fish(body_pose[:,:3], body_pose[:,3:], bone, scale)

        # Rx_90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        # Ry_90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        # Rz_90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        # #fish_posed = fish(global_p, local_p, bones, scale)
        # pose_mat = batch_rodrigues(body_pose.view(-1, 3))
        # rot = torch.tensor([[-1.5584e-01,  3.5939e-02,  9.8713e-01],
        #                     [ 9.5924e-01,  2.4401e-01,  1.4255e-01],
        #                     [-2.3574e-01,  9.6911e-01, -7.2499e-02]])
        #
        # rot_head = torch.tensor([[ 9.9000e-01, -4.5459e-05,  1.4104e-01],
        #                          [ 4.5459e-05,  1.0000e+00,  3.2218e-06],
        #                          [-1.4104e-01,  3.2218e-06,  9.9000e-01]])
        # rot = pose_mat[0]
        # rot_head = pose_mat[1]
        #
        # revert_mat = Ry_90 @ (Rz_90 @ Rz_90 @ Rz_90).T @ torch.inverse(rot) #@ torch.inverse(rot_head)
        # debug_vert = (revert_mat @ fish_posed['vertices'].unsqueeze(-1))[:,:,:,0] + torch.tensor([0, 0.4, 0])
        # debug_vert = fish_posed['vertices'] + torch.tensor([0, 0.4, 0])
        # individual_fit_parameters_2 += [body_pose_2, bone_2, scale_2, t_2]

        # et = EyeTracker((512,512), args.eye_model)
        # reverted_fish, eye_info = et.find_eye_pos(plt.imread(img_filenames[0]) * 255, body_pose, bboxs[0])
        # reverted_fish = reverted_fish.astype(np.uint8)
        # plt.imsave(args.outdir +'/standard_images/{}_reverted.png'.format(sample_index), reverted_fish)
        # eye_info_list.append(eye_info)

        # Save reconstruction results as images and meshes ==================

        for i in range(len(img_filenames)):
            frame = [frames[i]]
            img = plt.imread(img_filenames[i]) * 255
            img_pose, img_model = mutil.render_vertex_on_frame(img, [vertex_posed], fish, frame, [keypoints], bboxs[i])
            # img_pose, img_model = mutil.render_vertex_on_frame(img, [debug_vert], fish, frame, [keypoints], bboxs[i])

            plt.imsave(args.outdir + '/images_fish_{}/{}_view_{}.png'.format(args.fish_place, sample_index, i), img_pose)
            # plt.imsave(args.outdir + '/images/{}_model_view_{}.png'.format(sample_index, i), img_model)

        pbar.update(1)

        # save to .obj
        if args.save_models == True:
            if not os.path.exists(os.path.join(args.outdir, 'models')):
                os.mkdir(os.path.join(args.outdir, 'models'))

            with open(os.path.join(args.outdir, 'models', '{}_out_model_{}.obj'.format(sample_index, args.fish_place - 1)), 'w') as modelfile:
                for i in range(vertex_posed.size(1)):
                    modelfile.write("v {0} {1} {2}\n".format(vertex_posed[0,i,0], vertex_posed[0,i,1], vertex_posed[0,i,2]))

                for i in range(fish.faces.size(0)):
                    modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i,0]+1, fish.faces[i,1]+1, fish.faces[i,2]+1))

        # with open(os.path.join(args.outdir, 'models', '{}_out_model_1.obj'.format(sample_index)), 'w') as modelfile:
        #     for i in range(vertex_posed_2.size(1)):
        #         modelfile.write("v {0} {1} {2}\n".format(vertex_posed_2[0,i,0], vertex_posed_2[0,i,1], vertex_posed_2[0,i,2]))
        #
        #     for i in range(fish.faces.size(0)):
        #         modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i,0]+1, fish.faces[i,1]+1, fish.faces[i,2]+1))

    # plot eye tracking
    # xs = np.array(range(len(eye_info_list)))
    # ys = np.array([e['angle'] for e in eye_info_list])
    # ys2 = np.array([e['dist'] for e in eye_info_list])
    #
    # fig, axs = plt.subplots(2)
    # axs[0].plot(xs, ys)
    # axs[0].set_title('angle-time')
    # axs[1].plot(xs, ys2)
    # axs[1].set_title('dist-time')
    #
    # fig.savefig('data/angle_plot.png')

    # save posed vertices ======================

    packed_data = {"individual_fit_parameters": individual_fit_parameters,
                   "sample_data": sample_data,
                   # "eye_info_list": eye_info_list,
                   # "individual_fit_parameters_2": individual_fit_parameters_2,
                   # "sample_data_2": sample_data_2,
                   "indices": args.index,
                   "model_file": args.mesh}

    if not os.path.exists(os.path.join(args.outdir, 'pose_pickle')):
        os.mkdir(os.path.join(args.outdir, 'pose_pickle'))

    with open(os.path.join(args.outdir, 'pose_pickle/pose_result_{}-{}_({}).pickle'.format(args.index[0], args.index[-1] + 1, args.fish_place)),
              'wb') as pf:
        pickle.dump(packed_data, pf, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', nargs='+', default=list(range(50, 100)), # (1799,1935)
                        help='Index for an example in the multiview dataset')  # 1: 10 2: 1732 [10, 150, 500, 1320, 1732]
    # parser.add_argument('--window_size', type=int, default=7, help='number of consecutive frames to process together')
    # parser.add_argument('--stride', type=int, default=3,
    #                     help='distance between the first index of i-th sequence and (i+1)-th sequence')

    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json', help='file of template fish')
    parser.add_argument('--outdir', type=str, default='data/output/GoldFish20171216_BL320/20171216_124610', help='Folder for output images')
    parser.add_argument('--datadir', type=str, default='data/input/video_frames_20171215_101550',
                        help='Folder for input dataset')
    parser.add_argument('--fish_place', type=int, default=2, help='1 for front fish, 2 for back fish')

    parser.add_argument('--eye_model', type=str, default='model/saved_models/eye_detection_model_2022-03-09',
                        help='path to mask-rcnn model for eye area detection')
    parser.add_argument('--seed', type=int, default=1, help='RNG for reproducibility')

    parser.add_argument('--save_method', type=str, default='pickle',
                        help='how the result poses are stored (pickle / mongodb)')
    parser.add_argument('--save_models', type=bool, default=False,
                        help='save obj models')

    args = parser.parse_args()

    reconstruct(args)