"""
functions for multiview output from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""
# import trimesh
import yaml
import os
import numpy as np
import torch
import src.constants as c
import cv2

# from .renderer import Renderer
from .geometry import perspective_projection, perspective_projection_homo, perspective_projection_ref


def get_fullsize_masks(masks, bboxes, h=368, w=368):
    full_masks = []
    for i in range(len(masks)):
        box = bboxes[i]
        full_mask = torch.zeros([h, w], dtype=torch.bool)
        full_mask[box[1]:box[1] + box[3] + 1, box[0]:box[0] + box[2] + 1] = masks[i]
        full_masks.append(full_mask)
    full_masks = torch.stack(full_masks)

    return full_masks


def get_cam(device='cpu'):
    proj_m_set = torch.stack([c.proj_front, c.proj_bottom], 0).to(device)
    proj_m_set_homo = torch.cat([proj_m_set, torch.tensor([[[0,0,-1,0]], [[0,0,-1,0]]]).to(device)], 1)
    f1 = 3930.0
    f2 = 3930
    focal = torch.tensor([f1, f1]).to(device)
    center = torch.tensor([[1024.,520.],[1024.,520.]]).to(device)
    # K = torch.tensor([[f1/2048,0,0.],[0,f1/1040,0.],[0,0,1]]).to(device)
    K = torch.tensor([[f1, 0, 1024.], [0, f1, 520.], [0, 0, 1]]).to(device)
    # K = torch.tensor([[ 7.67578125,0.,-1.,0.],
    #                  [ 0.,15.11538462,1.,0.],
    #                  [ 0.,0.,-1.00010001,-0.100005],
    #                  [ 0.,0.,-1.,0.]])
    # K = torch.tensor([[3.83789062,0.,0.,0.],
    #                 [0.,7.55769231,0.,0.],
    #                 [0.,0.,- 1.00010001,- 0.100005],
    #                 [0.,0.,- 1.,0.]])
    H = torch.matmul(K.inverse(), proj_m_set)
    # H = torch.matmul(K.inverse(), proj_m_set_homo)

    distortion = torch.tensor(c.distortion).to(device)

    return  proj_m_set, focal, center, H[:,:,:-1], H[:,:,-1], distortion


def projection_loss(x, y):
    loss = (x.float() - y.float()).norm(p=2)
    return loss


def triangulation_LBFGS(x, proj_m, #rotation, camera_t, focal_length, camera_center,
                        distortion=None, device='cpu'):
    n = x.shape[0]
    X = torch.tensor([2.5, 1.2, 1.95])[None, None, :]
    X.requires_grad_()

    x = x.to(device)
    X = X.to(device)

    losses = []
    optimizer = torch.optim.LBFGS([X], lr=1, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        # projected_points = perspective_projection_ref(X.repeat(n, 1, 1), rotation, camera_t, focal_length, camera_center, distortion)
        projected_points = perspective_projection(X.repeat(n, 1, 1), proj_m)
        loss = projection_loss(projected_points.squeeze(), x)

        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        # projected_points = perspective_projection_ref(X.repeat(n, 1, 1), rotation, camera_t, focal_length, camera_center, distortion)
        projected_points = perspective_projection(X.repeat(n, 1, 1), proj_m)
        loss = projection_loss(projected_points.squeeze(), x)
        losses.append(loss.detach().item())
    X = X.detach().squeeze()

    return X, losses


def triangulation(x, proj_m, #rotation, camera_t, focal_length, camera_center,
                  distortion=None, device='cpu'):
    n = x.shape[0]
    X = torch.tensor([2.5, 1.2, 1.95])[None, None, :]
    X.requires_grad_()

    x = x.to(device)
    X = X.to(device)

    losses = []
    optimizer = torch.optim.Adam([X], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 90], gamma=0.1)
    for i in range(100):
        # projected_points = perspective_projection_ref(X.repeat(n, 1, 1), rotation, camera_t, focal_length, camera_center, distortion)
        projected_points = perspective_projection(X.repeat(n, 1, 1), proj_m)
        loss = projection_loss(projected_points.squeeze(), x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().item())

    X = X.detach().squeeze()

    return X, losses


def get_gt_3d(keypoints, frames, LBFGS=False):
    '''
    Input:
        keypoints (bn, kn, 2): 2D kpts from different views
        frames (bn): frame numbers
    Output:
        kpts_3d (kn, 4): ground truth 3D kpts, with validility
    '''
    bn, kn, _ = keypoints.shape
    kpts_3d = torch.zeros([kn, 4])

    #
    proj_m_set, focal, center, R, T, distortion = get_cam()
    kpts_valid = []
    cams = []
    for i in range(kn):
        valid = keypoints[:, i, -1] > 0
        kpts_valid.append(keypoints[valid, i, :2])
        cams.append(valid)

    #
    for i in range(kn):
        x = kpts_valid[i]
        if len(x) >= 2:
            # proj_m = proj_m_set

            if LBFGS:
                X, _ = triangulation_LBFGS(x, proj_m_set)
            else:
                X, _ = triangulation(x, proj_m_set)

            kpts_3d[i, :3] = X
            kpts_3d[i, -1] = 1

    return kpts_3d


def Procrustes(X, Y):
    """
    Solve full Procrustes: Y = s*RX + t
    Input:
        X (N,3): tensor of N points
        Y (N,3): tensor of N points in world coordinate
    Returns:
        R (3x3): tensor describing camera orientation in the world (R_wc)
        t (3,): tensor describing camera translation in the world (t_wc)
        s (1): scale
    """
    # remove translation
    A = (Y - Y.mean(dim=0, keepdim=True))
    B = (X - X.mean(dim=0, keepdim=True))

    # remove scale
    sA = (A * A).sum() / A.shape[0]
    sA = sA.sqrt()
    sB = (B * B).sum() / B.shape[0]
    sB = sB.sqrt()
    A = A / sA
    B = B / sB
    s = sA / sB

    # to numpy, then solve for R
    A = A.t().numpy()
    B = B.t().numpy()

    M = B @ A.T
    U, S, VT = np.linalg.svd(M)
    V = VT.T

    d = np.eye(3)
    d[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ d @ U.T

    # back to tensor
    R = torch.tensor(R).float()
    t = Y.mean(axis=0) - R @ X.mean(axis=0) * s

    return R, t, s

def render_vertex_on_frame(img, vertex_posed, fish, frame, kpts=None, bboxs=None):
    proj_m_set, focal, center, R, T, distortion = get_cam()

    # Extrinsic
    # R = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [1, 0, 0])
    # transformed_points = torch.einsum('bij,bkj->bki', torch.tensor(R[:-1,:-1]).float().unsqueeze(0), vertex_posed)
    # transformed_points = torch.einsum('bij,bkj->bki', R[frame], vertex_posed) + T[frame]#.unsqueeze(1).repeat(1,vertex_posed.size(1),1)
    # transformed_points = torch.einsum('bij,bkj->bki', T[frame], torch.cat([vertex_posed,torch.tensor([[[1]]]).repeat(1,1306,1)],2))
    # transformed_points = torch.div(transformed_points , torch.stack([transformed_points[:,:,-1]] * 4, -1))

    # transformed_points = torch.matmul(R[frame], vertex_posed.permute(0,2,1)).permute(0,2,1) + T[frame]
    # Distortion
    # if distortion is not None:
    #     kc = distortion
    #     d = points[:, :, 2:]
    #     points = points[:, :, :] / points[:, :, 2:]
    #
    #     r2 = points[:, :, 0] ** 2 + points[:, :, 1] ** 2
    #     dx = (2 * kc[:, [2]] * points[:, :, 0] * points[:, :, 1]
    #           + kc[:, [3]] * (r2 + 2 * points[:, :, 0] ** 2))
    #
    #     dy = (2 * kc[:, [3]] * points[:, :, 0] * points[:, :, 1]
    #           + kc[:, [2]] * (r2 + 2 * points[:, :, 1] ** 2))
    #
    #     x = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 0] + dx
    #     y = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 1] + dy
    #
    #     points = torch.stack([x, y, torch.ones_like(x)], dim=-1) * d

    # points = transformed_points[0,:,:]

    points = torch.einsum('bij,bkj->bki', R[frame], vertex_posed[0]) + T[frame]
    points = points[0]
    # rot = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
    rot = np.eye(3)
    # print('middle point: {}'.format(torch.sum(points, axis=-2) / points.size(-2)))

    # Rendering
    bg_img = np.zeros((1040,2048,3))
    bg_img[:img.shape[0], :img.shape[1],:] = torch.tensor(img)

    # manual transform
    # projected = perspective_projection(vertex_posed[0].repeat(2,1,1), proj_m_set)[ frame[0]]
    projected_ref = perspective_projection_ref(vertex_posed[0].repeat(2,1,1), R, T, focal, center)[ frame[0]]
    ix = (torch.minimum(torch.maximum(projected_ref[:, 1].int(), torch.tensor(0)), torch.tensor(1039))).tolist()
    iy = (torch.minimum(torch.maximum(projected_ref[:, 0].int(), torch.tensor(0)), torch.tensor(2047))).tolist()
    img[ix, iy] = np.array([0, 255, 0])

    # points = torch.einsum('bij,bkj->bki', R[frame], vertex_posed[1]) + T[frame]
    # points = points[0]
    # # rot = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
    # rot = np.eye(3)
    # # print('middle point: {}'.format(torch.sum(points, axis=-2) / points.size(-2)))

    # # Rendering
    # bg_img = np.zeros((1040, 2048, 3))
    # bg_img[:img.shape[0], :img.shape[1], :] = torch.tensor(img)
    #
    # # manual transform
    # # projected = perspective_projection(vertex_posed[1].repeat(2,1,1), proj_m_set)[ frame[0]]
    # projected_ref = perspective_projection_ref(vertex_posed[1].repeat(2, 1, 1), R, T, focal, center)[frame[0]]
    # ix = (torch.minimum(torch.maximum(projected_ref[:, 1].int(), torch.tensor(0)), torch.tensor(1039))).tolist()
    # iy = (torch.minimum(torch.maximum(projected_ref[:, 0].int(), torch.tensor(0)), torch.tensor(2047))).tolist()
    # img[ix, iy] = np.array([0, 255, 125])

    # global_p = torch.zeros(1,3)  # [rot_x, rot_y, rot_z]
    # global_p[0,0] = 0
    #
    # local_p = torch.zeros(c.num_bone,3)
    # local_p[0,2] = 1
    # local_p = local_p.view(1,c.num_bone * 3)
    #
    # bones = torch.ones(1,c.num_bone)
    # bones[0,2] = 1
    # fish_posed = fish(global_p, local_p, bones, 0.01)

    # init model
    # projected = perspective_projection(fish.V.repeat(2, 1, 1) * 0.01 + torch.tensor([0,0.,0.0]), proj_m_set)[frame[0]]
    # projected = perspective_projection(fish_posed['vertices'].repeat(2, 1, 1), proj_m_set)[frame[0]]
    # ix = (torch.minimum(torch.maximum(projected[:, 1].int(), torch.tensor(0)), torch.tensor(1039))).tolist()
    # iy = (torch.minimum(torch.maximum(projected[:, 0].int(), torch.tensor(0)), torch.tensor(2047))).tolist()
    # img[ix, iy] = np.array([0, 0, 255])
    #
    # with open(os.path.join('data/output/multiview_demo', '{}_out_model.obj'.format('debug')), 'w') as modelfile:
    #     for i in range(fish_posed['vertices'].size(1)):
    #         modelfile.write(
    #             "v {0} {1} {2}\n".format(fish_posed['vertices'][0, i, 0], fish_posed['vertices'][0, i, 1], fish_posed['vertices'][0, i, 2]))
    #
    #     for i in range(fish.faces.size(0)):
    #         modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i, 0] + 1, fish.faces[i, 1] + 1,
    #                                                                 fish.faces[i, 2] + 1))

    # simulate pyrenderer
    # K = torch.tensor([[7.67578125, 0., -1., 0.],
    #                                [ 0.,15.11538462,1.,0.],
    #                                [ 0.,0.,-1.00010001,-0.100005],
    #                                [ 0.,0.,-1.,0.]]).repeat(2,1,1)
    # projected_homo = - perspective_projection_homo(transformed_points.repeat(2, 1, 1), K.permute(0,2,1))[frame[0]]
    # ix = (torch.minimum(torch.maximum(projected_homo[:, 1].int(), torch.tensor(0)), torch.tensor(1039))).tolist()
    # iy = (torch.minimum(torch.maximum(projected_homo[:, 0].int(), torch.tensor(0)), torch.tensor(2047))).tolist()
    # img[ix, iy] = np.array([0, 0, 255])

    # visualize keypoints
    if kpts is not None:
        for i in range(kpts[0].size(1)):
            img[int(kpts[0][frame[0],i,1]), int(kpts[0][frame[0],i,0])] = np.array([255, 0, 0])
        # for i in range(kpts[1].size(1)):
        #     img[int(kpts[1][frame[0],i,1]), int(kpts[1][frame[0],i,0])] = np.array([255, 125, 0])

    if bboxs is not None:
        #print('draw bbox')
        img[bboxs[1].item():bboxs[3].item(), bboxs[0].item()] = np.array([255, 255, 0])
        img[bboxs[1].item():bboxs[3].item(), bboxs[2].item()] = np.array([255, 255, 0])
        img[bboxs[1].item(), bboxs[0].item():bboxs[2].item()] = np.array([255, 255, 0])
        img[bboxs[3].item(), bboxs[0].item():bboxs[2].item()] = np.array([255, 255, 0])

    # renderer = Renderer(focal[0] * 500, center[0], img_w=2048, img_h=1040, faces=fish.faces)
    # init_points = fish_posed['vertices']# torch.einsum('bij,bkj->bki', R[frame], fish_posed['vertices']) + T[frame]
    # init_points = init_points - init_points[:,[0],:] - torch.tensor([[0,0,0.1]])
    # img_pose, _ = renderer(points - torch.tensor([[0,0,0.2]]), rot, [0,0,0.00], img)  # T [0.10, -0.06, 0]
    # img_pose = img_pose.astype(np.uint8)
    img_pose = img.astype(np.uint8)

    # img_model, _ = renderer(points - torch.tensor([[0,0,0.2]]), rot, [0,0,0.00], np.zeros((1040,2048,3)))
    # img_model = img_model.astype(np.uint8)


    return img_pose, img_pose #img_model


# def render_mesh(bird, pose_est, bone_est, scale_est=1, camera_t=torch.tensor([[2, -7, 35]]).float()):
#     # Background
#     background = torch.ones([368, 368, 3]).float()
#
#     # Camera parameters
#     # camera_t = torch.tensor([[2, -7, 35]]).float()
#     camera_center = torch.tensor([[368 // 2, 368 // 2]]).float()
#     focal_length = 1000.1
#
#     # Bird Mesh
#     bird_output = bird(pose_est[:, 0:3], pose_est[:, 3:], bone_est, scale_est)
#     vertex_posed = bird_output['vertices']
#     # vertex_posed += torch.tensor([[[0,10,8]]]).float()
#
#     # Rendering
#     renderer = Renderer(focal_length=focal_length, center=(184, 184), img_w=368, img_h=368, faces=bird.faces)
#     img_1, _ = renderer(vertex_posed[0].clone().numpy(), np.eye(3), camera_t[0].clone().numpy(),
#                         background.clone().numpy())
#
#     # Render: Second View
#     aroundy = cv2.Rodrigues(np.array([0, np.radians(45.), 0]))[0]
#     center = vertex_posed.numpy()[0].mean(axis=0)
#     rot_vertices = np.dot((vertex_posed.numpy()[0] - center), aroundy) + center
#     img_2, _ = renderer(rot_vertices, np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())
#
#     # Render: Third View
#     aroundy = cv2.Rodrigues(np.array([0, np.radians(-45.), 0]))[0]
#     center = vertex_posed.numpy()[0].mean(axis=0)
#     rot_vertices = np.dot((vertex_posed.numpy()[0] - center), aroundy) + center
#     img_3, _ = renderer(rot_vertices, np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())
#
#     return [img_1, img_2, img_3]