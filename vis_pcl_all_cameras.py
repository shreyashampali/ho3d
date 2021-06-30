import os
import numpy as np
import open3d as o3d
from os.path import join
import cv2
import argparse
from open3d.open3d.geometry import create_rgbd_image_from_color_and_depth
from open3d.open3d.geometry import create_point_cloud_from_rgbd_image

# Paths and Params

# Path to the 'train' directory of HO3D dataset
# sequences_dir = '/media/shreyas/ssd2/Dataset/HO3D_Release_Final/train'


height, width = 480, 640
depth_threshold = 800

multiCamSeqs = [
    'ABF1',
    'BB1',
    'GPMF1',
    'GSF1',
    'MDF1',
    'SB1',
    'ShSu1',
    'SiBF1',
    'SMu4'
]


def inverse_relative(pose_1_to_2):
    pose_2_to_1 = np.zeros((4, 4), dtype='float32')
    pose_2_to_1[:3, :3] = np.transpose(pose_1_to_2[:3, :3])
    pose_2_to_1[:3, 3:4] = -np.dot(np.transpose(pose_1_to_2[:3, :3]), pose_1_to_2[:3, 3:4])
    pose_2_to_1[3, 3] = 1
    return pose_2_to_1


def get_intrinsics(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            ppx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            ppy = float(item.split(':')[1].strip())

    camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    return camMat

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    return dpt


def load_point_clouds(seq, fID):
    pcds = []

    seqDir = os.path.join(args.base_path, seq)
    calibDir = os.path.join(args.base_path, '../', 'calibration', seq, 'calibration')
    cams_order = np.loadtxt(join(calibDir, 'cam_orders.txt')).astype('uint8').tolist()
    depth_scale = 1. / np.loadtxt(join(calibDir, 'cam_0_depth_scale.txt'))
    Ts = []
    for i in range(len(cams_order)):
        T_i = np.loadtxt(join(calibDir, 'trans_{}.txt'.format(i)))
        Ts.append(T_i)

    for i in range(len(cams_order)):
        path_color = join(seqDir+str(cams_order[i]), 'rgb', fID + '.png')
        if not os.path.exists(path_color):
            continue

        color_raw = o3d.io.read_image(path_color)#o3d.geometry.Image(color_raw.astype(np.float32))
        depth_raw = read_depth_img(path_color.replace('rgb', 'depth'))

        depth_raw = o3d.geometry.Image(depth_raw.astype(np.float32))
        K = get_intrinsics(join(calibDir, 'cam_{}_intrinsics.txt'.format(i))).tolist()

        rgbd_image = create_rgbd_image_from_color_and_depth(color_raw,
                                                                        depth_raw,
                                                                        depth_scale=1,
                                                                        convert_rgb_to_intensity=False)
        pcd = create_point_cloud_from_rgbd_image(
            rgbd_image, o3d.camera.PinholeCameraIntrinsic(width=width,
                                                                 height=height,
                                                                 fx=K[0][0],
                                                                 fy=K[1][1],
                                                                 cx=K[0][2],
                                                                 cy=K[1][2]))

        pcd.transform((Ts[i]))
        pcds.append(pcd)

    return pcds


def combine_point_clouds(pcds):
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcd_combined += pcds[point_id]

    return pcd_combined



def manual_registration(annoFiles):
    if args.seq is not None:
        annoFiles = [args.seq]
    for idx, annoFile in enumerate(annoFiles):

        seq = annoFile
        if not os.path.exists(os.path.join(args.base_path,seq+'0','rgb')):
            print('Sequence %s not in %s. Check if path to \'train\' or \'eval\' folder is correct.' % (
            annoFile, args.base_path))
            return

        files = os.listdir(os.path.join(args.base_path,seq+'0','rgb'))
        files = [f[:-4] for f in files]
        if args.fid is not None:
            files = [args.fid]
        for fID in files[:]:
            pcds = load_point_clouds(seq, fID)
            pcd = combine_point_clouds(pcds)

            if len(pcds) == 0:
                print('Sequence %s not in %s. Check if path to \'train\' or \'eval\' folder is correct.'%(annoFile, args.base_path) )
                return

            # o3d.visualization.draw_geometries([pcd])
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--base_path', type=str,
                        help='Path to where the HO3D dataset is located.')
    parser.add_argument('--seq', type=str,
                        help='Sequence name.', required=False)
    parser.add_argument('--fid', type=str,
                        help='File ID', required=False)
    args = parser.parse_args()

    manual_registration(multiCamSeqs)
    # show_manual_annotations()
