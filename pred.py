from __future__ import print_function, unicode_literals
import argparse
from tqdm import tqdm

from utils.vis_utils import *


def main(base_path, pred_out_path, pred_func, version, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()

    # read list of evaluation files
    with open(os.path.join(base_path, set_name+'.txt')) as f:
        file_list = f.readlines()
    file_list = [f.strip() for f in file_list]

    assert len(file_list) == db_size(set_name, version), '%s.txt is not accurate. Aborting'%set_name

    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name, version))):
        if idx >= db_size(set_name, version):
            break

        seq_name = file_list[idx].split('/')[0]
        file_id = file_list[idx].split('/')[1]

        # load input image
        img = read_RGB_img(base_path, seq_name, file_id, set_name)
        aux_info = read_annotation(base_path, seq_name, file_id, set_name)

        # use some algorithm for prediction
        xyz, verts = pred_func(
            img,
            aux_info
        )

        # simple check if xyz and verts are in opengl coordinate system
        if np.all(xyz[:,2]>0) or np.all(verts[:,2]>0):
            raise Exception('It appears the pose estimates are not in OpenGL coordinate system. Please read README.txt in dataset folder. Aborting!')

        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def pred_template(img, aux_info):
    """ Predict joints and vertices from a given sample.
        img: (640, 480, 3) RGB image.
        aux_info: dictionary containing hand bounding box, camera matrix and root joint 3D location
    """
    # TODO: Put your algorithm here, which computes (metric) 3D joint coordinates and 3D vertex positions
    xyz = np.zeros((21, 3))  # 3D coordinates of the 21 joints
    verts = np.zeros((778, 3)) # 3D coordinates of the shape vertices
    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the HO3D dataset is located.')
    parser.add_argument('--out', type=str, default='pred.json',
                        help='File to save the predictions.')
    parser.add_argument('--version', type=str, choices=['v2', 'v3'],
                        help='version number')
    args = parser.parse_args()

    # call with a predictor function
    main(
        args.base_path,
        args.out,
        pred_func=pred_template,
        set_name='evaluation',
        version=args.version
    )

