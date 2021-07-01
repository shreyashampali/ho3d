"""
Visualize the projections in published HO-3D dataset
"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
from copy import deepcopy
import open3d

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])

try:
    import matplotlib.pyplot as plt
except:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import chumpy as ch
except:
    install('chumpy')
    import chumpy as ch


try:
    import pickle
except:
    install('pickle')
    import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D

MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'

# mapping of joints from MANO model order to simple order(thumb to pinky finger)
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m


if __name__ == '__main__':

    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("ho3d_path", type=str, help="Path to HO3D dataset")
    ap.add_argument("ycbModels_path", type=str, help="Path to ycb models directory")
    ap.add_argument("-split", required=False, type=str,
                    help="split type", choices=['train', 'evaluation'], default='train')
    ap.add_argument("-seq", required=False, type=str,
                    help="sequence name")
    ap.add_argument("-id", required=False, type=str,
                    help="image ID")
    ap.add_argument("-visType", required=False,
                    help="Type of visualization", choices=['open3d', 'matplotlib'], default='matplotlib')
    args = vars(ap.parse_args())

    baseDir = args['ho3d_path']
    YCBModelsDir = args['ycbModels_path']
    split = args['split']

    # some checks to decide if visualizing one single image or randomly picked images
    if args['seq'] is None:
        args['seq'] = random.choice(os.listdir(join(baseDir, split)))
        runLoop = True
    else:
        runLoop = False

    if args['id'] is None:
        args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
    else:
        pass

    if args['visType'] == 'matplotlib':
        o3dWin = Open3DWin()


    while(True):
        seqName = args['seq']
        id = args['id']

        # read image, depths maps and annotations
        img = read_RGB_img(baseDir, seqName, id, split)
        depth = read_depth_img(baseDir, seqName, id, split)
        anno = read_annotation(baseDir, seqName, id, split)

        if anno['objRot'] is None:
            print('Frame %s in sequence %s does not have annotations'%(args['id'], args['seq']))
            if not runLoop:
                break
            else:
                args['seq'] = random.choice(os.listdir(join(baseDir, split)))
                args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
                continue

        # get object 3D corner locations for the current pose
        objCorners = anno['objCorners3DRest']
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

        # get the hand Mesh from MANO model for the current pose
        if split == 'train':
            handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])

        # project to 2D
        if split == 'train':
            handKps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
        else:
            # Only root joint available in evaluation split
            handKps = project_3D_points(anno['camMat'], np.expand_dims(anno['handJoints3D'],0), is_OpenGL_coords=True)
        objKps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)

        # visualize the hand contact map
        if 'handVertContact' in anno.keys() and args['visType'] == 'matplotlib' and split == 'train':
            contactMesh = deepcopy(handMesh)
            contactMesh.fullpose[:] = contactMesh.fullpose.r * 0
            contactMesh.trans[:] = np.array([0., 0., 1.0])
            contactCols = np.zeros((contactMesh.r.shape[0], 3))
            contactCols[:, 2] = anno['handVertContact']
            contactMeshO3d = open3d.geometry.TriangleMesh()
            contactMeshO3d.vertices = open3d.utility.Vector3dVector(np.copy(contactMesh.r))
            contactMeshO3d.triangles = open3d.utility.Vector3iVector(contactMesh.f)
            contactMeshO3d.vertex_colors = open3d.utility.Vector3dVector(contactCols)

            contactMapHand = o3dWin.capture_view([contactMeshO3d], 'utils/hand_r.txt')

        # Visualize
        if args['visType'] == 'open3d':
            # open3d visualization

            if not os.path.exists(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')):
                raise Exception('3D object models not available in %s'%(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')))

            # load object model
            objMesh = read_obj(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))

            # apply current pose to the object model
            objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

            # show
            if split == 'train':
                open3dVisualize([handMesh, objMesh], ['r', 'g'])
            else:
                open3dVisualize([objMesh], ['r', 'g'])



        elif args['visType'] == 'matplotlib':

            # draw 2D projections of annotations on RGB image
            if split == 'train':
                imgAnno = showHandJoints(img, handKps[jointsMapManoToSimple])
            else:
                # show only projection of root joint in evaluation split
                imgAnno = showHandJoints(img, handKps)
                # show the hand bounding box
                imgAnno = show2DBoundingBox(imgAnno, anno['handBoundingBox'])
            imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)

            # create matplotlib window
            fig = plt.figure(figsize=(2, 3))
            figManager = plt.get_current_fig_manager()
            figManager.resize(*figManager.window.maxsize())

            # show RGB image
            ax0 = fig.add_subplot(2, 3, 1)
            ax0.imshow(img[:, :, [2, 1, 0]])
            ax0.title.set_text('RGB Image')

            # show depth map
            ax1 = fig.add_subplot(2, 3, 2)
            im = ax1.imshow(depth)
            ax1.title.set_text('Depth Map')

            # show contact map
            if 'handVertContact' in anno.keys():
                ax3 = fig.add_subplot(2, 3, 3)
                im = ax3.imshow(contactMapHand)
                ax3.title.set_text('Contact Map')

            # show 3D hand mesh
            ax3 = fig.add_subplot(2, 3, 4, projection="3d")
            if split=='train':
                plot3dVisualize(ax3, handMesh, flip_x=False, isOpenGLCoords=True, c="r")
            ax3.title.set_text('Hand Mesh')

            # show 2D projections of annotations on RGB image
            ax4 = fig.add_subplot(2, 3, 5)
            ax4.imshow(imgAnno[:, :, [2, 1, 0]])
            ax4.title.set_text('3D Annotations projected to 2D')

            plt.show()
        else:
            raise Exception('Unknown visualization type')

        if runLoop:
            args['seq'] = random.choice(os.listdir(join(baseDir, split)))
            args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
        else:
            break