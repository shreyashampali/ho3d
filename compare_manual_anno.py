import os
import numpy as np
import argparse
from os.path import join
from utils.vis_utils import load_pickle_data

#  Paths and Parameters


fingerTipIds = [20, 19, 18, 17, 16]


def calculateMSEPerFrame(manualAnno, annotFile):
    seq = annotFile.split('_')[0]
    fID = annotFile.split('_')[1]

    optDir = (os.path.join(args.base_path, 'train', seq+'1',
                           'meta',
                           fID + '.pkl'))


    if not os.path.exists(optDir):
        print('[INFO] Skipping sequence %s file ID %s as it is part of test set'%(seq,fID))
        return np.nan

    optPickData = load_pickle_data(optDir)

    rightHandJointLocs = optPickData['handJoints3D'][fingerTipIds]

    mse = np.mean(np.linalg.norm(manualAnno - rightHandJointLocs, axis=1))

    return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare with manual annotations')
    parser.add_argument('base_path', type=str,
                        help='Path to where the HO3D dataset is located.')

    args = parser.parse_args()

    annotSaveDir = join(args.base_path, 'manual_annotations')
    annotFiles = sorted(os.listdir(annotSaveDir))

    mseSum = []
    for annotFile in annotFiles:
        annotPickData = np.load(os.path.join(annotSaveDir, annotFile))

        mseSum.append(calculateMSEPerFrame(annotPickData, annotFile[:-4]))

    mseSum = np.array(mseSum, dtype=np.float32)

    print('Number of samples = %d'%(mseSum.shape[0] - np.sum(np.isnan(mseSum))))
    print('Average MSE - {}mm, Standard Deviation - {}mm'.format(np.nanmean(mseSum)*1000,
                                                                        np.std(mseSum[np.logical_not(np.isnan(mseSum))])*1000))
