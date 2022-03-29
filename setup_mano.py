from __future__ import print_function, unicode_literals
import argparse
import os
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import hashlib
import shutil
import pickle

def replace(file_path, line_ids, new_lines):
    """ Replace a line in a given file with a new given line. """
    line_ids = [i - 1 for i in line_ids]
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for i, line in enumerate(old_file):
                if i in line_ids:
                    new_file.write(new_lines[line_ids.index(i)] + '\n')
                else:
                    new_file.write(line)

    #Remove original file
    remove(file_path)

    #Move new file
    move(abs_path, file_path)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _patch_mano_loader():
    file = 'mano/webuser/smpl_handpca_wrapper_HAND_only.py'

    replace(file, *zip(*
        [
            (23, '    import pickle'),
            (26, '    from mano.webuser.posemapper import posemap'),
            (66, '    from mano.webuser.verts import verts_core'),
            (92, '    smpl_data[\'fullpose\'] = ch.array(smpl_data[\'fullpose\'].r)'),
            (74, '        smpl_data = pickle.load(open(fname_or_dict, \'rb\'))')
        ]
    ))

    file = 'mano/webuser/verts.py'

    replace(file, *zip(*
                       [
                           (29, 'import mano.webuser.lbs as lbs'),
                           (30, 'from mano.webuser.posemapper import posemap'),
                       ]
                       ))

    file = 'mano/webuser/lbs.py'

    replace(file, *zip(*
                       [
                           (27, 'from mano.webuser.posemapper import posemap'),
                           (38, '        from mano.webuser.posemapper import Rodrigues'),
                           (77, '    v = v[:,:3]'),
                           (78, '    for tp in [744, 333, 444, 555, 672]:  # THUMB, INDEX, MIDDLE, RING, PINKY'),
                           (79, '        A_global.append(xp.vstack((xp.hstack((np.zeros((3, 3)), v[tp, :3].reshape((3, 1)))), xp.array([[0.0, 0.0, 0.0, 1.0]]))))')
                       ]
                       ))


def patch_files():
    _patch_mano_loader()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import needed files from MANO repository.')
    parser.add_argument('mano_path', type=str, help='Path to where the original MANO repository is located.')
    parser.add_argument('--clear', action='store_true', help='Util call for me to remove mano files before committing.')
    args = parser.parse_args()

    # files we attempt to copy from the original mano repository
    files_needed = [
        'models/MANO_RIGHT.pkl',
        'models/MANO_LEFT.pkl',
        'webuser/verts.py',
        'webuser/posemapper.py',
        'webuser/lbs.py',
        'webuser/smpl_handpca_wrapper_HAND_only.py',
        '__init__.py',
        'webuser/__init__.py'
    ]


    if args.clear:
        if os.path.exists('./mano'):
            shutil.rmtree('./mano')
        print('Repository cleaned.')
        exit()

    # check input files
    files_copy_to = [os.path.join('mano', f) for f in files_needed]
    files_needed = [os.path.join(args.mano_path, f) for f in files_needed]
    assert all([os.path.exists(f) for f in files_needed]), 'Could not find one of the needed MANO files in the directory you provided.'

    # coursely check content
    hash_ground_truth = [
        'fd5a9d35f914987cf1cc04ffe338caa1',
        '11aeaca8631aa20db964d4eba491e885',
        '998c30fd83c473da6178aa2cb23b6a5d',
        'c5e9eacc535ec7d03060e0c8d6f80f45',
        'd11c767d5db8d4a55b4ece1c46a4e4ac',
        '5afc7a3eb1a6ce0c2dac1483338a5f58',
        'fd4025c7ee315dc1ec29ac94e4105825'
    ]
    # print([md5(f) for f, gt in zip(files_needed, hash_ground_truth)])
    assert all([md5(f) == gt for f, gt in zip(files_needed, hash_ground_truth)]), 'Hash sum of provided files differs from what was expected.'

    # copy files
    if not os.path.exists('mano'):
        os.mkdir('mano')
    if not os.path.exists('mano/models'):
        os.mkdir('mano/models')
    if not os.path.exists('mano/webuser'):
        os.mkdir('mano/webuser')
    for a, b in zip(files_needed, files_copy_to):
        shutil.copy2(a, b)
        if 'MANO_LEFT' in b:
            smpl_data = pickle.load(open(b, 'rb'))
            smpl_data['shapedirs'] *= -1
            pickle.dump(smpl_data, open(b, 'wb'), protocol=2)


    # some files need to be modified
    patch_files()

