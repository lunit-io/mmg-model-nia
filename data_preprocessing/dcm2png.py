import sys
import os.path
import argparse

import numpy as np
import pydicom as dicom
import cv2
from collections.abc import Sequence
from multiprocessing import Process

def dcm2png(files, args, tid):
    # set destination roots
    dst_root_img = args.dst_root

    # do the job
    num_fail = 0
    num_file = len(files)
    for i, f in enumerate(files):

        # define destination paths
        dst_dir_img = os.path.join(dst_root_img, os.path.dirname(f))
        dst_path_img = os.path.join(dst_dir_img, os.path.basename(f))
        dst_path_img = dst_path_img[:-4] + '.png'
        print(dst_dir_img + '/' + f)

        if os.path.isfile(dst_path_img):
            continue

        # convert dicom pixel arrays to images
        try:
            # read dicom to get an image
            dcm = dicom.read_file(os.path.join(args.dcm_root, f))
            image = get_image(dcm)
            image = image.astype(np.float32)

            # adjust slope and bias
            try:
                slope = np.float32(dcm.RescaleSlope)
                if slope != 1:
                    assert slope > 0, 'wrong rescale slope ({})'.format(slope)
                    print('rescale slope = {} ({})'.format(slope, f))
                    image *= slope
                bias = np.float32(dcm.RescaleIntercept)
                if bias != 0:
                    print('rescale intercept = {} ({})'.format(slope, f))
                    image += bias
            except:
                pass

            # rescale
            center = np.float32(dcm.WindowCenter)
            width = np.float32(dcm.WindowWidth)
            if isinstance(center, Sequence) or isinstance(center, np.ndarray):
                center = center[0]
            if isinstance(width, Sequence) or isinstance(width, np.ndarray):
                width = width[0]
            pixel_min = center - width / 2 + 1
            pixel_max = center + width / 2
            image = (image - pixel_min) / (pixel_max - pixel_min)
            image[image < 0] = 0
            image[image > 1] = 1
            image_uint8 = (image * 255).astype(np.uint8)

            # save image
            os.makedirs(dst_dir_img, exist_ok=True)
            cv2.imwrite(dst_path_img, image_uint8, [cv2.IMWRITE_PNG_COMPRESSION, args.comp_level])


        except AssertionError as err:
            print('warning) skip {}: {}'.format(f, err.args[0]))
            num_fail += 1

        except:
            print('warning) skip {}: {}'.format(f, sys.exc_info()))
            num_fail += 1
    # print status
    if (i + 1) % 100 == 0:
        print('Thread {}:, {:d}/{:d}) images saved'.format(tid, i + 1, num_file))


    print('Thread {} : successful : {}, failed : {}'.format(tid, num_file - num_fail, num_fail))

def get_image(dcm):
    im = dcm.pixel_array
    assert len(im.shape) == 2, 'wrong shape ({})'.format(im.shape)
    assert im.shape[1] > 100, 'too small width ({})'.format(im.shape[1])
    assert im.shape[0] > 100, 'too small height ({})'.format(im.shape[0])
    return im

def get_dicom_files(root):
    flist = []
    if not root.endswith(os.sep):
        root += os.sep
    for abs_dir, _, files in os.walk(root):
        for f in files:
            if f.endswith('.dcm'):
                flist.append(os.path.join(abs_dir, f)[len(root):])
    return sorted(flist)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='script to convert dicom pixel arrays to images')
    parser.add_argument('--dcm-root', default='/path/to/dicom/', metavar='DIR', type=str,
                        help='root to the dbt dicom files')
    parser.add_argument('--dst-root', default='/path/for/dst_png/', metavar='DIR', type=str,
                        help='destination directory to the png files, of which the directory hierarchy preserved')
    parser.add_argument('--comp-level', default=1, metavar='N', type=int,
                        help='png compression level from 0 to 9 (higher level, smaller size)')
    parser.add_argument('--n-threads', default=16, metavar='N', type=int,
                        help='number of threads to convert images')
    args = parser.parse_args()
    assert args.dcm_root != args.dst_root

    # get dicom file list
    print('list dicom files')
    files = get_dicom_files(args.dcm_root)
    print('{} dicom files found'.format(len(files)))

    print('convert dicom pixel arrays to images and write results to ' + args.dst_root)

    n_threads = args.n_threads

    file_split = [files[i::n_threads] for i in range(n_threads)]
    threads = []
    for tid in range(n_threads):
        thread = Process(target=dcm2png, args=(file_split[tid], args, tid))
        thread.start()
        threads.append(thread)

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        for thread in threads:
            thread.terminate()
            thread.join()
        
if __name__ == '__main__':
    main()
