# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

from reshape_base_algos.body_retoucher import BodyRetoucher
import time
import cv2
import argparse
import numpy as np
import glob
import tqdm
import os
import json
import shutil
from utils.eval_util import cal_lpips_and_ssim, psnr
from config.test_config import TESTCONFIG, load_config
import toml


def recurve_search(root_path, all_paths, suffix=[]):
    for file in os.listdir(root_path):
        target_file = os.path.join(root_path, file)
        if os.path.isfile(target_file):
            (path, extension) = os.path.splitext(target_file)
            
            if extension in suffix:
                all_paths.append(target_file)
        else:
            recurve_search(target_file, all_paths, suffix)


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test_hp_setting.toml')
    args = parser.parse_args()

    with open(args.config) as f:
        load_config(toml.load(f))

    print('TEST CONFIG: \n{}'.format(TESTCONFIG))
    print("loading model:{}".format(TESTCONFIG.reshape_ckpt_path))

    ret = BodyRetoucher.init(reshape_ckpt_path=TESTCONFIG.reshape_ckpt_path,
                             pose_estimation_ckpt=TESTCONFIG.pose_estimation_ckpt,
                             device=0, log_level='error',
                             log_path='test_log.txt',
                             debug_level=1)
    if ret == 0:
        print('init done')
    else:
        print('init error:{}'.format(ret))
        exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    if os.path.exists(TESTCONFIG.save_dir):
        shutil.rmtree(TESTCONFIG.save_dir)

    os.makedirs(TESTCONFIG.save_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(TESTCONFIG.save_dir, os.path.basename(args.config )))

    if os.path.isfile(TESTCONFIG.src_dir):
        img_list = [TESTCONFIG.src_dir]
    elif os.path.exists(os.path.join(TESTCONFIG.src_dir, "src")):
        img_list = glob.glob("{}/*.*g".format(os.path.join(TESTCONFIG.src_dir, "src")))
    else:
        img_list = []
        recurve_search(TESTCONFIG.src_dir, img_list, suffix=['.png', '.jpg', '.jpeg','.JPG'])

    img_list = sorted(img_list)

    lpips_list = []
    ssim_list = []
    psnr_list = []
    epe_list = []

    src_lpips_list = []
    src_ssim_list = []
    src_psnr_list = []

    bad_sample = []


    pbar = tqdm.tqdm(img_list)
    for src_path in pbar:
        print('image_path: {}'.format(src_path))
        basename = os.path.basename(src_path)

        gt_path = os.path.join(TESTCONFIG.gt_dir, basename)

        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path)
        else:
            gt = None

        base = os.path.splitext(basename)[0]

        src_img = cv2.imread(src_path)
        if src_img is None:
            print('Error: src_img is None')
            continue

        cv2.imwrite(os.path.join(TESTCONFIG.save_dir, base + ".jpg"), src_img)
        t1 = time.time()
        for i, degree in enumerate(np.linspace(-3., 3., 13, endpoint=True)):
            pred, flow = BodyRetoucher.reshape_body(src_img, degree=degree)

            print('time of BodyRetoucher.run: {}ms/frame'.format(int((time.time() - t1) * 1000)))

            if flow is None:
                bad_sample.append(src_path)
                continue

            info = ""

            output_path = os.path.join(TESTCONFIG.save_dir, base + "_warp_{:02d}_{:.1f}.jpg".format(i, degree))
            cv2.imwrite(output_path, pred)

            path_pairs = []

            if BodyRetoucher._debug_level > 0:
                # scale을 여러개 하지 않는이상 zero 영상임
                # path_pairs.append(('x_fusion_map.jpg', base+'_x_fusion_map_{:.1f}.jpg'.format(degree+1)),)
                # path_pairs.append(('y_fusion_map.jpg', base+'_y_fusion_map_{:.1f}.jpg'.format(degree+1)),)

                # 모두 다 똑같음
                if degree == -1.0:
                    path_pairs.append(('flow.jpg', base+'_flow.jpg'))
                    path_pairs.append(('pred.jpg', base+'_pred.jpg'))
                    path_pairs.append(('flow_all.jpg', base+'_flow_all.jpg'))

                # -, 0, +
                # if degree == -1.0:
                #     path_pairs.append(('flow_all.jpg', base+'_flow_all_-.jpg'))
                # elif degree == 0.0:
                #     path_pairs.append(('flow_all.jpg', base+'_flow_all_0.jpg'))
                # elif degree == 1.0:
                #     path_pairs.append(('flow_all.jpg', base+'_flow_all_+.jpg'))

                for src_path, dst_path in path_pairs:
                    if os.path.exists(src_path):
                        os.rename(src_path, dst_path)
                        shutil.move(dst_path, TESTCONFIG.save_dir)

    BodyRetoucher.release()
    print('all done')

