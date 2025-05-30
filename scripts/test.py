

import argparse
import os.path
import os.path
import numpy as np
import pickle as pkl
from distutils.util import strtobool
import torch
from torch import nn
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from collections import OrderedDict
import glob
from dominate import document
from dominate.tags import *

#import sys
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../', 'src'))
from barc_release.stacked_hourglass.datasets.stanext24 import StanExt
from barc_release.combined_model.train_main_image_to_3d_withbreedrel import do_validation_epoch
from barc_release.combined_model.model_shape_v7 import ModelImageTo3d_withshape_withproj  
from barc_release.combined_model.loss_image_to_3d_withbreedrel import Loss
from barc_release.configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated


def main(args):

    # load configs
    #   step 1: load default configs
    #   step 2: load updates from .yaml file
    path_config = os.path.join(get_cfg_defaults().barc_dir, 'src', 'configs', args.config)
    update_cfg_global_with_yaml(path_config)
    cfg = get_cfg_global_updated()

    # Select the hardware device to use for training.
    if torch.cuda.is_available() and cfg.device=='cuda':
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    path_model_file_complete = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, args.model_file_complete) 

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    # prepare complete model
    complete_model = ModelImageTo3d_withshape_withproj(
        num_stage_comb=cfg.params.NUM_STAGE_COMB, num_stage_heads=cfg.params.NUM_STAGE_HEADS, \
        num_stage_heads_pose=cfg.params.NUM_STAGE_HEADS_POSE, trans_sep=cfg.params.TRANS_SEP, \
        arch=cfg.params.ARCH, n_joints=cfg.params.N_JOINTS, n_classes=cfg.params.N_CLASSES, \
        n_keyp=cfg.params.N_KEYP, n_bones=cfg.params.N_BONES, n_betas=cfg.params.N_BETAS, n_betas_limbs=cfg.params.N_BETAS_LIMBS, \
        n_breeds=cfg.params.N_BREEDS, n_z=cfg.params.N_Z, image_size=cfg.params.IMG_SIZE, \
        silh_no_tail=cfg.params.SILH_NO_TAIL, thr_keyp_sc=cfg.params.KP_THRESHOLD, add_z_to_3d_input=cfg.params.ADD_Z_TO_3D_INPUT,
        n_segbps=cfg.params.N_SEGBPS, add_segbps_to_3d_input=cfg.params.ADD_SEGBPS_TO_3D_INPUT, add_partseg=cfg.params.ADD_PARTSEG, n_partseg=cfg.params.N_PARTSEG, \
        fix_flength=cfg.params.FIX_FLENGTH, structure_z_to_betas=cfg.params.STRUCTURE_Z_TO_B, structure_pose_net=cfg.params.STRUCTURE_POSE_NET,
        nf_version=cfg.params.NF_VERSION) 

    # load trained model
    print(path_model_file_complete)
    assert os.path.isfile(path_model_file_complete)
    print('Loading model weights from file: {}'.format(path_model_file_complete))
    checkpoint_complete = torch.load(path_model_file_complete)
    state_dict_complete = checkpoint_complete['state_dict']
    complete_model.load_state_dict(state_dict_complete, strict=False)        
    complete_model = complete_model.to(device)

    # prepare output folder name
    epoch = checkpoint_complete['epoch']
    if 'model_best' in path_model_file_complete: 
        model_file_complete_last = path_model_file_complete.replace('model_best.pth.tar', 'checkpoint.pth.tar')
        if os.path.isfile(model_file_complete_last):
            final_epoch = 'e' + str(torch.load(model_file_complete_last)['epoch'])
        else:
            final_epoch = 'end'
        out_sub_name =  cfg.data.VAL_OPT + '_best_until_' + final_epoch
    else:
        final_epoch = 'e' + str(epoch)
        out_sub_name = cfg.data.VAL_OPT + '_' + final_epoch
    save_imgs_path = os.path.join(os.path.dirname(path_model_file_complete).replace(cfg.paths.ROOT_CHECKPOINT_PATH, cfg.paths.ROOT_OUT_PATH + 'results/'), out_sub_name)
    print('epoch: ' + str(epoch))
    print('best IoU score: ' + str(checkpoint_complete['best_acc']*100))
    print('pck threshold: ' + str(cfg.params.PCK_THRESH))
    print('path to save images: ' + save_imgs_path)
    if not os.path.exists(save_imgs_path):
        os.makedirs(save_imgs_path)
    if args.save_images:
        save_imgs_path_final = save_imgs_path       
    else:
        save_imgs_path_final = None

    # load loss module
    loss_module = Loss(data_info=StanExt.DATA_INFO, nf_version=cfg.params.NF_VERSION).to(device)    

    # Initialise the MPII validation set dataloader
    if cfg.data.VAL_OPT == 'test':
        val_dataset = StanExt(image_path=None, is_train=False, dataset_mode='complete', V12=cfg.data.V12, val_opt='test')
        test_name_list = val_dataset.test_name_list
    elif cfg.data.VAL_OPT == 'val':
        val_dataset = StanExt(image_path=None, is_train=False, dataset_mode='complete', V12=cfg.data.V12, val_opt='val')
        test_name_list = val_dataset.test_name_list
    elif cfg.data.VAL_OPT == 'train':
        val_dataset = StanExt(image_path=None, is_train=True, do_augment='no', dataset_mode='complete', V12=cfg.data.V12)
        test_name_list = val_dataset.train_name_list
    else:
        raise ValueError
    val_loader = DataLoader(val_dataset, batch_size=cfg.optim.BATCH_SIZE, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)  # , drop_last=True    args.batch_size
    len_val_dataset = len(val_dataset)

    # run evaluation 
    valid_string, summary = do_validation_epoch(val_loader, complete_model, loss_module, device,
                                                        StanExt.DATA_INFO,
                                                        weight_dict=None,       # weight_dict,
                                                        acc_joints=StanExt.ACC_JOINTS,
                                                        save_imgs_path=save_imgs_path_final,    # None,     #save_imgs_path,
                                                        metrics=args.metrics, 
                                                        test_name_list=test_name_list,
                                                        render_all=cfg.params.RENDER_ALL,
                                                        pck_thresh=cfg.params.PCK_THRESH, 
                                                        len_dataset=len_val_dataset)

    # calculate and print results
    iou = np.nanmean(summary['acc_sil_2d'])
    pck = np.nanmean(summary['pck'])
    pck_legs = np.nanmean(summary['pck_by_part']['legs'])
    pck_tail = np.nanmean(summary['pck_by_part']['tail'])
    pck_ears = np.nanmean(summary['pck_by_part']['ears'])
    pck_face = np.nanmean(summary['pck_by_part']['face'])
    print('------------------------------------------------')
    print("iou:         {:.2f}".format(iou*100))
    print('                                                ')
    print("pck:         {:.2f}".format(pck*100))
    print('                                                ')
    print("pck_legs:    {:.2f}".format(pck_legs*100))
    print("pck_tail:    {:.2f}".format(pck_tail*100))
    print("pck_ears:    {:.2f}".format(pck_ears*100))
    print("pck_face:    {:.2f}".format(pck_face*100))
    print('------------------------------------------------')
    # save results in a .txt file
    with open(save_imgs_path + "/a_evaluation.txt", "w") as text_file:
        print("iou:         {:.2f}".format(iou*100), file=text_file)
        print("pck:         {:.2f}".format(pck*100), file=text_file)
        print("pck_legs:    {:.2f}".format(pck_legs*100), file=text_file)
        print("pck_tail:    {:.2f}".format(pck_tail*100), file=text_file)
        print("pck_ears:    {:.2f}".format(pck_ears*100), file=text_file)
        print("pck_face:    {:.2f}".format(pck_face*100), file=text_file)
    if not os.path.exists(cfg.paths.ROOT_OUT_PATH + 'results_summarized'):
        os.makedirs(cfg.paths.ROOT_OUT_PATH + 'results_summarized')
    with open(cfg.paths.ROOT_OUT_PATH + 'results_summarized/a_summary-' + save_imgs_path.split('/')[-2] + '-' + save_imgs_path.split('/')[-1] + '.pkl', 'wb') as handle: 
        pkl.dump(summary, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # use as follows:
    # python scripts/test_image_to_3d_withshape.py --workers 12 --save-images True --config barc_cfg_test.yaml --model-file-complete=barc_new_v2/model_best.pth.tar

    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--model-file-complete', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('-cg', '--config', default='barc_cfg_test.yaml', type=str, metavar='PATH',
                        help='name of config file (default: barc_cfg_test.yaml within src/configs folder)')
    parser.add_argument('--save-images', default='True', type=lambda x: bool(strtobool(x)),
                        help='bool indicating if images should be saved')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--metrics', '-m', metavar='METRICS', default='all',
                        choices=['all', None],
                        help='model architecture')
    main(parser.parse_args())
