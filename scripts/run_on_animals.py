# === CODE BASED ON BARC === 
import os 
import argparse
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from barc_release.stacked_hourglass.datasets.stanext24 import StanExt
from barc_release.stacked_hourglass.datasets.imgcrops import ImgCrops
from barc_release.combined_model.train_main_image_to_3d_withbreedrel import do_visual_epoch
from barc_release.combined_model.model_shape_v7 import ModelImageTo3d_withshape_withproj 
from barc_release.configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated
from smalr_mod.optimize import get_stage1_parameters 

def main(args):
    # === LOAD CONFIGS ===
    #   step 1: load default configs
    #   step 2: load updates from .yaml file
    path_config = os.path.join(get_cfg_defaults().barc_dir, 'barc_release', 'configs', args.config)
    update_cfg_global_with_yaml(path_config)
    cfg = get_cfg_global_updated()

    # === SELECT THE HARDWARE DEVICE TO USE FOR INFERENCE. ===
    #if torch.cuda.is_available() and cfg.device=='cuda':
    #    device = torch.device('cuda', torch.cuda.current_device())
    #    torch.backends.cudnn.benchmark = True
    #else:
    device = torch.device('cpu')

    path_model_file_complete = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, args.model_file_complete) 

    # === DISABLE GRADIENT CALCULATIONS. ===
    torch.set_grad_enabled(False)

    # === PREPARE COMPLETE MODEL ===
    complete_model = ModelImageTo3d_withshape_withproj(
        num_stage_comb=cfg.params.NUM_STAGE_COMB, 
        num_stage_heads=cfg.params.NUM_STAGE_HEADS, 
        num_stage_heads_pose=cfg.params.NUM_STAGE_HEADS_POSE, 
        trans_sep=cfg.params.TRANS_SEP, 
        arch=cfg.params.ARCH, 
        n_joints=cfg.params.N_JOINTS, 
        n_classes=cfg.params.N_CLASSES, 
        n_keyp=cfg.params.N_KEYP, 
        n_bones=cfg.params.N_BONES, 
        n_betas=cfg.params.N_BETAS, 
        n_betas_limbs=cfg.params.N_BETAS_LIMBS,
        n_breeds=cfg.params.N_BREEDS, 
        n_z=cfg.params.N_Z, 
        image_size=cfg.params.IMG_SIZE, 
        silh_no_tail=cfg.params.SILH_NO_TAIL, 
        thr_keyp_sc=cfg.params.KP_THRESHOLD, 
        add_z_to_3d_input=cfg.params.ADD_Z_TO_3D_INPUT,
        n_segbps=cfg.params.N_SEGBPS, 
        add_segbps_to_3d_input=cfg.params.ADD_SEGBPS_TO_3D_INPUT, 
        add_partseg=cfg.params.ADD_PARTSEG, 
        n_partseg=cfg.params.N_PARTSEG, 
        fix_flength=cfg.params.FIX_FLENGTH, 
        structure_z_to_betas=cfg.params.STRUCTURE_Z_TO_B, 
        structure_pose_net=cfg.params.STRUCTURE_POSE_NET,
        nf_version=cfg.params.NF_VERSION) 

    # === LOAD TRAINED MODEL ===
    print(path_model_file_complete)
    assert os.path.isfile(path_model_file_complete)
    print('Loading model weights from file: {}'.format(path_model_file_complete))
    checkpoint_complete = torch.load(path_model_file_complete)
    state_dict_complete = checkpoint_complete['state_dict']
    complete_model.load_state_dict(state_dict_complete, strict=False)        
    complete_model = complete_model.to(device)
    # PUT THE MODEL IN EVAL MODE - BC WE ARE GENERATING DATA 
    complete_model.eval()

    # === LOAD DATA === 
    img_dataset = ImgCrops(image_path=args.image_folder_crops, is_train=False, dataset_mode='complete')
    img_loader = DataLoader(img_dataset, 
                            batch_size=cfg.optim.BATCH_SIZE, 
                            shuffle=False,
                            num_workers=args.workers, 
                            pin_memory=True, 
                            drop_last=False)

    # === GET INTIAL SHAPE, POSE, CAMERA, TRANSLATION PARAMETERS === 
    initial_parameters = get_stage1_parameters(complete_model, img_loader, device) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--model-file-complete', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--config', '-cg', default='barc_cfg_test.yaml', type=str, metavar='PATH',
                        help='name of config file (default: barc_cfg_test.yaml within src/configs folder)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--metrics', '-m', metavar='METRICS', default='all',
                        choices=['all', None],
                        help='model architecture')        
    parser.add_argument('--image_folder_crops', '-ifc', type=str, metavar='PATH',
                        help='folder that contains the test image crops') 
    main(parser.parse_args())


