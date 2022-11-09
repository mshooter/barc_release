# === CODE BASED ON BARC === 
import torch 
from barc_release.stacked_hourglass.datasets.stanext24 import StanExt

def get_stage1_parameters(model, img_loader, device):
    """
    Gets the initial shape, pose, translation and focal length of the frames.
    Parameters:
        model (nn.Module): barc ML model 
        frames (list): a list of image frames from different point of views
    """
   # information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(StanExt.DATA_INFO.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(StanExt.DATA_INFO.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(StanExt.DATA_INFO.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(StanExt.DATA_INFO.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(StanExt.DATA_INFO.flength_std).float().to(device)
        } 
      
    # so you should add the outputs to 1 => for the total image frames  - check script barc_cfg_visualization.yaml 
    for i, (input, target_dict) in enumerate(img_loader): 
        print(i) # len(img_loader) / batch_size
        batch_size = input.shape[0]
        input = input.float().to(device)
        with torch.no_grad():
            # output: contains all output from model_image_to_3d  - keys = 'flength', 'trans', 'pose', 'normflow_z', 'keypoints_norm', 'keypoints_scores'
            # output_unnorm: same as output, but normalizations are undone
            # output_reproj: smal output and reprojected keypoints as well as silhouette
            output, output_unnorm, output_reproj = model(input, norm_dict=norm_dict)

            # shape (beta), pose, trans, camera this should be fixed 
            betas = output_reproj['betas'] # (bs, 30) 
            print(betas.shape)
            betas_limbs = output_reproj['betas_limbs'] # (bs, 7)
            pose = output_unnorm['pose_rotmat'] # (bs, 35, 3, 3)
            trans = output_unnorm['trans'] # (bs, 3)
            flength = output_unnorm['flength'] # (bs, 1)

            #return {'betas': betas, 
            #        'betas_limbs': betas_limbs, 
            #        'pose': pose, 
            #        'trans': trans,
            #        'flength': flength}
            
        
