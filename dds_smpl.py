from phalp.models.hmar import HMAR
from omegaconf.dictconfig import DictConfig
import numpy as np
import torch
import os
from phalp.models.hmar.hmr import HMR2018Predictor
from torch import tensor
import trimesh
from PIL import Image
from torchvision.utils import save_image
import sys
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.utils import ico_sphere
from loguru import logger
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    HardPhongShader,
    TexturesVertex
)
import imageio
import wandb
import sd_utils
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse

CACHE_DIR = os.path.expanduser('~/.cache')
class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                          image_size=self.img_size,
                                          light_intensity_ambient=1,
                                          light_intensity_directional=0,
                                          anti_aliasing=False)

class DDS_SMPL():
    def __init__(self) -> None:
        self.dictionary = dictionary = {'seed': 42, 'track_dataset': 'demo', 'device': 'cuda', 'base_tracker': 'PHALP', 'train': False, 'debug': False, 'use_gt': False, 'overwrite': True, 'task_id': -1, 'num_tasks': 100, 'verbose': False, 'detect_shots': False, 'video_seq': None, 'video': {'source': 'example_data/videos/gymnasts_1.mp4', 'output_dir': 'outputs/', 'extract_video': True, 'base_path': None, 'start_frame': -1, 'end_frame': 1300, 'useffmpeg': False, 'start_time': '0s', 'end_time': '10s'}, 'phalp': {'predict': 'TPL', 'pose_distance': 'smpl', 'distance_type': 'EQ_019', 'alpha': 0.1, 'low_th_c': 0.8, 'hungarian_th': 100.0, 'track_history': 7, 'max_age_track': 50, 'n_init': 5, 'encode_type': '4c', 'past_lookback': 1, 'detector': 'vitdet', 'shot': 0, 'start_frame': -1, 'end_frame': 10, 'small_w': 50, 'small_h': 100}, 'pose_predictor': {'config_path': '/home/paulj/.cache/phalp/weights/pose_predictor.yaml', 'weights_path': '/home/paulj/.cache/phalp/weights/pose_predictor.pth', 'mean_std': '/home/paulj/.cache/phalp/3D/mean_std.npy'}, 'ava_config': {'ava_labels_path': '/home/paulj/.cache/phalp/ava/ava_labels.pkl', 'ava_class_mappping_path': '/home/paulj/.cache/phalp/ava/ava_class_mapping.pkl'}, 'hmr': {'hmar_path': '/home/paulj/.cache/phalp/weights/hmar_v2_weights.pth'}, 'render': {'enable': True, 'type': 'TEX_P_HUMAN_MESH', 'up_scale': 2, 'res': 256, 'side_view_each': False, 'metallicfactor': 0.0, 'roughnessfactor': 0.7, 'colors': 'phalp', 'head_mask': False, 'head_mask_path': '/home/paulj/.cache/phalp/3D/head_faces.npy', 'output_resolution': 1440, 'fps': 30, 'blur_faces': False, 'show_keypoints': False}, 'post_process': {'apply_smoothing': True, 'phalp_pkl_path': '_OUT/videos_v0', 'save_fast_tracks': False}, 'SMPL': {'MODEL_PATH': '/home/paulj/.cache/phalp/3D/models/smpl/', 'GENDER': 'neutral', 'MODEL_TYPE': 'smpl', 'NUM_BODY_JOINTS': 23, 'JOINT_REGRESSOR_EXTRA': '/home/paulj/.cache/phalp/3D/SMPL_to_J19.pkl', 'TEXTURE': '/home/paulj/.cache/phalp/3D/texture.npz'}, 'MODEL': {'IMAGE_SIZE': 256, 'SMPL_HEAD': {'TYPE': 'basic', 'POOL': 'max', 'SMPL_MEAN_PARAMS': '/home/paulj/.cache/phalp/3D/smpl_mean_params.npz', 'IN_CHANNELS': 2048}, 'BACKBONE': {'TYPE': 'resnet', 'NUM_LAYERS': 50, 'MASK_TYPE': 'feat'}, 'TRANSFORMER': {'HEADS': 1, 'LAYERS': 1, 'BOX_FEATS': 6}, 'pose_transformer_size': 2048}, 'EXTRA': {'FOCAL_LENGTH': 5000}, 'hmr_type': 'hmr2018', 'shape_edit': 1, 'expand_bbox_shape': [192, 256]}
        self.conf = DictConfig(dictionary)
        self.hmar = HMAR(self.conf)
        self.hmar.to("cuda")
        self.betas = torch.zeros(1,10,device="cuda",requires_grad=True)
        self.smpl_base_model = load_objs_as_meshes(["/home/paulj/projects/collosal/4D-Humans/data/smpl_uv_20200910/smpl_uv.obj"], device="cuda")
        self.num_views = 2
        self.elev = torch.linspace(0, 0, self.num_views)
        self.azim = torch.linspace(-90, 0, self.num_views)
        self.device = "cuda"
        self.lights = DirectionalLights(device=self.device, ambient_color=[[1.0,1.0,1.0]])
        self.R , self.T = look_at_view_transform(dist=2.7, elev=self.elev, azim=self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=self.R, T=self.T)
        self.raster_setttings_soft = RasterizationSettings(
            image_size=512,
            blur_radius=0,
            faces_per_pixel=50,
            perspective_correct=False
        )
        self.sigma = 1e-4
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_setttings_soft
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights
            ),
        )

        pass

    def smpl_render(self):

        smpl = self.hmar.smpl(betas = self.betas, pose2rot=False)

        verts  = smpl.vertices[0]
        self.smpl_base_model._verts_list[0] = verts



        # verts = self.smpl_base_model.verts_packed()
        # N = verts.shape[0]
        # center = verts.mean(0)
        # scale = max((verts - center).abs().max(0)[0])
        # self.smpl_base_model = self.smpl_base_model.offset_verts(-center)
        # self.smpl_base_model = self.scale_verts((1.0 / float(scale)));



        meshes = self.smpl_base_model.extend(self.num_views)


        images_predicted = self.renderer_textured(meshes, cameras=self.cameras, lights=self.lights)

        predicted_rgb = images_predicted[..., :3]
        predicted_rgb = predicted_rgb.permute(0, 3, 1, 2)

        # Rotate images by 180 degrees for visualization

        
        save_image(predicted_rgb, "predicted_rgb.png", nrow=self.num_views)
        return predicted_rgb[None,1]



@hydra.main(config_path="", config_name="default", version_base=None)
def main(args):
    args = OmegaConf.to_container(args, resolve=True)
    print(args)
    args = argparse.Namespace(**args)
    img_name = '/home/paulj/projects/collosal/dds/reference_images/running.png'
    prompt_ref = args.prompt_ref    
    prompt = args.prompt
    guidance_scale = args.guidance_scale
    batch_size = 1

    sd_utils.seed_everything(42)
    wandb.init(
            project="collosal", name=args.name,mode=args.wandb_mode ,config=vars(args),tags=list(args.tags)
        )
    guidance_model = sd_utils.StableDiffusion('cuda', fp16=True, vram_O=False,args=args)
    guidance_model.eval()
    for p in guidance_model.parameters():
        p.requires_grad = False

    dds_smpl = DDS_SMPL()
    #img_ref = load_image_as_tensor(img_name).cuda()
    with torch.no_grad():
        img_ref = dds_smpl.smpl_render()
        latent_ref = guidance_model.encode_imgs(img_ref.half())

    as_latent = True


    with torch.no_grad():
        text_z_ref = torch.cat([guidance_model.get_text_embeds(""), guidance_model.get_text_embeds(prompt_ref), ], dim=0)
        text_z = torch.cat([guidance_model.get_text_embeds(""), guidance_model.get_text_embeds(prompt), ], dim=0)
        # latent_ref = guidance_model.encode_imgs(img_ref)
        # latent_ref = latent_ref.unsqueeze(2)
        # latent[:] = latent_ref

    # optim = torch.optim.Adam([latent], lr=0.01)
    # optim = torch.optim.SGD([latent], lr=0.02)
    optim = torch.optim.SGD([dds_smpl.betas],lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 50, 0.1)
  
    import torch.nn.functional as F

    cos_sim = []
    animation = []
    for i in tqdm(range(args.epochs)):
        optim.zero_grad()
        rendered_smpl_image = dds_smpl.smpl_render().cuda()
        rendered_smpl_image = rendered_smpl_image.half()
        latent = guidance_model.encode_imgs(rendered_smpl_image)


        x = latent.half()
        
        noise_pred, t, noise = guidance_model.predict_noise(text_z, x, guidance_scale=guidance_scale, as_latent=as_latent)
        with torch.no_grad():
            noise_pred_ref, _, _ = guidance_model.predict_noise(text_z_ref, latent_ref, guidance_scale=guidance_scale, as_latent=as_latent, t=t, noise=noise)
             
        w =  (1 - guidance_model.alphas[t])
        if args.loss == "sds":
            grad = w * (noise_pred - noise)
        else:
            grad = w * (noise_pred - noise_pred_ref)
        grad = torch.nan_to_num(grad)
        loss = sd_utils.SpecifyGradient.apply(x, grad)
        loss.backward(retain_graph=True)
        #L2 norm loss on betas
        if args.norm_coeff > 0:
            loss_norm = args.norm_coeff * torch.norm(dds_smpl.betas, p=2)
            loss_norm.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_value_(dds_smpl.betas,2.0)
        optim.step()
        scheduler.step()
        if not args.clip_grad:
            dds_smpl.betas.data = torch.clamp(dds_smpl.betas.data, min=-3.0, max=3.0)
        # log all values of beta
        beta_dict = {}
        for j in range(10):
            beta_dict[f"beta_{j}"] = dds_smpl.betas[0,j].item()
        #print(beta_dict)
        wandb.log(beta_dict)
        if i % 5 == 0:
            with torch.no_grad():
                wandb.log({"cos sim": F.cosine_similarity(noise_pred - noise, noise_pred_ref - noise).mean().item()})
                if as_latent:
                    wandb.log({"result": wandb.Image(rendered_smpl_image[0])})
                    
                    save_image(rendered_smpl_image[0], "result.png")
                    pred = rendered_smpl_image[0].detach()
                    pred = pred.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    animation.append(pred)
                else:
                    plt.imshow(latent[0].detach().cpu().permute(1, 2, 0))
            plt.show()

    # self.all_preds = np.concatenate(self.all_preds, axis=0)
    # imageio.mimwrite(f"video.mp4", self.all_preds, fps=10, quality=10)

    animation = np.stack(animation, axis=0)
    os.makedirs(f"results/{args.name}", exist_ok=True)
    imageio.mimwrite(f"results/{args.name}/animation.mp4", animation, fps=10, quality=10)
    print(dds_smpl.betas)
    print(torch.norm(dds_smpl.betas, p=2))

if __name__ == '__main__':
    main()


# import pyrender

# mask_v = np.load("/home/paulj/projects/collosal/4D-Humans/mask_v.npy")
# im =  Image.open("/home/paulj/projects/collosal/4D-Humans/data/smpl_uv_20200910/texture.png")
# material = trimesh.visual.texture.SimpleMaterial(image=im)
# color_visuals = trimesh.visual.TextureVisuals(uv=smpl_base_model.visual.uv,image=im,material=material)

# mesh_smpl = trimesh.Trimesh(vertices=verts[mask_v].copy(), faces=smpl_base_model.faces, process=False)
# mesh_smpl.visual = color_visuals  
# mesh = pyrender.Mesh.from_trimesh(mesh_smpl)
# scene = pyrender.Scene()
# scene.add(mesh)
# dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=100.0)
# scene.add(dl)
# pyrender.Viewer(scene, use_raymond_lighting=True)
