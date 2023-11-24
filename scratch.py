# %%
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


@hydra.main(config_path="", config_name="default", version_base=None)
def main(args):
    args = OmegaConf.to_container(args, resolve=True)
    args = argparse.Namespace(**args)
    img_name = '/home/paulj/projects/collosal/dds/reference_images/running.png'
    prompt_ref = args.prompt_ref    
    prompt = args.prompt
    guidance_scale = args.guidance_scale
    batch_size = 1

    sd_utils.seed_everything(0)
    wandb.init(
            project="collosal", name="dds_test",mode="disabled",config=vars(args)
        )
    guidance_model = sd_utils.StableDiffusion('cuda', fp16=True, vram_O=False,args=args)
    guidance_model.eval()
    for p in guidance_model.parameters():
        p.requires_grad = False

    def load_image_as_tensor(image_path):
        # Load the image with PIL
        img = Image.open(image_path).convert("RGB")
        # width, height = img.size

        # Calculate the coordinates for the center crop
        # left = (width - 768) / 2
        # top = (height - 768) / 2
        # right = (width + 768) / 2
        # bottom = (height + 768) / 2
        # img = img.crop((left,top,right,bottom))
        # Convert the image to a PyTorch tensor
        img_tensor = torch.from_numpy(np.array(img)).half().permute(2, 0, 1)

        # Normalize pixel values to range [0, 1]
        img_tensor /= 255

        # Add batch dimension
        batch_img_tensor = img_tensor.unsqueeze(0)

        return batch_img_tensor

    img_ref = load_image_as_tensor(img_name).cuda()
    img_ref.requires_grad = False

    as_latent = True
    latent_size = args.latent_size
    latent = torch.randn(1, 4,1, latent_size, latent_size, requires_grad=True, device="cuda")


    with torch.no_grad():
        text_z_ref = torch.cat([guidance_model.get_text_embeds(""), guidance_model.get_text_embeds(prompt_ref), ], dim=0)
        text_z = torch.cat([guidance_model.get_text_embeds(""), guidance_model.get_text_embeds(prompt), ], dim=0)
        latent_ref = guidance_model.encode_imgs(img_ref)
        latent_ref = latent_ref.unsqueeze(2)
        latent[:] = latent_ref

    # optim = torch.optim.Adam([latent], lr=0.01)
    optim = torch.optim.SGD([latent], lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.9)

    import torch.nn.functional as F

    cos_sim = []
    animation = []
    for i in tqdm(range(args.epochs)):
        optim.zero_grad()
        x = latent.half()

        noise_pred, t, noise = guidance_model.predict_noise(text_z, x, guidance_scale=guidance_scale, as_latent=as_latent)
        with torch.no_grad():
            noise_pred_ref, _, _ = guidance_model.predict_noise(text_z_ref, latent_ref, guidance_scale=guidance_scale, as_latent=as_latent, t=t, noise=noise)
        
        if i % 20 == 0:
            with torch.no_grad():
                wandb.log({"cos sim": F.cosine_similarity(noise_pred - noise, noise_pred_ref - noise).mean().item()})
                if as_latent:
                    wandb.log({"result": wandb.Image(guidance_model.decode_latents(x)[0])})
                    from torchvision.utils import save_image
                    save_image(guidance_model.decode_latents(x)[0], "result.png")
                    pred = guidance_model.decode_latents(x)[0].detach()
                    pred = pred.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    animation.append(pred)
                else:
                    plt.imshow(latent[0].detach().cpu().permute(1, 2, 0))
            plt.show()
        w =  (1 - guidance_model.alphas[t])
        if args.loss == "sds":
            grad = w * (noise_pred - noise)
        else:
            grad = w * (noise_pred - noise_pred_ref)
        grad = torch.nan_to_num(grad)

        loss = sd_utils.SpecifyGradient.apply(x, grad)
        loss.backward()
        optim.step()
        scheduler.step()

    # self.all_preds = np.concatenate(self.all_preds, axis=0)
    # imageio.mimwrite(f"video.mp4", self.all_preds, fps=10, quality=10)

    animation = np.stack(animation, axis=0)
    imageio.mimwrite(f"animation.mp4", animation, fps=10, quality=10)

if __name__ == '__main__':
    main()
