losses = ["dds","sds"]
guidance_scales = [7.5]
epochs = [50]
prompts = ["A front view of a tall person in T pose  realistic  3D model  photorealistic",
           "A front view of a fat person in T pose  realistic  3D model  photorealistic",
           "A front view of a short person in T pose  realistic  3D model  photorealistic",
           "A front view of a skinny person in T pose  realistic  3D model  photorealistic",
           "A front view of a muscular person in T pose  realistic  3D model  photorealistic"]
#prompts = ["A front view of a fat person in T pose  realistic  3D model  photorealistic"]
norm_coeff = [0.0]
lrs = [1e-2]
clip_grads = [False]
number = 0
for loss in losses:
    for gs in guidance_scales:
        for ep in epochs:
            for prompt in prompts:
                for norm in norm_coeff:
                    for lr in lrs:
                        for clip_grad in clip_grads:
                            name = f"afterclip_{loss}_{gs}_{ep}_{prompt.replace(' ','_')}_norm_{norm}_lr_{lr}_clip_grad_{clip_grad}"
                            script = f"python dds_smpl.py loss=\"{loss}\" guidance_scale={gs} epochs={ep} prompt=\"{prompt}\" name={name} norm_coeff={norm} clip_grad={clip_grad} lr={lr} tags=[after_clip,smpl]"
                            print(script)
                            number += 1

print(f"Total number of jobs: {number}")