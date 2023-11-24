for guidance_scale in [50,100,200]:
    for loss in ["sds"]:
        name = f"editaction_{loss}_{guidance_scale}"
        script = f"python dds.py name={name} loss={loss} guidance_scale={guidance_scale} epochs=1000"
        print(script)