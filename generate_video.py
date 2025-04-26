
def crop_frame(video):
    from PIL import Image

    cropped_video = []
    for frame in video:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)
        width, height = frame.size
        
        if width < 512 or height < 512:
            cropped_frame = frame.resize((512, 512))
        else:
            left = (width - 512) // 2
            top = (height - 512) // 2
            right = left + 512
            bottom = top + 512
            cropped_frame = frame.crop((left, top, right, bottom))

        cropped_video.append(cropped_frame)
    return cropped_video


def generate_video(prompt, save_path='generated_video.mp4'):
    """
    Generate a video based on a text prompt.
    Keep 512x512, 5s, 8FPS for fair comparison.

    TODO: Implement your own model generation logic in the marked section below.
    """

    # ================== ✨ User should implement their model call here ✨ ==================
    # Replace this part with your model inference code.
    """
    width = 512
    height = 512
    duration = 5
    fps = 8
    video = ...
    video = crop_frame(video) # Ensure that the video is 512*512 resolution
    export_to_video(video, save_path, fps=fps)
    """

    # Example: Local model generation using CogVideoXPipeline
    """
    import torch
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=duration*fps+1,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    # Ensure that the video is 512*512 resolution
    video = crop_frame(video)

    export_to_video(video, save_path, fps=fps)
    """
