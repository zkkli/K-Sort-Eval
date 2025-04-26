def generate_image(prompt, save_path='generated_image.jpg'):
    """
    Generate an image based on a text prompt.
    Keep 512x512 for fair comparison.

    TODO: Implement your own model generation logic in the marked section below.
    """

    # ================== ✨ User should implement their model call here ✨ ==================
    # Replace this part with your model inference code.
    """
    width = 512
    height = 512
    image = ...
    image.save(save_path)
    """

    # Example: Local model generation using FluxPipeline
    """
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    generator = torch.Generator("cpu").manual_seed(0)
    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=28,
        max_sequence_length=512,
        generator=generator
    ).images[0]

    image.save(save_path)
    """
