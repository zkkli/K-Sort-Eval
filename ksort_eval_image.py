import os
import json
import numpy as np
import argparse
import logging
import math
import base64
import random
from openai import OpenAI
from trueskill import TrueSkill
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
from generate_image import generate_image


def get_args_parser():
    parser = argparse.ArgumentParser(description="K-Sort Eval", add_help=False)
    parser.add_argument('--dataset_path', required=True,
                        help='path to local dataset')
    parser.add_argument('--log_path', default="log_results", 
                        help='path to local dataset')
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--exp_name", default='exp_name', type=str, help="Experiment Name")
    parser.add_argument("--alpha", default='0.5', type=float, help="alpha")
    parser.add_argument("--sigmoid_k", default='5.0', type=float, help="sigmoid_k")

    return parser


def load_trueskill_from_json(score_list_path):
    with open(score_list_path, 'r') as file:
        sorted_score_list = json.load(file)

    model_ratings = {}
    std_list = []
    for model_info in sorted_score_list['sorted_score_list']:
        model_name = model_info['🤖 Model']
        mean, std = map(float, model_info['⭐ Score (μ/σ)'].split(" ")[1][1:-1].split("/"))
        model_ratings[model_name] = trueskill_env.create_rating(mu=mean, sigma=std)
        std_list.append(std)

    std_mean = np.mean(std_list)
    
    return model_ratings, std_mean


def process_subfolder(subfolder_path):
    # Step 1: Read JSON and image files
    json_file = os.path.join(subfolder_path, "result.json")
    images = [os.path.join(subfolder_path, f"{i}.jpg") for i in range(0, 4)]

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found in {subfolder_path}")

    with open(json_file, "r") as file:
        data = json.load(file)

    prompt = data["prompt"][0]
    prior_rank = data["img_rank"]
    models = data["models_name"]
    models = [models.split("_")[1] for models in models]
    models = ['LCM-v1.5' if model == 'LCM' else model for model in models]
    models = ['Dreamshaper-xl' if model == 'Dreamshaper-xl-turbo' else model for model in models]

    # Step 2: Generate a new image
    save_path = 'generated_image.jpg'
    generate_image(prompt=prompt, save_path=save_path)

    with open(save_path, "rb") as img_file:
        generated_image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Step 3: Use OpenAI ChatGPT to sort the images
    image_data = []
    for img_path in images:
        with open(img_path, "rb") as img_file:
            image_data.append(base64.b64encode(img_file.read()).decode("utf-8"))

    # Add the generated image
    image_data.append(generated_image_data)
    # Shuffle the images
    indexed_data = list(enumerate(image_data))
    random.shuffle(indexed_data)
    shuffled_indices, shuffled_images = zip(*indexed_data)

    # Prepare the prompt for VLM
    vlm_prompt = (
        "You will be given 5 images. Your task is to rank them from best to worst based on visual aesthetics (50%) and alignment with the given description (50%): " + prompt + "."
        "Aesthetics includes photorealism (30%), light and shadow (10%), and absence of artifacts (10%), and Alignment includes entity matching (30%) and style matching (20%)."
        "You MUST respond with only the sorted image names in the strict format: \"Image 1, Image 2, Image 3, Image 4, Image 5\"."
        "Do NOT include any explanations, line breaks, or additional text. Only output the final ranking in a single line. A ranking MUST always be returned and rejection is not allowed."
    )

    client = OpenAI(
        api_key = os.getenv('OPENAI_API_KEY'),
    )    
    messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": vlm_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type":"text", "text": "Here are the images:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shuffled_images[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shuffled_images[1]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shuffled_images[2]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shuffled_images[3]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shuffled_images[4]}"}}
                ]
            }
        ]
    completion = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
    )
    sorted_images = completion.choices[0].message.content

    return models, prior_rank, sorted_images, shuffled_indices


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    log_folder = args.log_path
    log_file = os.path.join(log_folder, f"{args.exp_name}.log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if os.path.exists(log_file):
        user_input = input(f"Log file '{log_file}' already exists. Do you want to delete it? (y/n): ").strip().lower()
        if user_input == 'y':
            os.remove(log_file)
            print("Log file deleted. Proceeding with the program...")
        else:
            print("User chose not to delete the log file. Exiting the program.")
            exit()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(args)

    benchmark_ratings, std_mean = load_trueskill_from_json(score_list_path)
    rating = trueskill_env.create_rating()

    used_data = []
    for itr in range(150):
        # Dynamic matching
        uncertainty = np.abs(np.mean(avg_capacitys, axis=1) - rating.mu)
        diversity = total_overlaps
        alpha = args.alpha
        active_score = uncertainty + alpha * diversity
        if used_data:
            active_score[used_data] = float('inf')
        idx = np.argmin(active_score)
        subfolder = subfolders[idx]
        used_data.append(idx)

        logger.info(f"Itr: {itr}, Subfolder: {subfolder}")

        subfolder_path = os.path.join(main_folder, subfolder)
        models, prior_rank, sorted_images, shuffled_indices = process_subfolder(subfolder_path)

        image_order = [int(x.split()[1]) for x in sorted_images.split(', ')]
        shuffled_observe_rank = [0] * len(image_order)
        for rank, image_index in enumerate(image_order, start=1):
            shuffled_observe_rank[image_index - 1] = rank - 1

        observe_rank = [0] * len(image_order)
        for shuffled_idx, rank in enumerate(shuffled_observe_rank):
            original_idx = shuffled_indices[shuffled_idx] 
            observe_rank[original_idx] = rank 

        opp_ratings = [benchmark_ratings[model] for model in models]

        rho, _ = spearmanr(prior_rank, observe_rank[:-1])
        def sigmoid(rho, k=5.0, rho0=0.0):
            return 1 / (1 + np.exp(-k * (rho - rho0)))
        lamda = sigmoid(rho, k=args.sigmoid_k)
        lamda_square = lamda**2 / (lamda**2 + (1 - lamda)**2)

        all_ratings = [[r] for r in opp_ratings] + [[rating]]
        updated_ratings = trueskill_env.rate(all_ratings, observe_rank)
        new_rating = updated_ratings[-1][0]

        # Posterior correction
        new_mu = lamda * new_rating.mu + (1 - lamda) * rating.mu
        new_sigma = math.sqrt(lamda_square * new_rating.sigma**2 + (1 - lamda_square) * rating.sigma**2)

        rating = trueskill_env.create_rating(mu=new_mu, sigma=new_sigma)

        logger.info(f"Prior Rank: {prior_rank}")
        logger.info(f"Observe Rank: {observe_rank[:-1]}")
        logger.info(f"Spearman Coef.: {rho}, After Sigmoid: {lamda}")
        logger.info(f"μ: {rating.mu}, σ: {rating.sigma}")
        logger.info("=" * 60)

        if rating.sigma < 0.75:
            final_score = rating.mu - 3*std_mean
            logger.info(f"Final Score: {final_score:.2f}")
            logger.info("=" * 60)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser('K-Sort Eval', parents=[get_args_parser()])
    args = parser.parse_args()

    snapshot_download(
        repo_id="ksort/K-Sort-Eval",
        repo_type="dataset",
        allow_patterns=["Image/**", "Image_meta/**"], 
        local_dir=args.dataset_path,
        local_dir_use_symlinks=False
    )

    main_folder = os.path.join(args.dataset_path, 'Image')
    score_list_path = os.path.join(args.dataset_path, 'Image_meta/Arena-Leaderboard-Image.json')
    avg_capacitys = np.load(os.path.join(args.dataset_path, 'Image_meta/avg_capacitys.npy'))
    total_overlaps = np.load(os.path.join(args.dataset_path, 'Image_meta/total_overlaps.npy'))
    subfolders = np.load(os.path.join(args.dataset_path, 'Image_meta/subfolders.npy'))

    trueskill_env = TrueSkill()

    main(args)
