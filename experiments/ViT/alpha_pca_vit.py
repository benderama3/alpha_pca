from transformers import AutoFeatureExtractor, AutoModelForImageClassification, ViTFeatureExtractor
from datasets import load_dataset
import torch 
from alpha_pca import AlphaPCA
import matplotlib.pyplot as plt
import os
import argparse

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 5), dpi=300)

parser = argparse.ArgumentParser()

parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--add_optimal_alpha', action='store_true')
parser.add_argument('--project_early', action='store_true')
parser.add_argument('--dataset', type=str, required=False, default="mnist")
parser.add_argument('--device', type=str, required=False, default="cpu")
parser.add_argument('--max_samples', type=int, required=False, default=500)
parser.add_argument('--noise_std', type=float, required=False, default=0.25)
parser.add_argument('--plot_name', type=str, required=False, default="plot.png")
parser.add_argument('--seed', type=int, required=False, default=123)
args = parser.parse_args()

def process_dataset(dataset, max_samples):
    dataset = dataset[:max_samples]
    inputs = [d.convert("RGB") for d in dataset[image_name]]
    labels = [d for d in dataset[label_name]]
    return base_feature_extractor(inputs, return_tensors="pt",)["pixel_values"], labels

def resize_img(dataset):
    dataset = [d.squeeze(0) for d in dataset.split(1, dim=0)]
    return feature_extractor(dataset, return_tensors="pt")["pixel_values"]

def fit(pca, dataset, is_rgb=False):
    n, c, h, w = dataset.size()
    if is_rgb:
        pca.fit(dataset.reshape(n, c*h*w))
    else:
        pca.fit(dataset[:, 0].reshape(n, h*w))
    return pca

def approximate(pca, dataset, is_rgb=False):
    n, c, h, w = dataset.size()
    if is_rgb:
        pca.fit(dataset.reshape(n, c*h*w))
        dataset = pca.approximate(dataset.reshape(n, c*h*w))
        return dataset.reshape(n, c, h, w)
    else:
        pca.fit(dataset[:, 0].reshape(n, h*w))
        dataset = pca.approximate(dataset[:, 0].reshape(n, h*w))
        return dataset.reshape(n, 1, h, w).expand(n, c, h, w)


if __name__ == "__main__":

    
    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=16)
    plt.rcParams["figure.figsize"] = (8, 5)

    dataset = args.dataset
    assert dataset in ["mnist", "fashion_mnist", "cats_vs_dogs", "cifar100", "food101"], "--dataset must be in ['mnist', 'fashion_mnist', 'cats_vs_dogs', 'cifar100', 'food101']"

    SEED = args.seed
    device = args.device
    max_samples = args.max_samples
    plot_name = args.plot_name
    project_early = args.project_early
    path = "plot/" + dataset + "/"

    alphas = [0.5, 0.75, 1., 1.25, 1.5]
    if args.add_optimal_alpha:
        alphas = [None] + alphas

    if not os.path.exists(path):
        os.makedirs(path)

    if dataset == "mnist":
        n_components = [1, 24, 48, 72, 96, 120, 144]
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "mnist", "farleyknight-org-username/vit-base-mnist"

    elif dataset == "cifar100":
        n_components = [1, 48, 96, 144, 192, 240, 288, 336, 384, 432, 476, 524]
        image_name, label_name, is_rgb = "img", "fine_label", True
        dataset_name, model_name = "cifar100", "Ahmed9275/Vit-Cifar100"

    elif dataset == "food101":
        n_components = [1, 64, 128, 192, 256, 320, 384, 448, 512]
        image_name, label_name, is_rgb = "image", "label", True
        dataset_name, model_name = "food101", "eslamxm/vit-base-food101"

    elif dataset == "cats_vs_dogs":
        n_components = [1, 16, 32, 48, 64, 80, 96]
        image_name, label_name, is_rgb = "image", "labels", True
        dataset_name, model_name = "Bingsu/Cat_and_Dog", "nateraw/vit-base-cats-vs-dogs"

    # Model has low accuracy
    elif dataset == "fashion_mnist":
        n_components = [1, 32, 64, 96, 128, 160, 192]
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "fashion_mnist", "abhishek/autotrain_fashion_mnist_vit_base"

    else:
        raise()

    
    dataset = load_dataset(dataset_name)

    if project_early:
        base_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_resize=True, image_mean=[0, 0, 0], image_std=[1, 1, 1])
    else:
        base_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

    dataset_train = dataset["train"].shuffle(seed=SEED)[:max_samples]
    X_train, y_train = process_dataset(dataset["train"], max_samples=max_samples)

    try:
        dataset_test = dataset["test"].shuffle(seed=SEED)[:max_samples]
        base_X_test, y_test = process_dataset(dataset["test"], max_samples=max_samples)
    except:
        dataset_test = dataset["validation"].shuffle(seed=SEED)[:max_samples]
        base_X_test, y_test = process_dataset(dataset["validation"], max_samples=max_samples)

    if args.add_noise:
        X_train += torch.normal(mean=torch.zeros_like(X_train), std=torch.ones_like(X_train)*args.noise_std)
        base_X_test += torch.normal(mean=torch.zeros_like(base_X_test), std=torch.ones_like(base_X_test)*args.noise_std)

    for alpha in alphas:
        scores = []
        for n_comp in n_components:
            if alpha is not None:
                pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                pca = fit(pca, X_train, is_rgb)
                X_test = approximate(pca, base_X_test, is_rgb)
            else:
                pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                n, c, h, w = X_train.size()
                data = X_train.reshape(n, c*h*w) if is_rgb else X_train[:, 0].reshape(n, h*w)
                best_alpha = pca.compute_optimal_alpha(data, n_components=n_comp)

                pca = AlphaPCA(n_components=n_comp, alpha=best_alpha, random_state=SEED)
                pca = fit(pca, X_train, is_rgb)
                X_test = approximate(pca, base_X_test, is_rgb)
            
            if project_early:
                X_test = resize_img(X_test)
            #X_test = base_X_test

            accuracy = 0
            for inp, label in zip(X_test.split(1, dim=0), y_test):
                with torch.no_grad():
                    outputs = model(inp.to(device))
                    if outputs.logits.argmax(dim=-1).cpu()[0] == label:
                        accuracy += 1

            print("n_comp:", n_comp, "  alpha:", alpha if alpha is not None else best_alpha, "  accuracy:", accuracy / max_samples)
            scores.append(accuracy / max_samples)

        label_value = "alpha: " + str(alpha) if alpha is not None else "optimal alpha"
        plt.plot(n_components, scores, label=label_value)
        
    plt.legend(loc="best")
    plt.xlabel("Number of components")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path + plot_name, dpi=250)