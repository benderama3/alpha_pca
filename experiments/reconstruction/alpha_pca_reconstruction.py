import torch 
import numpy as np
from alpha_pca import AlphaPCA
import matplotlib.pyplot as plt
import os
import argparse
from transformers import AutoFeatureExtractor
from datasets import load_dataset

def process_dataset(dataset, extractor, is_rgb=False):
    dataset_train = dataset["train"].shuffle(seed=SEED)[:max_samples]
    try:
        dataset_test = dataset["test"].shuffle(seed=SEED)[:max_samples]
    except:
        dataset_test = dataset["validation"].shuffle(seed=SEED)[:max_samples]

    X_train, y_train = dataset_train[image_name], np.array(dataset_train[label_name])
    base_X_test, y_test = dataset_test[image_name], np.array(dataset_test[label_name])

    if not is_rgb:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="pt")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="pt")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
    else:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="pt")["pixel_values"][:max_samples].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="pt")["pixel_values"][:max_samples].reshape(max_samples, -1)
    return X_train, y_train, base_X_test, y_test


parser = argparse.ArgumentParser()

parser.add_argument('--add_outliers', action='store_true')
parser.add_argument('--add_optimal_alpha', action='store_true')
parser.add_argument('--fit_latent_space', action='store_true')
parser.add_argument('--dataset', type=str, required=False, default="digits")
parser.add_argument('--device', type=str, required=False, default="cpu")
parser.add_argument('--max_samples', type=int, required=False, default=500)
parser.add_argument('--outlier_prob', type=float, required=False, default=0.01)
parser.add_argument('--outlier_factor', type=float, required=False, default=10)
parser.add_argument('--plot_name', type=str, required=False, default="plot.png")
parser.add_argument('--project_early', action='store_true')
parser.add_argument('--remove_legend', action='store_true')
parser.add_argument('--seed', type=int, required=False, default=123)
args = parser.parse_args()



if __name__ == "__main__":
    
    plt.rcParams.update({'font.size': 17})
    plt.rc('axes', labelsize=20)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams.update({'figure.autolayout': True})

    dataset = args.dataset
    assert dataset in ['fashion_mnist', 'mnist', 'cifar100', 'food101', 'cats_vs_dogs'], "--dataset must be in ['digits', 'diabetes', 'fashion_mnist']"

    SEED = args.seed
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
        do_resize = False
        n_components = [1, 2, 4, 8, 16, 24, 32, 64]
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "mnist", "farleyknight-org-username/vit-base-mnist"

    elif dataset == "cifar100":
        do_resize = False
        n_components = [1, 48, 96, 144, 192, 240, 288, 336, 384, 432, 476]
        image_name, label_name, is_rgb = "img", "fine_label", True
        dataset_name, model_name = "cifar100", "Ahmed9275/Vit-Cifar100"

    elif dataset == "food101":
        do_resize = True
        n_components = [1, 64, 128, 192, 256, 320, 384, 448, 512]
        image_name, label_name, is_rgb = "image", "label", True
        dataset_name, model_name = "food101", "eslamxm/vit-base-food101"

    elif dataset == "cats_vs_dogs":
        do_resize = True
        n_components = [1, 16, 32, 48, 64, 80, 96]
        image_name, label_name, is_rgb = "image", "labels", True
        dataset_name, model_name = "Bingsu/Cat_and_Dog", "nateraw/vit-base-cats-vs-dogs"

    # Model has low accuracy
    elif dataset == "fashion_mnist":
        do_resize = False
        n_components = [1, 2, 4, 8, 16, 24, 32, 64]
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "fashion_mnist", "abhishek/autotrain_fashion_mnist_vit_base"

    else:
        raise()

    
    dataset = load_dataset(dataset_name)
    if project_early:
        extractor = AutoFeatureExtractor.from_pretrained(model_name, do_resize=do_resize, image_mean=[0, 0, 0], image_std=[1, 1, 1])
    else:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
    #X_train, y_train = process_dataset(dataset["train"], max_samples=max_samples)
    X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    X_train, base_X_test = X_train.to(args.device), base_X_test.to(args.device)
    
    print(type(X_train), type(base_X_test))
    for alpha in alphas:
        scores = []
        for n_comp in n_components:
            if alpha is not None:
                pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                pca.fit(X_train)
                X_test_ = pca.approximate(base_X_test)
                loss = (X_test_ - base_X_test).abs().mean().cpu()
                scores.append(loss)
                print("n_comp:", n_comp, "  alpha:", alpha, "  MAE:", loss)
            else:
                pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                best_alpha = pca.compute_optimal_alpha(X_train, n_components=n_comp)
                pca = AlphaPCA(n_components=n_comp, alpha=best_alpha, random_state=SEED)
                pca.fit(X_train)
                X_test_ = pca.approximate(base_X_test)
                loss = (X_test_ - base_X_test).abs().mean().cpu()
                scores.append(loss)
                print("n_comp:", n_comp, "  best_alpha:", best_alpha, "  MAE:", loss)

        label_value = r"$\alpha$ = " + str(alpha) if alpha is not None else r"Approx. $\alpha^*$"
        if args.remove_legend: label_value = " "
        plt.plot(n_components, scores, label=label_value)

        

    plt.legend(loc="best", frameon=False)
    plt.xlabel("Number of components")
    plt.ylabel("MAE")
    #plt.tight_layout()
    #plt.ylim(0.05, 0.39)
    plt.savefig(path + plot_name, dpi=250)