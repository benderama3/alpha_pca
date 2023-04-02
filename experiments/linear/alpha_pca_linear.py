import torch 
import numpy as np
from alpha_pca import AlphaPCA
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 5), dpi=300)

def process_dataset(dataset, extractor, is_rgb=False):
    dataset_train = dataset["train"].shuffle(seed=SEED)[:max_samples]
    try:
        dataset_test = dataset["test"].shuffle(seed=SEED)[:max_samples]
    except:
        dataset_test = dataset["validation"].shuffle(seed=SEED)[:max_samples]

    X_train, y_train = dataset_train[image_name], np.array(dataset_train[label_name])
    base_X_test, y_test = dataset_test[image_name], np.array(dataset_test[label_name])

    if not is_rgb:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="np")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="np")["pixel_values"][:max_samples, 0].reshape(max_samples, -1)
    else:
        X_train = extractor([x.convert("RGB") for x in X_train], return_tensors="np")["pixel_values"][:max_samples].reshape(max_samples, -1)
        base_X_test = extractor([x.convert("RGB") for x in base_X_test], return_tensors="np")["pixel_values"][:max_samples].reshape(max_samples, -1)
    return X_train, y_train, base_X_test, y_test

"""
plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=22)
"""

parser = argparse.ArgumentParser()

parser.add_argument('--add_outliers', action='store_true')
parser.add_argument('--add_optimal_alpha', action='store_true')
parser.add_argument('--fit_latent_space', action='store_true')
parser.add_argument('--dataset', type=str, required=False, default="digits")
parser.add_argument('--device', type=str, required=False, default="cpu")
parser.add_argument('--max_samples', type=int, required=False, default=500)
parser.add_argument('--outlier_prob', type=float, required=False, default=0.01)
parser.add_argument('--outlier_factor', type=float, required=False, default=10)
parser.add_argument('--noise_std', type=float, required=False, default=0.25)
parser.add_argument('--plot_name', type=str, required=False, default="plot.png")
parser.add_argument('--seed', type=int, required=False, default=123)
args = parser.parse_args()



if __name__ == "__main__":

    plt.rcParams.update({'font.size': 17})
    plt.rc('axes', labelsize=20)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams.update({'figure.autolayout': True})

    dataset = args.dataset
    assert dataset in ['digits', 'diabetes', 'wine', 'fashion_mnist', 'mnist', 'cifar100', 'food101', 'cats_vs_dogs'], "--dataset must be in ['digits', 'diabetes', 'fashion_mnist']"

    SEED = args.seed
    max_samples = args.max_samples
    plot_name = args.plot_name
    path = "plot/" + dataset + "/"

    alphas = [0.5, 0.75, 1., 1.25, 1.5]
    if args.add_optimal_alpha:
        alphas = [None] + alphas

    if not os.path.exists(path):
        os.makedirs(path)

    if dataset == "digits":
        n_components = [1, 2, 4, 8, 16, 24, 32]
        is_regression = False
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        X_train, base_X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)

    elif dataset == "mnist":
        n_components = [1, 8, 16, 32, 48, 64, 80]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset
        
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "mnist", "farleyknight-org-username/vit-base-mnist"
        extractor = ViTFeatureExtractor(do_resize=False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        #extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=False)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    elif dataset == "fashion_mnist":
        n_components = [1, 2, 4, 8, 16, 24, 32]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset
        
        image_name, label_name, is_rgb = "image", "label", False
        dataset_name, model_name = "fashion_mnist", "abhishek/autotrain_fashion_mnist_vit_base"
        extractor = ViTFeatureExtractor(do_resize=False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        #extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=False)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)
    
    elif dataset == "cifar100":
        n_components = [1, 16, 32, 48, 64, 80, 96, 112, 128]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset

        image_name, label_name, is_rgb = "img", "fine_label", True
        dataset_name, model_name = "cifar100", "Ahmed9275/Vit-Cifar100"
        #extractor = ViTFeatureExtractor(do_resize=False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=False)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    elif dataset == "food101":
        n_components = [1, 16, 32, 48, 64, 80, 96, 112, 128]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset

        image_name, label_name, is_rgb = "image", "label", True
        dataset_name, model_name = "food101", "eslamxm/vit-base-food101"
        #extractor = ViTFeatureExtractor(size=128, do_resize=True, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=True)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    elif dataset == "cats_vs_dogs":
        n_components = [1, 16, 32, 48, 64, 80, 96, 112, 128]
        is_regression = False
        
        from transformers import ViTFeatureExtractor
        from datasets import load_dataset

        image_name, label_name, is_rgb = "image", "labels", True
        dataset_name, model_name = "Bingsu/Cat_and_Dog", "nateraw/vit-base-cats-vs-dogs"
        #extractor = ViTFeatureExtractor(size=128, do_resize=True, image_mean=[0, 0, 0], image_std=[1, 1, 1])
        extractor = ViTFeatureExtractor.from_pretrained(model_name, do_resize=True)
        dataset = load_dataset(dataset_name)

        X_train, y_train, base_X_test, y_test = process_dataset(dataset, extractor, is_rgb)

    elif dataset == "wine":
        n_components = [1, 2, 3, 4, 5, 6, 7, 8]
        is_regression = False
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        X_train, base_X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)

    elif dataset == "diabetes":
        n_components = [1, 2, 3, 4, 5, 6, 7, 8]
        is_regression = True
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        X_train, base_X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)
    else:
        raise()

    if args.add_outliers:
        n, d = base_X_test.shape
        outliers = np.random.binomial(1, args.outlier_prob, n*d).reshape(n, d)
        base_X_test += (args.outlier_factor - 1) * base_X_test * outliers


    if not is_regression:
        base_model = LogisticRegression(max_iter=500, random_state=SEED).fit(X_train, y_train)
    else:
        base_model = LinearRegression().fit(X_train, y_train)

    X_train, base_X_test = torch.from_numpy(X_train).to(args.device), torch.from_numpy(base_X_test).to(args.device)
    
    for alpha in alphas:
        scores = []

        for n_comp in n_components:
            
            if args.fit_latent_space:
                if alpha is not None:
                    pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                    pca.fit(X_train)
                else:
                    pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                    best_alpha = pca.compute_optimal_alpha(X_train, n_components=n_comp)
                    pca = AlphaPCA(n_components=n_comp, alpha=best_alpha, random_state=SEED)
                    pca.fit(X_train)

                if not is_regression:
                    base_model = LogisticRegression(max_iter=500, random_state=SEED).fit(pca.transform(X_train).cpu().numpy(), y_train)
                else:
                    base_model = LinearRegression().fit(pca.transform(X_train).cpu().numpy(), y_train)

                X_test = pca.transform(base_X_test).cpu().numpy()
                accuracy = base_model.score(X_test, y_test)
            else:
                if alpha is not None:
                    pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                    pca.fit(X_train)
                    X_test = pca.approximate(base_X_test).cpu().numpy()
                    accuracy = base_model.score(X_test, y_test)
                else:
                    pca = AlphaPCA(n_components=n_comp, alpha=alpha, random_state=SEED)
                    best_alpha = pca.compute_optimal_alpha(X_train, n_components=n_comp)

                    pca = AlphaPCA(n_components=n_comp, alpha=best_alpha, random_state=SEED)
                    pca.fit(X_train)
                    X_test = pca.approximate(base_X_test).cpu().numpy()
                    accuracy = base_model.score(X_test, y_test)

            print("n_comp:", n_comp, "  alpha:", alpha if alpha is not None else best_alpha, "  accuracy:", accuracy)
            scores.append(accuracy)

        label_value = r"$\alpha$ = " + str(alpha) if alpha is not None else r"Approx. $\alpha^*$"
        plt.plot(n_components, scores, label=label_value)

    plt.legend(loc="best", frameon=False)
    plt.xlabel("Number of components")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.ylim(0.2, 0.90)
    plt.savefig(path + plot_name, dpi=250)