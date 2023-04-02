
export MAX_SAMPLES=750

python alpha_pca_vit.py \
    --add_noise \
    --dataset mnist \
    --device cuda \
    --max_samples $MAX_SAMPLES \
    --noise_std 0.25 \
    --plot_name mnist_with_noise.png \
    --seed 123
