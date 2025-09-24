import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image
from skimage import measure, morphology


# -----------------------------
# 1) 개별 Grad-CAM Heatmap
# -----------------------------
def make_grad_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def display_gradcam_with_img(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    if len(img.shape) == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[-1] == 1:
        img_rgb = np.concatenate([img] * 3, axis=-1)
    else:
        img_rgb = img

    superimposed_img = jet_heatmap * alpha + img_rgb
    superimposed_img = np.uint8(superimposed_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    axes[0].imshow(img_rgb[..., 0], cmap='gray')
    axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(superimposed_img)
    axes[2].set_title("Overlay")
    plt.show()

# -----------------------------
# 2) 그룹 평균 & 차이 Heatmap
# -----------------------------
def compute_group_heatmap(X_group, model, last_conv_layer_name, target_shape=(383,186)):
    heatmap_list = []
    for i in tqdm(range(len(X_group))):
        img = X_group[i].reshape(383,186,1)   # 원본 input 크기
        heatmap = make_grad_heatmap(img[np.newaxis], model, last_conv_layer_name)

        im = Image.fromarray(heatmap)
        im = im.resize(target_shape[::-1], Image.NEAREST)
        heatmap_resized = np.array(im)

        heatmap_list.append(heatmap_resized)

    return np.nanmean(np.stack(heatmap_list), axis=0)


def compute_difference_heatmap(heatmap_a, heatmap_b):
    diff = heatmap_a - heatmap_b
    diff_norm = (diff - np.min(diff)) / (np.max(diff) - np.min(diff) + 1e-8)
    return diff_norm


def resize_and_normalize_heatmap(heatmap, target_shape=(383, 186)):
    im = Image.fromarray(heatmap)
    im = im.resize(target_shape[::-1], Image.NEAREST)
    arr = np.array(im)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
    return arr


# -----------------------------
# 3) Gene-Pathway 매핑 & Top Spot
# -----------------------------
def get_gene_pathway_from_position(row, col, gene_matrix, pathway_list):
    try:
        if isinstance(gene_matrix, pd.DataFrame):
            gene = gene_matrix.iloc[row, col]
        else:
            gene = gene_matrix[row][col]
        pathway = pathway_list[col]
    except (IndexError, KeyError):
        gene, pathway = None, None
    return gene, pathway


def get_top_n_bright_spots(heatmap, gene_matrix, pathway_list, top_n=5):
    flat_idx = np.argsort(heatmap.ravel())[::-1]
    top_coords = [np.unravel_index(idx, heatmap.shape) for idx in flat_idx]

    results = []
    for row, col in top_coords:
        gene, pathway = get_gene_pathway_from_position(row, col, gene_matrix, pathway_list)
        if gene is not None and str(gene).strip() != "":
            results.append((row, col, gene, pathway, heatmap[row, col]))
        if len(results) >= top_n:
            break
    return results


def find_gene_position(gene_name, gene_matrix, pathway_list):
    for col, pathway in enumerate(pathway_list):
        for row in range(len(gene_matrix)):
            if gene_matrix[row][col] == gene_name:
                return row, col, pathway
    return None, None, None


# -----------------------------
# 4) Heatmap 시각화 (Top spots 강조)
# -----------------------------
def plot_heatmap_with_max_region(heatmap, gene_matrix, pathway_list, top_n=5):
    # Top-N 밝은 spot 뽑기
    top_results = get_top_n_bright_spots(heatmap, gene_matrix, pathway_list, top_n=top_n)
    sorted_results = sorted(top_results, key=lambda x: (-x[4], x[1], x[0]))

    # 최대값 좌표
    max_val = np.max(heatmap)
    max_coords = np.argwhere(heatmap == max_val)[0]  # (row, col)
    r0, c0 = max_coords

    # 연결된 영역 추출 (최대값 포함된 component만)
    mask = (heatmap == max_val)
    labeled = morphology.label(mask, connectivity=2)
    region_label = labeled[r0, c0]
    region_mask = (labeled == region_label)

    contours = measure.find_contours(region_mask, 0.5)

    # ---------------- 시각화 ----------------
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.4)

    # Heatmap
    ax0 = plt.subplot(gs[0])
    im = ax0.imshow(heatmap, vmin=0.0, vmax=1.0, cmap='viridis')
    ax0.set_xlabel("Pathway")
    ax0.set_ylabel("Gene")
    ax0.set_title(f"Top {top_n} Bright Spots", fontsize=12)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Top-1 spot 화살표
    row, col, gene, pathway, value = sorted_results[0]
    ax0.annotate(
        "", xy=(col, row), xytext=(col + 10, row - 10),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", color='white', lw=2)
    )

    # 최대값 주변 영역 테두리
    for contour in contours:
        ax0.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)

    # 오른쪽 정보
    ax1 = plt.subplot(gs[1])
    ax1.axis('off')
    text_lines = [
        f"{i+1}. Gene: {gene}\n   Pathway: {pathway}\n   Row: {row}, Col: {col}, Value: {value:.3f}"
        for i, (row, col, gene, pathway, value) in enumerate(sorted_results)
    ]
    ax1.text(0, 1, "\n\n".join(text_lines), fontsize=9, va='top', ha='left', wrap=True)

    plt.show()
    plt.close()

    return sorted_results, (r0, c0, max_val)


# -----------------------------
# 5) 그룹 차이 Heatmap & Top Spots
# -----------------------------
def plot_heatmap_diff(heatmap_res, heatmap_nonres):
    heatmap_diff = heatmap_res - heatmap_nonres
    vmax = np.max(np.abs(heatmap_diff))

    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_diff, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Responder - Non-responder")
    plt.title("Grad-CAM Difference Heatmap", fontsize=12)
    plt.xlabel("Pathway")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.show()
    plt.close()
    return heatmap_diff


def get_top_diff_spots(heatmap_diff, gene_matrix, pathway_list, top_n=10):
    flat_idx = np.argsort(np.abs(heatmap_diff).ravel())[::-1]
    top_coords = [np.unravel_index(idx, heatmap_diff.shape) for idx in flat_idx]

    results = []
    for row, col in top_coords:
        gene = gene_matrix[row][col]
        pathway = pathway_list[col]
        value = heatmap_diff[row, col]
        if gene and str(gene).strip() and str(pathway).strip():
            results.append((row, col, gene, pathway, value))
        if len(results) >= top_n:
            break
    return results


def plot_diff_heatmap_with_top_genes(heatmap_diff, gene_matrix, pathway_list, top_n=10):
    results = get_top_diff_spots(heatmap_diff, gene_matrix, pathway_list, top_n=top_n)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

    ax0 = plt.subplot(gs[0])
    vmax = np.max(np.abs(heatmap_diff))
    im = ax0.imshow(heatmap_diff, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title("Grad-CAM Difference Heatmap", fontsize=12)
    ax0.set_xlabel("Pathway")
    ax0.set_ylabel("Gene")

    row, col, _, _, _ = results[0]
    ax0.annotate("", xy=(col, row), xytext=(col + 10, row - 10),
                 textcoords='data', arrowprops=dict(arrowstyle="->", color='red'))

    ax1 = plt.subplot(gs[1])
    ax1.axis('off')
    text_lines = [
        f"{i+1}. Gene: {gene}\n   Pathway: {pathway}\n   Row: {row}, Col: {col}\n   Diff: {value:.3f}"
        for i, (row, col, gene, pathway, value) in enumerate(results)
    ]
    ax1.text(0, 1, "\n\n".join(text_lines), va='top', ha='left', fontsize=8, wrap=True)

    plt.show()
    plt.close()
    return results


# -----------------------------
# 6) Positive / Negative Difference Spots
# -----------------------------
def get_top_positive_diff_spots(heatmap_diff, gene_matrix, pathway_list, top_n=10):
    flat_idx = np.argsort(heatmap_diff.ravel())[::-1]
    top_coords = [np.unravel_index(i, heatmap_diff.shape) for i in flat_idx]

    results = []
    for row, col in top_coords:
        value = heatmap_diff[row, col]
        gene = gene_matrix[row][col]
        pathway = pathway_list[col]
        if value > 0 and gene and str(gene).strip() and str(pathway).strip():
            results.append((row, col, gene, pathway, value))
        if len(results) >= top_n:
            break
    return results


def get_top_negative_diff_spots(heatmap_diff, gene_matrix, pathway_list, top_n=10):
    flat_idx = np.argsort(heatmap_diff.ravel())
    top_coords = [np.unravel_index(i, heatmap_diff.shape) for i in flat_idx]

    results = []
    for row, col in top_coords:
        value = heatmap_diff[row, col]
        gene = gene_matrix[row][col]
        pathway = pathway_list[col]
        if value < 0 and gene and str(gene).strip() and str(pathway).strip():
            results.append((row, col, gene, pathway, value))
        if len(results) >= top_n:
            break
    return results


def plot_diff_heatmap_subset(heatmap_diff, gene_matrix, pathway_list, top_results, title):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

    ax0 = plt.subplot(gs[0])
    vmax = np.max(np.abs(heatmap_diff))
    im = ax0.imshow(heatmap_diff, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title(title, fontsize=12)
    ax0.set_xlabel("Pathway")
    ax0.set_ylabel("Gene")

    if top_results:
        row, col, _, _, _ = top_results[0]
        ax0.annotate("", xy=(col, row), xytext=(col + 10, row - 10),
                     textcoords='data', arrowprops=dict(arrowstyle="->", color='red'))

    ax1 = plt.subplot(gs[1])
    ax1.axis('off')
    text_lines = [
        f"Gene: {gene}\nPathway: {pathway}\nRow: {row}, Col: {col}, Diff: {value:.3f}\n"
        for row, col, gene, pathway, value in top_results
    ]
    ax1.text(0, 1, "\n".join(text_lines), va='top', ha='left', fontsize=8, wrap=True)

    plt.show()
    plt.close()
