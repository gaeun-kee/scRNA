# Preprocessing
from .preprocessing import (
    load_dataset,         
    split_data,           
    scale_features,       
    extract_gene_pathway, 
    build_gene_matrix,     
    cells_to_numpy,
    load_numpy_arrays,
    convert_to_cnn2d_input,
    save_numpy_arrays
)

# Deep Learning Models
from .model import (
    build_cnn1d,
    build_cnn2d
)

# Grad-CAM Utilities
from .gradcam_utils import (
    make_grad_heatmap,
    display_gradcam_with_img,
    compute_group_heatmap,
    compute_difference_heatmap,
    resize_and_normalize_heatmap,
    plot_heatmap_with_max_region,
    plot_heatmap_diff,
    plot_diff_heatmap_with_top_genes,
    get_top_positive_diff_spots,
    get_top_negative_diff_spots,
    plot_diff_heatmap_subset
)

# Training
from .train import (
    set_seed,
    train_model
)

# Evaluation
from .evaluate import (
    roc_auc_ci,
    roc_auprc_ci,
    evaluate_model,
    plot_curves
)
