import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# ============================================================
# 1. 데이터 로딩
# ============================================================
def load_dataset(dataset_path):
    """
    CSV 데이터셋 불러오기 (마지막 컬럼 = label 사용)

    Args:
        dataset_path (str): dataset.csv 경로

    Returns:
        X (pd.DataFrame): feature 데이터
        y (pd.Series): label (dataset의 마지막 컬럼)
    """
    dataset = pd.read_csv(dataset_path, index_col="Unnamed: 0")

    # 마지막 컬럼을 label로 사용
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]

    return X, y


# ============================================================
# 2. Train/Val/Test 분할
# ============================================================
def split_data(X, y, test_size=0.2, val_size=0.2, random_state=25):
    """
    stratify를 사용하여 train/val/test 분할
    """
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 3. Feature Scaling
# ============================================================
def scale_features(X_train, X_val=None, X_test=None):
    """
    MinMaxScaler로 feature scaling
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================
# 4. Gene-Pathway 매핑
# ============================================================
def extract_gene_pathway(X, save_path=None):
    """
    gene-pathway 매핑 추출
    """
    gene_pathway = pd.DataFrame({
        "gene": X.columns.str.split(".").str[0],
        "pathway": X.columns.str.split(".").str[1]
    })
    if save_path:
        gene_pathway.to_csv(save_path, index=False)
    return gene_pathway


# ============================================================
# 5. Gene × Pathway Matrix 템플릿 생성
# ============================================================
def build_gene_matrix(X, *, pathway_list, cell_list, max_genes=383):
    """
    Gene-Pathway 매트릭스 생성 (패딩 포함)
    """
    gene_matrix = [[None for _ in range(len(pathway_list))] for _ in range(max_genes)]

    cell = cell_list[0]
    cell_series = X.loc[cell]

    for col_idx, pw in enumerate(pathway_list):
        cols_in_pw = cell_series[cell_series.index.str.contains(f"\.{pw}")].index.tolist()
        genes = [col.split(".")[0] for col in cols_in_pw]
        genes_padded = genes + [None] * (max_genes - len(genes))
        for row_idx, gene in enumerate(genes_padded[:max_genes]):
            gene_matrix[row_idx][col_idx] = gene

    return gene_matrix


# ============================================================
# 6. Cell-wise → Numpy 변환 
# ============================================================
def cells_to_numpy(X, y, pathway_list, max_genes=383):
    """
    각 cell을 (max_genes × n_pathways) 행렬로 변환하여 
    responder / nonresponder numpy 배열을 생성

    Args:
        X (pd.DataFrame): feature 데이터 (cell × gene.pathway)
        y (pd.Series or np.ndarray): responder label (0/1)
        pathway_list (list): pathway 이름 리스트 (고정 순서)
        max_genes (int): pathway별 최대 gene 수 (default=383)

    Returns:
        X_responder (np.ndarray): responder cell 배열 (N_r, H, W)
        y_responder (np.ndarray): responder 라벨 (N_r,)
        X_nonresponder (np.ndarray): nonresponder cell 배열 (N_nr, H, W)
        y_nonresponder (np.ndarray): nonresponder 라벨 (N_nr,)
    """

    def cell_to_matrix(cell_series, pathway_list, max_genes=383):
        df = pd.DataFrame(index=range(max_genes), columns=pathway_list)
        for pw in pathway_list:
            temp = cell_series[cell_series.index.str.contains(f"\.{pw}")].values
            df[pw] = np.pad(temp, (0, max_genes - len(temp)),
                            'constant', constant_values=0)
        return df.values  # (max_genes × n_pathways)

    # 전체 cell 변환
    cell_matrices = []
    for cell in tqdm(X.index, desc="Converting cells to numpy"):
        mat = cell_to_matrix(X.loc[cell], pathway_list, max_genes)
        cell_matrices.append(mat)

    X_array = np.stack(cell_matrices, axis=0)   # (n_cells, H, W)
    y_array = np.array(y)

    # responder / nonresponder 분리
    X_responder = X_array[y_array == 1]
    X_nonresponder = X_array[y_array == 0]
    y_responder = np.ones(len(X_responder), dtype="float32")
    y_nonresponder = np.zeros(len(X_nonresponder), dtype="float32")

    return X_responder, y_responder, X_nonresponder, y_nonresponder


# ============================================================
# 7. CNN2D 입력 변환
# ============================================================
def convert_to_cnn2d_input(X_array):
    """
    (N, H, W) → (N, H, W, 1) 변환 (채널 차원 추가)
    """
    if len(X_array.shape) != 3:
        raise ValueError(f"Expected 3D array (N, H, W), got {X_array.shape}")
    return X_array[..., np.newaxis]


# ============================================================
# 8. Numpy 저장 및 로드
# ============================================================
def save_numpy_arrays(X_responder, y_responder, X_nonresponder, y_nonresponder, save_dir="../data/npy"):
    """
    CNN2D 입력용 numpy 배열 저장
    """
    os.makedirs(save_dir, exist_ok=True)

    X = np.concatenate((X_responder, X_nonresponder), axis=0)
    y = np.concatenate((y_responder, y_nonresponder), axis=None)

    np.save(os.path.join(save_dir, "X_responder.npy"), X_responder)
    np.save(os.path.join(save_dir, "y_responder.npy"), y_responder)
    np.save(os.path.join(save_dir, "X_nonresponder.npy"), X_nonresponder)
    np.save(os.path.join(save_dir, "y_nonresponder.npy"), y_nonresponder)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)

    print(f" Saved numpy arrays to {save_dir}")
    print(f"  X_responder: {X_responder.shape}, y_responder: {y_responder.shape}")
    print(f"  X_nonresponder: {X_nonresponder.shape}, y_nonresponder: {y_nonresponder.shape}")
    print(f"  X (merged): {X.shape}, y (merged): {y.shape}")


def load_numpy_arrays(save_dir="../data/npy"):
    """
    CNN2D 입력용 numpy 배열 불러오기
    """
    X_responder = np.load(os.path.join(save_dir, "X_responder.npy"))
    y_responder = np.load(os.path.join(save_dir, "y_responder.npy"))
    X_nonresponder = np.load(os.path.join(save_dir, "X_nonresponder.npy"))
    y_nonresponder = np.load(os.path.join(save_dir, "y_nonresponder.npy"))
    return X_responder, y_responder, X_nonresponder, y_nonresponder


