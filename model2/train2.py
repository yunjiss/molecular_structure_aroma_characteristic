from __future__ import annotations

import os
import math
import joblib
import warnings
import numpy as np
import pandas as pd

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from scipy.stats import spearmanr
    _has_scipy = True
except Exception:
    _has_scipy = False

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# 기존 descriptor + 2개 추가
DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHBA",
    "NumHBD",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount"
]

def _compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    if mol is None:
        return np.array([np.nan] * len(DESCRIPTOR_NAMES), dtype=float)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol) + rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol)
    ], dtype=float)

def _morgan_bits(mol: Chem.Mol, radius: int = 2, n_bits: int = 4096) -> np.ndarray:
    if mol is None:
        return np.zeros((n_bits,), dtype=np.uint8)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr

def _smiles_to_mol(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol

def featurize_smiles(smiles_list, radius=2, n_bits=4096) -> Tuple[np.ndarray, np.ndarray]:
    fps, descs = [], []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        fps.append(_morgan_bits(mol, radius=radius, n_bits=n_bits))
        descs.append(_compute_descriptors(mol))
    return np.asarray(fps, dtype=np.uint8), np.asarray(descs, dtype=float)

@dataclass
class FeatureConfig:
    radius: int = 2
    n_bits: int = 4096
    descriptor_names: Tuple[str, ...] = tuple(DESCRIPTOR_NAMES)
    y_transform: str = "log1p"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["descriptor_names"] = list(self.descriptor_names)
        return d

def _transform_y(y: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if mode == "log1p":
        return np.log1p(y.clip(min=0)), {"mode": mode}
    return y, {"mode": "none"}

def _inverse_transform_y(yhat: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    mode = info.get("mode", "none")
    if mode == "log1p":
        return np.expm1(yhat)
    return yhat

def _load_training_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {"Drug_ID":"Drug_ID","Drug":"Drug","SMILES":"Drug","Y":"Y","Target":"Y"}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    if "Drug" not in df.columns or "Y" not in df.columns:
        raise ValueError("CSV에 'Drug'(SMILES)와 'Y' 열이 필요합니다.")
    return df[["Drug_ID"] + ["Drug","Y"]] if "Drug_ID" in df.columns else df[["Drug","Y"]]

def train(train_data_path: str, save_path: str):
    df = _load_training_csv(train_data_path).copy()
    df = df.dropna(subset=["Drug","Y"]).reset_index(drop=True)
    cfg = FeatureConfig()

    fps, descs = featurize_smiles(df["Drug"].tolist(), radius=cfg.radius, n_bits=cfg.n_bits)

    # NaN 처리
    if np.isnan(descs).any():
        col_medians = np.nanmedian(descs, axis=0)
        inds = np.where(np.isnan(descs))
        descs[inds] = np.take(col_medians, inds[1])

    y_raw = df["Y"].astype(float).values
    y_tr, y_info = _transform_y(y_raw, cfg.y_transform)

    X_desc_train, X_desc_val, X_fp_train, X_fp_val, y_train, y_val = train_test_split(
        descs, fps, y_tr, test_size=0.15, random_state=42
    )

    scaler = StandardScaler()
    X_desc_train_scaled = scaler.fit_transform(X_desc_train)
    X_desc_val_scaled = scaler.transform(X_desc_val)

    X_train = np.hstack([X_fp_train.astype(np.float32), X_desc_train_scaled.astype(np.float32)])
    X_val = np.hstack([X_fp_val.astype(np.float32), X_desc_val_scaled.astype(np.float32)])

    model = HistGradientBoostingRegressor(
        max_depth=12,
        learning_rate=0.05,
        max_iter=1000,
        l2_regularization=0.0,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_val_pred_tr = model.predict(X_val)
    y_val_pred = _inverse_transform_y(y_val_pred_tr, y_info)
    y_val_true = _inverse_transform_y(y_val, y_info)

    rmse = math.sqrt(mean_squared_error(y_val_true, y_val_pred))
    mae = mean_absolute_error(y_val_true, y_val_pred)
    r2 = r2_score(y_val_true, y_val_pred)
    if _has_scipy:
        sp,_ = spearmanr(y_val_true, y_val_pred)
    else:
        sp = pd.Series(y_val_true).rank().corr(pd.Series(y_val_pred).rank(), method="pearson")

    print(f"[Validation] RMSE={rmse:.4f}  MAE={mae:.4f}  Spearman={sp:.4f}  R2={r2:.4f}")

    # 전체 학습
    scaler_full = StandardScaler()
    descs_scaled_full = scaler_full.fit_transform(descs.astype(np.float32))
    X_full = np.hstack([fps.astype(np.float32), descs_scaled_full.astype(np.float32)])
    model_full = HistGradientBoostingRegressor(
        max_depth=12,
        learning_rate=0.05,
        max_iter=1000,
        l2_regularization=0.0,
        min_samples_leaf=10,
        random_state=42
    ).fit(X_full, y_tr)

    ckpt = {
        "feature_config": cfg.to_dict(),
        "descriptor_scaler": scaler_full,
        "model": model_full,
        "y_info": y_info,
        "columns": {"smiles":"Drug","target":"Y"}
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(ckpt, save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="model_improved/model.pkl")
    args = parser.parse_args()
    train(args.train_csv, args.save_path)
