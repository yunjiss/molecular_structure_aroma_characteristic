import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import inchi  # InChI 변환용
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import os
import copy
from tqdm import tqdm  # 실시간 진행상황

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

def convert_inchi_to_smiles(inchi_str: str):
    try:
        mol = inchi.MolFromInchi(inchi_str)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"[InChI 변환 실패] {inchi_str} | {str(e)}")
    return None

class MoleculeDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
    def smiles_to_graph(self, smiles):
        smiles = smiles.strip()
        if smiles.startswith("InChI="):
            smiles_converted = convert_inchi_to_smiles(smiles)
            if smiles_converted is None:
                print(f"[InChI→SMILES 변환 실패] {smiles}")
                return Data(x=torch.zeros((1, 9)), edge_index=torch.zeros((2, 0), dtype=torch.long))
            smiles = smiles_converted

        mol = None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    )
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception as e:
                    print(f"[Kekulize 오류] {smiles} | {str(e)}")
                    try:
                        Chem.SanitizeMol(
                            mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        )
                    except Exception as e2:
                        print(f"[Sanitize 오류] {smiles} | {str(e2)}")
                        mol = None
            if mol is None and '.' in smiles:
                parts = smiles.split('.')
                for part in sorted(parts, key=len, reverse=True):
                    try:
                        part = part.strip()
                        mol = Chem.MolFromSmiles(part, sanitize=False)
                        if mol is not None:
                            Chem.SanitizeMol(
                                mol,
                                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                            )
                            break
                    except Exception as e:
                        print(f"[복합SMILES 파싱 오류] {part} | {str(e)}")
                        continue
        except Exception as e:
            print(f"[SMILES 파싱 최종 실패] {smiles} | {str(e)}")
            mol = None

        if mol is None:
            return Data(x=torch.zeros((1, 9)), edge_index=torch.zeros((2, 0), dtype=torch.long))

        try:
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetMass(),
                    atom.GetTotalValence(),
                    int(atom.IsInRing()),
                    atom.GetNumRadicalElectrons()
                ]
                atom_features.append(features)

            edge_indices = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])

            edge_index = (
                torch.zeros((2, 0), dtype=torch.long)
                if len(edge_indices) == 0
                else torch.tensor(edge_indices).t().contiguous()
            )
            x = torch.tensor(atom_features, dtype=torch.float)
            return Data(x=x, edge_index=edge_index)
        except Exception as e:
            print(f"[그래프 변환 실패] {smiles} | {str(e)}")
            return Data(x=torch.zeros((1, 9)), edge_index=torch.zeros((2, 0), dtype=torch.long))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        mol1_graph = self.smiles_to_graph(item['mol1'])
        mol2_graph = self.smiles_to_graph(item['mol2'])
        mol1_notes_encoded = self.label_encoder.transform([item['mol1_notes']])[0]
        mol2_notes_encoded = self.label_encoder.transform([item['mol2_notes']])[0]
        blend_notes_encoded = self.label_encoder.transform([item['blend_notes']])[0]
        return {
            'mol1_graph': mol1_graph,
            'mol2_graph': mol2_graph,
            'mol1_notes': torch.tensor(mol1_notes_encoded, dtype=torch.float),
            'mol2_notes': torch.tensor(mol2_notes_encoded, dtype=torch.float),
            'blend_notes': torch.tensor(blend_notes_encoded, dtype=torch.float)
        }

def collate_fn(batch):
    mol1_graphs = [item['mol1_graph'] for item in batch]
    mol2_graphs = [item['mol2_graph'] for item in batch]
    mol1_batch = Batch.from_data_list(mol1_graphs)
    mol2_batch = Batch.from_data_list(mol2_graphs)
    mol1_notes = torch.stack([item['mol1_notes'] for item in batch])
    mol2_notes = torch.stack([item['mol2_notes'] for item in batch])
    blend_notes = torch.stack([item['blend_notes'] for item in batch])
    return {
        'mol1_batch': mol1_batch,
        'mol2_batch': mol2_batch,
        'mol1_notes': mol1_notes,
        'mol2_notes': mol2_notes,
        'blend_notes': blend_notes
    }

class ImprovedFragranceGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout=0.3, model_variant='base'):
        super(ImprovedFragranceGNN, self).__init__()
        self.model_variant = model_variant
        if model_variant == 'deep':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim//2)
            self.conv5 = GCNConv(hidden_dim//2, hidden_dim//2)
        elif model_variant == 'wide':
            hidden_dim = int(hidden_dim * 1.5)
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        else:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim//2)
        self.dropout = nn.Dropout(dropout)
        self.notes_fc1 = nn.Linear(num_labels, hidden_dim)
        self.notes_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        final_dim = (hidden_dim//2) * 4
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_labels)
        )
    def forward(self, mol1_batch, mol2_batch, mol1_notes, mol2_notes):
        x1 = torch.relu(self.conv1(mol1_batch.x, mol1_batch.edge_index))
        x1 = self.dropout(x1)
        x1 = torch.relu(self.conv2(x1, mol1_batch.edge_index))
        x1 = self.dropout(x1)
        x1 = torch.relu(self.conv3(x1, mol1_batch.edge_index))
        x1 = self.dropout(x1)
        if hasattr(self, 'conv4'):
            x1 = torch.relu(self.conv4(x1, mol1_batch.edge_index))
            if hasattr(self, 'conv5') and self.model_variant == 'deep':
                x1 = torch.relu(self.conv5(x1, mol1_batch.edge_index))
        mol1_graph_repr = global_mean_pool(x1, mol1_batch.batch)
        x2 = torch.relu(self.conv1(mol2_batch.x, mol2_batch.edge_index))
        x2 = self.dropout(x2)
        x2 = torch.relu(self.conv2(x2, mol2_batch.edge_index))
        x2 = self.dropout(x2)
        x2 = torch.relu(self.conv3(x2, mol2_batch.edge_index))
        x2 = self.dropout(x2)
        if hasattr(self, 'conv4'):
            x2 = torch.relu(self.conv4(x2, mol2_batch.edge_index))
            if hasattr(self, 'conv5') and self.model_variant == 'deep':
                x2 = torch.relu(self.conv5(x2, mol2_batch.edge_index))
        mol2_graph_repr = global_mean_pool(x2, mol2_batch.batch)
        mol1_notes_repr = torch.relu(self.notes_fc1(mol1_notes))
        mol1_notes_repr = self.dropout(mol1_notes_repr)
        mol1_notes_repr = torch.relu(self.notes_fc2(mol1_notes_repr))
        mol2_notes_repr = torch.relu(self.notes_fc1(mol2_notes))
        mol2_notes_repr = self.dropout(mol2_notes_repr)
        mol2_notes_repr = torch.relu(self.notes_fc2(mol2_notes_repr))
        combined_features = torch.cat([
            mol1_graph_repr,
            mol2_graph_repr,
            mol1_notes_repr,
            mol2_notes_repr
        ], dim=1)
        output = self.classifier(combined_features)
        return output

class EnsembleFragranceGNN:
    def __init__(self, models):
        self.models = models
        self.device = torch.device('cpu')
    def predict(self, mol1_batch, mol2_batch, mol1_notes, mol2_notes):
        ensemble_outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(mol1_batch, mol2_batch, mol1_notes, mol2_notes)
                ensemble_outputs.append(torch.sigmoid(output))
        final_output = torch.stack(ensemble_outputs).mean(dim=0)
        return final_output

def calculate_advanced_class_weights(train_data, label_encoder):
    all_blend_notes = []
    for item in train_data:
        all_blend_notes.extend(item['blend_notes'])
    label_counts = Counter(all_blend_notes)
    print("=== 라벨별 데이터 샘플 수 ===")
    for label, count in label_counts.most_common():
        print(f"{label!r}: {count}")
    print(f"=== 총 라벨 수: {len(label_counts)} ===\n")
    total_samples = len(all_blend_notes)
    strategies = {
        'sqrt': [],
        'log': [],
        'inverse': [],
        'balanced': []
    }
    for class_name in label_encoder.classes_:
        pos_count = label_counts.get(class_name, 1)
        sqrt_weight = np.sqrt(total_samples / pos_count)
        strategies['sqrt'].append(sqrt_weight)
        log_weight = np.log(total_samples / pos_count + 1)
        strategies['log'].append(log_weight)
        inv_weight = total_samples / pos_count
        strategies['inverse'].append(inv_weight)
        balanced_weight = total_samples / (len(label_encoder.classes_) * pos_count)
        strategies['balanced'].append(balanced_weight)
    final_strategies = {}
    for strategy_name, weights in strategies.items():
        weights = np.array(weights, dtype=np.float32)
        if strategy_name == 'sqrt':
            weights = np.clip(weights, 1.0, 10.0)
        elif strategy_name == 'log':
            weights = np.clip(weights, 1.0, 5.0)
        elif strategy_name == 'inverse':
            weights = np.clip(weights, 1.0, 50.0)
        else:
            weights = np.clip(weights, 1.0, 20.0)
        final_strategies[strategy_name] = torch.tensor(weights, dtype=torch.float32)
        print(f"{strategy_name} 가중치 범위: {weights.min():.4f} ~ {weights.max():.4f}")
    return final_strategies, label_counts

def create_diverse_samplers(dataset, label_counts, label_encoder, num_samplers=3):
    samplers = []
    for sampler_id in range(num_samplers):
        sample_weights = []
        for idx in range(len(dataset)):
            if hasattr(dataset, 'indices'):
                item = dataset.dataset.data[dataset.indices[idx]]
            else:
                item = dataset.data[idx]
            blend_notes = item['blend_notes']
            if sampler_id == 0:
                weight = sum(1.0 / np.log(label_counts.get(note, 1) + 2) for note in blend_notes)
            elif sampler_id == 1:
                weight = sum(1.0 / np.sqrt(label_counts.get(note, 1)) for note in blend_notes)
            else:
                weight = sum(1.0 / (label_counts.get(note, 1) ** 0.7) for note in blend_notes)
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        samplers.append(sampler)
    return samplers

def train_ensemble_models(train_data_path: str, save_path: str, num_models=3):
    device = torch.device('cpu')
    print(f"사용 중인 디바이스: {device}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"앙상블 학습 데이터 로드 완료: {len(train_data)}개 샘플")
    all_notes = set()
    for item in train_data:
        all_notes.update(item['mol1_notes'])
        all_notes.update(item['mol2_notes'])
        all_notes.update(item['blend_notes'])
    print(f"총 {len(all_notes)}개의 고유 향기 노트 발견")
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit([list(all_notes)])
    weight_strategies, label_counts = calculate_advanced_class_weights(train_data, label_encoder)
    dataset = MoleculeDataset(train_data, label_encoder)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    samplers = create_diverse_samplers(train_dataset, label_counts, label_encoder, num_models)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model_variants = ['base', 'deep', 'wide']
    weight_strategy_names = ['sqrt', 'log', 'balanced']
    trained_models = []
    model_configs = []
    print(f"\n=== {num_models}개 앙상블 모델 학습 시작 ===")
    for model_idx in range(num_models):
        print(f"\n--- 모델 {model_idx + 1}/{num_models} 학습 시작 ---")
        variant = model_variants[model_idx % len(model_variants)]
        strategy_name = weight_strategy_names[model_idx % len(weight_strategy_names)]
        pos_weights = weight_strategies[strategy_name]
        sampler = samplers[model_idx]
        print(f"모델 변형: {variant}, 가중치 전략: {strategy_name}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            sampler=sampler,
            collate_fn=collate_fn
        )
        model = ImprovedFragranceGNN(
            input_dim=9,
            hidden_dim=256,
            num_labels=len(label_encoder.classes_),
            dropout=0.3,
            model_variant=variant
        ).to(device)
        if model_idx % 2 == 0:
            criterion = FocalLoss(alpha=1, gamma=2, pos_weight=pos_weights.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        lrs = [0.001, 0.0008, 0.0012]
        lr = lrs[model_idx % len(lrs)]
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7)
        num_epochs = 10
        best_train_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            epoch_iterator = tqdm(train_loader, desc=f"모델{model_idx+1} 에포크 {epoch+1}/{num_epochs}", leave=False)
            for batch_idx, batch in enumerate(epoch_iterator):
                mol1_batch = batch['mol1_batch'].to(device)
                mol2_batch = batch['mol2_batch'].to(device)
                mol1_notes = batch['mol1_notes'].to(device)
                mol2_notes = batch['mol2_notes'].to(device)
                blend_notes = batch['blend_notes'].to(device)
                optimizer.zero_grad()
                outputs = model(mol1_batch, mol2_batch, mol1_notes, mol2_notes)
                loss = criterion(outputs, blend_notes)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                epoch_iterator.set_postfix({'batch_loss': loss.item()})
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mol1_batch = batch['mol1_batch'].to(device)
                    mol2_batch = batch['mol2_batch'].to(device)
                    mol1_notes = batch['mol1_notes'].to(device)
                    mol2_notes = batch['mol2_notes'].to(device)
                    blend_notes = batch['blend_notes'].to(device)
                    outputs = model(mol1_batch, mol2_batch, mol1_notes, mol2_notes)
                    if model_idx % 2 == 0:
                        loss = criterion(outputs, blend_notes)
                    else:
                        loss = F.binary_cross_entropy_with_logits(outputs, blend_notes, pos_weight=pos_weights.to(device))
                    val_loss += loss.item()
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"  에포크 {epoch+1}: 학습 손실 {train_loss:.4f}, 검증 손실 {val_loss:.4f}")
            scheduler.step(val_loss)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  모델 {model_idx + 1} 조기 종료 (에포크 {epoch + 1})")
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        trained_models.append(model)
        model_configs.append({
            'variant': variant,
            'strategy': strategy_name,
            'best_train_loss': best_train_loss
        })
        print(f'모델 {model_idx + 1} 학습 완료! 최적 학습 손실: {best_train_loss:.4f}')
    ensemble_model = EnsembleFragranceGNN(trained_models)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ensemble_data = {
        'models_state_dict': [model.state_dict() for model in trained_models],
        'model_configs': model_configs,
        'label_encoder': label_encoder,
        'weight_strategies': weight_strategies,
        'label_counts': label_counts,
        'ensemble_config': {
            'input_dim': 9,
            'hidden_dim': 256,
            'num_labels': len(label_encoder.classes_),
            'num_models': num_models
        }
    }
    with open(save_path, 'wb') as f:
        pickle.dump(ensemble_data, f)
    print(f'\n=== 앙상블 모델 학습 완료! ===')
    print(f'총 {num_models}개 모델이 {save_path}에 저장되었습니다.')
    print("\n=== 개별 모델 성능 요약 ===")
    for i, config in enumerate(model_configs):
        print(f"모델 {i+1}: {config['variant']} + {config['strategy']} -> 학습 손실: {config['best_train_loss']:.4f}")

def train(train_data_path: str, save_path: str):
    train_ensemble_models(train_data_path, save_path, num_models=3)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        num_models = int(sys.argv[3])
        train_ensemble_models(sys.argv[1], sys.argv[2], num_models)
    else:
        train(sys.argv[1], sys.argv[2])
