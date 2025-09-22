import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import os
import copy


class FocalLoss(nn.Module):
    """Focal Loss - 불균형 데이터에 효과적"""
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
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class MoleculeDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
        
    def smiles_to_graph(self, smiles):
        """개선된 SMILES → 그래프 변환"""
        mol = None

        try:
            # 1차: 기본 파싱 시도
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    )
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    try:
                        Chem.SanitizeMol(
                            mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        )
                    except:
                        mol = None

            # 2차: 복합 SMILES 처리
            if mol is None and '.' in smiles:
                parts = smiles.split('.')
                for part in sorted(parts, key=len, reverse=True):
                    try:
                        mol = Chem.MolFromSmiles(part, sanitize=False)
                        if mol is not None:
                            Chem.SanitizeMol(
                                mol,
                                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                            )
                            break
                    except:
                        continue

        except Exception:
            mol = None

        # 분자 파싱 완전 실패 시 빈 그래프 반환
        if mol is None:
            return Data(x=torch.zeros((1, 9)), edge_index=torch.zeros((2, 0), dtype=torch.long))

        # 원자/결합 특성 추출
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

            if len(edge_indices) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_indices).t().contiguous()

            x = torch.tensor(atom_features, dtype=torch.float)
            return Data(x=x, edge_index=edge_index)
            
        except Exception:
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
    """배치 데이터를 위한 collate 함수"""
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
        
        # 모델 변형에 따른 구조 다양화
        if model_variant == 'deep':
            # 더 깊은 모델
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim//2)
            self.conv5 = GCNConv(hidden_dim//2, hidden_dim//2)
        elif model_variant == 'wide':
            # 더 넓은 모델
            hidden_dim = int(hidden_dim * 1.5)
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        else:  # 'base'
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim//2)
        
        self.dropout = nn.Dropout(dropout)
        
        # 분자 노트 처리 레이어
        self.notes_fc1 = nn.Linear(num_labels, hidden_dim)
        self.notes_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        # 최종 분류 레이어
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
        # 분자 1 그래프 처리
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
        
        # 분자 2 그래프 처리
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
        
        # 분자 노트 처리
        mol1_notes_repr = torch.relu(self.notes_fc1(mol1_notes))
        mol1_notes_repr = self.dropout(mol1_notes_repr)
        mol1_notes_repr = torch.relu(self.notes_fc2(mol1_notes_repr))
        
        mol2_notes_repr = torch.relu(self.notes_fc1(mol2_notes))
        mol2_notes_repr = self.dropout(mol2_notes_repr)
        mol2_notes_repr = torch.relu(self.notes_fc2(mol2_notes_repr))
        
        # 모든 특성 결합
        combined_features = torch.cat([
            mol1_graph_repr, 
            mol2_graph_repr, 
            mol1_notes_repr, 
            mol2_notes_repr
        ], dim=1)
        
        output = self.classifier(combined_features)
        return output


class EnsembleFragranceGNN:
    """앙상블 모델 클래스"""
    def __init__(self, models):
        self.models = models
        self.device = torch.device('cpu')
    
    def predict(self, mol1_batch, mol2_batch, mol1_notes, mol2_notes):
        """앙상블 예측"""
        ensemble_outputs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(mol1_batch, mol2_batch, mol1_notes, mol2_notes)
                ensemble_outputs.append(torch.sigmoid(output))
        
        # 평균 앙상블
        final_output = torch.stack(ensemble_outputs).mean(dim=0)
        return final_output


def calculate_advanced_class_weights(train_data, label_encoder):
    """고급 클래스 가중치 계산 (다중 전략)"""
    all_blend_notes = []
    for item in train_data:
        all_blend_notes.extend(item['blend_notes'])
    
    label_counts = Counter(all_blend_notes)
    
    print("=== 라벨별 데이터 샘플 수 ===")
    for label, count in label_counts.most_common():
        print(f"{label!r}: {count}")
    print(f"=== 총 라벨 수: {len(label_counts)} ===\n")
    
    total_samples = len(all_blend_notes)
    
    # 다중 가중치 전략
    strategies = {
        'sqrt': [],      # Square root scaling
        'log': [],       # Log scaling  
        'inverse': [],   # Inverse frequency
        'balanced': []   # Balanced class weight
    }
    
    for class_name in label_encoder.classes_:
        pos_count = label_counts.get(class_name, 1)
        
        # 1. Square root scaling
        sqrt_weight = np.sqrt(total_samples / pos_count)
        strategies['sqrt'].append(sqrt_weight)
        
        # 2. Log scaling
        log_weight = np.log(total_samples / pos_count + 1)
        strategies['log'].append(log_weight)
        
        # 3. Inverse frequency
        inv_weight = total_samples / pos_count
        strategies['inverse'].append(inv_weight)
        
        # 4. Balanced (sklearn style)
        balanced_weight = total_samples / (len(label_encoder.classes_) * pos_count)
        strategies['balanced'].append(balanced_weight)
    
    # 각 전략별 정규화 및 클리핑
    final_strategies = {}
    for strategy_name, weights in strategies.items():
        weights = np.array(weights, dtype=np.float32)
        
        if strategy_name == 'sqrt':
            weights = np.clip(weights, 1.0, 10.0)
        elif strategy_name == 'log':
            weights = np.clip(weights, 1.0, 5.0)
        elif strategy_name == 'inverse':
            weights = np.clip(weights, 1.0, 50.0)
        else:  # balanced
            weights = np.clip(weights, 1.0, 20.0)
        
        final_strategies[strategy_name] = torch.tensor(weights, dtype=torch.float32)
        print(f"{strategy_name} 가중치 범위: {weights.min():.4f} ~ {weights.max():.4f}")
    
    return final_strategies, label_counts


def create_diverse_samplers(dataset, label_counts, label_encoder, num_samplers=3):
    """다양한 샘플링 전략 생성"""
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
                # 기본 역 빈도 기반
                weight = sum(1.0 / np.log(label_counts.get(note, 1) + 2) for note in blend_notes)
            elif sampler_id == 1:
                # 더 강한 희귀 라벨 선호
                weight = sum(1.0 / np.sqrt(label_counts.get(note, 1)) for note in blend_notes)
            else:
                # 중간 강도
                weight = sum(1.0 / (label_counts.get(note, 1) ** 0.7) for note in blend_notes)
            
            sample_weights.append(weight)
        
        sampler = WeightedRandomSampler(
            sample_weights, 
            num_samples=len(sample_weights),
            replacement=True
        )
        samplers.append(sampler)
    
    return samplers


def train_ensemble_models(train_data_path: str, save_path: str, num_models=3):
    """앙상블 모델 학습 함수"""
    device = torch.device('cpu')
    print(f"사용 중인 디바이스: {device}")
    
    # 데이터 로드
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"앙상블 학습 데이터 로드 완료: {len(train_data)}개 샘플")
    
    # 모든 향기 노트 수집
    all_notes = set()
    for item in train_data:
        all_notes.update(item['mol1_notes'])
        all_notes.update(item['mol2_notes'])
        all_notes.update(item['blend_notes'])
    
    print(f"총 {len(all_notes)}개의 고유 향기 노트 발견")
    
    # 레이블 인코더 생성
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit([list(all_notes)])
    
    # 고급 클래스 가중치 계산
    weight_strategies, label_counts = calculate_advanced_class_weights(train_data, label_encoder)
    
    # 데이터셋 생성
    dataset = MoleculeDataset(train_data, label_encoder)
    
    # 학습/검증 데이터 분할
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 다양한 샘플러 생성
    samplers = create_diverse_samplers(train_dataset, label_counts, label_encoder, num_models)
    
    # 검증 데이터로더
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 모델 변형 정의
    model_variants = ['base', 'deep', 'wide']
    weight_strategy_names = ['sqrt', 'log', 'balanced']
    
    trained_models = []
    model_configs = []
    
    print(f"\n=== {num_models}개 앙상블 모델 학습 시작 ===")
    
    for model_idx in range(num_models):
        print(f"\n--- 모델 {model_idx + 1}/{num_models} 학습 시작 ---")
        
        # 각 모델별 다른 설정
        variant = model_variants[model_idx % len(model_variants)]
        strategy_name = weight_strategy_names[model_idx % len(weight_strategy_names)]
        pos_weights = weight_strategies[strategy_name]
        sampler = samplers[model_idx]
        
        print(f"모델 변형: {variant}, 가중치 전략: {strategy_name}")
        
        # 학습 데이터로더
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            sampler=sampler,
            collate_fn=collate_fn
        )
        
        # 모델 생성
        model = ImprovedFragranceGNN(
            input_dim=9,
            hidden_dim=256,
            num_labels=len(label_encoder.classes_),
            dropout=0.3,
            model_variant=variant
        ).to(device)
        
        # 각 모델별 다른 손실 함수
        if model_idx % 2 == 0:
            criterion = FocalLoss(alpha=1, gamma=2, pos_weight=pos_weights.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        
        # 옵티마이저 (각 모델별 다른 learning rate)
        lrs = [0.001, 0.0008, 0.0012]
        lr = lrs[model_idx % len(lrs)]
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7)
        
        # 학습 루프
        num_epochs = 10
        best_train_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
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
            
            # 검증 단계
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
            
            if (epoch + 1) % 5 == 0:
                print(f'  에포크 {epoch+1}: 학습 손실 {train_loss:.4f}, 검증 손실 {val_loss:.4f}')
            
            scheduler.step(val_loss)
            
            # 최적 모델 저장
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'  모델 {model_idx + 1} 조기 종료 (에포크 {epoch + 1})')
                break
        
        # 최적 모델 상태 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        trained_models.append(model)
        model_configs.append({
            'variant': variant,
            'strategy': strategy_name,
            'best_train_loss': best_train_loss
        })

        print(f'모델 {model_idx + 1} 학습 완료! 최적 학습 손실: {best_train_loss:.4f}')

    # 앙상블 모델 생성
    ensemble_model = EnsembleFragranceGNN(trained_models)
    
    # 전체 모델 저장
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
    
    # 각 모델 성능 요약
    print("\n=== 개별 모델 성능 요약 ===")
    for i, config in enumerate(model_configs):
        print(f"모델 {i+1}: {config['variant']} + {config['strategy']} -> 검증 손실: {config['best_val_loss']:.4f}")


def train(train_data_path: str, save_path: str):
    """기본 train 함수 (앙상블 버전으로 리다이렉트)"""
    train_ensemble_models(train_data_path, save_path, num_models=3)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        # 앙상블 모델 수 지정 가능
        num_models = int(sys.argv[3])
        train_ensemble_models(sys.argv[1], sys.argv[2], num_models)
    else:
        train(sys.argv[1], sys.argv[2])