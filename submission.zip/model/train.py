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
        """개선된 SMILES → 그래프 변환 (Kekulization 오류 해결)"""
        mol = None

        try:
            # 1차: 기본 파싱 시도
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    # 부분 sanitization 시도
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    )
                    # Kekulization을 별도로 시도
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    try:
                        # Kekulization 없이 진행
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
        
        # 분자 그래프 생성
        mol1_graph = self.smiles_to_graph(item['mol1'])
        mol2_graph = self.smiles_to_graph(item['mol2'])
        
        # 개별 분자 노트를 원-핫 인코딩
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
    def __init__(self, input_dim, hidden_dim, num_labels, dropout=0.3):
        super(ImprovedFragranceGNN, self).__init__()
        
        # GNN 레이어들 (더 깊게 구성)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim//2)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 분자 노트 처리를 위한 레이어 (더 복잡하게)
        self.notes_fc1 = nn.Linear(num_labels, hidden_dim)
        self.notes_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        # 최종 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim//2) * 4, hidden_dim),
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
        x1 = torch.relu(self.conv4(x1, mol1_batch.edge_index))
        mol1_graph_repr = global_mean_pool(x1, mol1_batch.batch)
        
        # 분자 2 그래프 처리
        x2 = torch.relu(self.conv1(mol2_batch.x, mol2_batch.edge_index))
        x2 = self.dropout(x2)
        x2 = torch.relu(self.conv2(x2, mol2_batch.edge_index))
        x2 = self.dropout(x2)
        x2 = torch.relu(self.conv3(x2, mol2_batch.edge_index))
        x2 = self.dropout(x2)
        x2 = torch.relu(self.conv4(x2, mol2_batch.edge_index))
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
        
        # 최종 예측
        output = self.classifier(combined_features)
        return output


def calculate_balanced_class_weights(train_data, label_encoder):
    """균형 잡힌 클래스 가중치 계산"""
    all_blend_notes = []
    for item in train_data:
        all_blend_notes.extend(item['blend_notes'])
    
    label_counts = Counter(all_blend_notes)
    print("라벨 분포 (상위 20개):")
    print(label_counts.most_common(20))
    
    total_samples = len(all_blend_notes)
    
    # Square root scaling으로 극단적 불균형 완화
    pos_weights = []
    for class_name in label_encoder.classes_:
        pos_count = label_counts.get(class_name, 1)
        # Square root로 불균형 완화
        weight = np.sqrt(total_samples / pos_count)
        pos_weights.append(weight)
    
    pos_weights = np.array(pos_weights, dtype=np.float32)
    
    # 가중치 범위 제한 (1~5 사이)
    pos_weights = np.clip(pos_weights, 1.0, 5.0)
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"균형 조정된 가중치 범위: {pos_weights.min():.4f} ~ {pos_weights.max():.4f}")
    
    return pos_weights, label_counts


def create_balanced_sampler(dataset, label_counts, label_encoder):
    """균형 잡힌 샘플링을 위한 가중치 생성"""
    sample_weights = []
    
    for idx in range(len(dataset)):
        if hasattr(dataset, 'indices'):  # Subset의 경우
            item = dataset.dataset.data[dataset.indices[idx]]
        else:
            item = dataset.data[idx]
            
        blend_notes = item['blend_notes']
        
        # 희귀한 라벨이 포함된 샘플일수록 높은 가중치
        weight = 0
        for note in blend_notes:
            note_count = label_counts.get(note, 1)
            weight += 1.0 / np.log(note_count + 2)  # +2로 0 방지
        
        sample_weights.append(weight)
    
    return sample_weights


def train(train_data_path: str, save_path: str):
    """개선된 모델 학습 함수"""
    device = torch.device('cpu')
    print(f"사용 중인 디바이스: {device}")
    
    # 데이터 로드
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"학습 데이터 로드 완료: {len(train_data)}개 샘플")
    
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
    
    # 균형 잡힌 클래스 가중치 계산
    pos_weights, label_counts = calculate_balanced_class_weights(train_data, label_encoder)
    
    # 데이터셋 생성
    dataset = MoleculeDataset(train_data, label_encoder)
    
    # 학습/검증 데이터 분할
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 균형 잡힌 샘플링을 위한 가중치 생성
    sample_weights = create_balanced_sampler(train_dataset, label_counts, label_encoder)
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 데이터 로더 생성 (균형 샘플링 적용)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        sampler=sampler,  # WeightedRandomSampler 사용
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 개선된 모델 생성
    model = ImprovedFragranceGNN(
        input_dim=9,
        hidden_dim=256,
        num_labels=len(label_encoder.classes_)
    ).to(device)
    
    # Focal Loss 사용
    criterion = FocalLoss(alpha=1, gamma=2, pos_weight=pos_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
    
    # 학습 루프
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print("균형 조정된 모델 학습을 시작합니다...")
    
    for epoch in range(num_epochs):
        print(f'에포크 {epoch + 1} 시작', flush=True)
        
        # 학습 단계
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

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = train_loss / (batch_idx + 1)
                print(f'  배치 {batch_idx + 1}/{len(train_loader)}, 현재 평균 손실: {avg_loss:.4f}', flush=True)
        
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
                loss = criterion(outputs, blend_notes)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'에포크 {epoch+1}/{num_epochs}, 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}', flush=True)
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        # 조기 종료 및 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'pos_weights': pos_weights,
                'label_counts': label_counts,
                'model_config': {
                    'input_dim': 9,
                    'hidden_dim': 256,
                    'num_labels': len(label_encoder.classes_)
                }
            }
            patience_counter = 0
            print(f'새로운 최적 모델 저장! 검증 손실: {val_loss:.4f}', flush=True)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'조기 종료: {patience} 에포크 동안 개선되지 않음')
            break
    
    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(best_model_state, f)
    
    print(f'균형 조정된 모델 학습이 완료되어 {save_path}에 저장되었습니다.')
    print(f'최종 검증 손실: {best_val_loss:.4f}')


if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])