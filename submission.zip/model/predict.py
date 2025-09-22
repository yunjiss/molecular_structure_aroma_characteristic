import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import pickle
import numpy as np


class ImprovedFragranceGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout=0.3):
        super(ImprovedFragranceGNN, self).__init__()
        
        # GNN 레이어들
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim//2)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 분자 노트 처리를 위한 레이어
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


class PredictionDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
        
    def smiles_to_graph(self, smiles):
        """SMILES 문자열을 그래프 데이터로 변환"""
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
    
        try:
            # 원자 특성 추출
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
        
            # 결합 정보 추출
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
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
        
        return {
            'mol1_graph': mol1_graph,
            'mol2_graph': mol2_graph,
            'mol1_notes': torch.tensor(mol1_notes_encoded, dtype=torch.float),
            'mol2_notes': torch.tensor(mol2_notes_encoded, dtype=torch.float)
        }


def collate_fn_predict(batch):
    """예측용 배치 데이터를 위한 collate 함수"""
    mol1_graphs = [item['mol1_graph'] for item in batch]
    mol2_graphs = [item['mol2_graph'] for item in batch]
    
    mol1_batch = Batch.from_data_list(mol1_graphs)
    mol2_batch = Batch.from_data_list(mol2_graphs)
    
    mol1_notes = torch.stack([item['mol1_notes'] for item in batch])
    mol2_notes = torch.stack([item['mol2_notes'] for item in batch])
    
    return {
        'mol1_batch': mol1_batch,
        'mol2_batch': mol2_batch,
        'mol1_notes': mol1_notes,
        'mol2_notes': mol2_notes
    }


def predict_with_adaptive_threshold(test_data_path: str, model_path: str) -> list:
    """적응적 임계값과 Top-K를 사용한 다중 라벨 예측"""
    device = torch.device('cpu')
    
    # 모델 및 레이블 인코더 로드
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    label_encoder = model_data['label_encoder']
    model_config = model_data['model_config']
    label_counts = model_data.get('label_counts', {})
    
    print(f"모델 로드 완료. 레이블 수: {model_config['num_labels']}")
    
    # 테스트 데이터 로드
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"테스트 데이터 로드 완료: {len(test_data)}개 샘플")
    
    # 모델 초기화
    model = ImprovedFragranceGNN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_labels=model_config['num_labels']
    ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    # 데이터셋 및 데이터로더 생성
    test_dataset = PredictionDataset(test_data, label_encoder)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn_predict
    )
    
    # 예측 수행
    all_predictions = []
    
    print("적응적 다중 라벨 예측을 시작합니다...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            mol1_batch = batch['mol1_batch'].to(device)
            mol2_batch = batch['mol2_batch'].to(device)
            mol1_notes = batch['mol1_notes'].to(device)
            mol2_notes = batch['mol2_notes'].to(device)
            
            # 모델 예측 (logits 출력)
            outputs = model(mol1_batch, mol2_batch, mol1_notes, mol2_notes)
            # Sigmoid 적용하여 확률로 변환
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            for prob in probabilities:
                predicted_labels = []
                
                # 1. Top-K 방식 (상위 5개 후보 선택)
                top_k_indices = np.argsort(prob)[-5:][::-1]  # 상위 5개
                top_k_probs = prob[top_k_indices]
                
                # 2. 적응적 임계값 계산
                max_prob = top_k_probs[0]
                
                # 가장 높은 확률의 40% 이상이면서, 최소 0.2 이상인 라벨들 선택
                adaptive_threshold = max(0.5, max_prob * 0.9)
                
                # 3. 라벨 선택
                for i, idx in enumerate(top_k_indices):
                    if prob[idx] >= adaptive_threshold and len(predicted_labels) < 2:
                        predicted_labels.append(label_encoder.classes_[idx])
                    elif len(predicted_labels) >= 1 and prob[idx] < adaptive_threshold:
                        break
                
                # 4. 최소 1개 보장
                if len(predicted_labels) == 0:
                    predicted_labels = [label_encoder.classes_[top_k_indices[0]]]
                
                # 5. 빈도 기반 후처리 (너무 흔한 라벨만 있으면 다양성 추가)
                if len(predicted_labels) == 1 and len(label_counts) > 0:
                    main_label = predicted_labels[0]
                    main_count = label_counts.get(main_label, 0)
                    
                    # 매우 흔한 라벨(상위 5개)이면서 두 번째 확률이 충분히 높으면 추가
                    if main_count > 10000 and len(top_k_indices) > 1:
                        second_prob = prob[top_k_indices[1]]
                        if second_prob > 0.05:  # 15% 이상이면 추가
                            predicted_labels.append(label_encoder.classes_[top_k_indices[1]])
                
                all_predictions.append(predicted_labels)
    
    print(f"예측 완료. 총 {len(all_predictions)}개의 예측 결과 생성")
    
    # 예측 통계
    prediction_counts = [len(pred) for pred in all_predictions]
    unique_labels = set()
    for pred in all_predictions:
        unique_labels.update(pred)
    
    print(f"평균 예측 라벨 수: {np.mean(prediction_counts):.2f}")
    print(f"예측된 고유 라벨 수: {len(unique_labels)}")
    print(f"1개 라벨 예측: {sum(1 for x in prediction_counts if x == 1)}개")
    print(f"2개 라벨 예측: {sum(1 for x in prediction_counts if x == 2)}개")
    print(f"3개 라벨 예측: {sum(1 for x in prediction_counts if x == 3)}개")
    
    # 예측된 라벨 분포
    all_predicted_labels = []
    for pred in all_predictions:
        all_predicted_labels.extend(pred)
    
    from collections import Counter
    pred_distribution = Counter(all_predicted_labels)
    print(f"예측 라벨 분포 (상위 10개): {pred_distribution.most_common(10)}")
    
    return all_predictions


# 기존 predict 함수를 대체
predict = predict_with_adaptive_threshold


if __name__ == "__main__":
    import sys
    import json

    # 커맨드라인 인자로 테스트 데이터 경로와 모델 경로를 받음
    test_data_path = sys.argv[1]
    model_path = sys.argv[2]

    predictions = predict(test_data_path, model_path)

    # 예측 결과를 JSON으로 파일 저장
    output_path = "balanced_predictions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    
    print(f"예측 완료. 결과가 '{output_path}' 에 저장되었습니다.")