import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
from pathlib import Path

# =================================================================
# 1. CONFIGURATION (Cấu hình hệ thống)
# =================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG = {
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "max_seq_length": 512,
    "embed_dim": 100,
    
    # Cấu hình cho BiLSTM
    "bilstm": {
        "path_weights": BASE_DIR / 'Weights' / 'Deep_Learning' / 'BiLSTM_weights' / 'bilstm_weights.pth',
        "path_word2idx": BASE_DIR / 'Weights' / 'Deep_Learning' / 'BiLSTM_weights' / 'word2idx.pkl',
        "path_label_encoder": BASE_DIR / 'Weights' / 'Deep_Learning' / 'BiLSTM_weights' / 'label_encoder.pkl',
        "hidden_dim": 128,
        "dropout": 0.5
    },
    
    # Cấu hình cho TextCNN
    "textcnn": {
        "path_weights": BASE_DIR / 'Weights' / 'Deep_Learning' / 'TextCNN_weights' / 'textcnn_weights.pth',
        "path_word2idx": BASE_DIR / 'Weights' / 'Deep_Learning' / 'TextCNN_weights' / 'word2idx.pkl',
        "path_label_encoder": BASE_DIR / 'Weights' / 'Deep_Learning' / 'TextCNN_weights' / 'label_encoder.pkl',
        "filter_sizes": [3, 4, 5],
        "num_filters": 100,
        "dropout": 0.5
    }
}

# =================================================================
# 2. ĐỊNH NGHĨA CẤU TRÚC MODEL
# =================================================================

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled_out, _ = torch.max(lstm_out, dim=1)
        return self.fc(self.dropout(pooled_out))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            pooled_outputs.append(p)
        x_cat = torch.cat(pooled_outputs, dim=1)
        return self.fc(self.dropout(x_cat))

# =================================================================
# 3. KHỞI TẠO VÀ LOAD DỮ LIỆU (Load 2 bộ Word2Idx riêng)
# =================================================================

# --- Load cho BiLSTM ---
with open(CONFIG["bilstm"]["path_word2idx"], 'rb') as f:
    word2idx_lstm = pickle.load(f)
with open(CONFIG["bilstm"]["path_label_encoder"], 'rb') as f:
    le_lstm = pickle.load(f)

model_bilstm = BiLSTMClassifier(
    len(word2idx_lstm), CONFIG["embed_dim"], CONFIG["bilstm"]["hidden_dim"], 
    len(le_lstm.classes_), CONFIG["bilstm"]["dropout"]
).to(CONFIG["device"])
model_bilstm.load_state_dict(torch.load(CONFIG["bilstm"]["path_weights"], map_location=CONFIG["device"]))
model_bilstm.eval()

# --- Load cho TextCNN ---
with open(CONFIG["textcnn"]["path_word2idx"], 'rb') as f:
    word2idx_cnn = pickle.load(f)
with open(CONFIG["textcnn"]["path_label_encoder"], 'rb') as f:
    le_cnn = pickle.load(f)

model_textcnn = TextCNN(
    len(word2idx_cnn), CONFIG["embed_dim"], len(le_cnn.classes_), 
    CONFIG["textcnn"]["filter_sizes"], CONFIG["textcnn"]["num_filters"], CONFIG["textcnn"]["dropout"]
).to(CONFIG["device"])
model_textcnn.load_state_dict(torch.load(CONFIG["textcnn"]["path_weights"], map_location=CONFIG["device"]))
model_textcnn.eval()

print("Hệ thống đã sẵn sàng: Đã load 2 bộ Word2Idx và 2 Model riêng biệt.")

# =================================================================
# 4. HÀM XỬ LÝ VÀ DỰ ĐOÁN
# =================================================================

def text_to_tensor(title, article, word2idx, max_len):
    """Hàm bổ trợ chuyển text thành tensor dựa trên word2idx cụ thể"""
    full_text = str(title) + " " + str(article)
    tokens = re.findall(r'\b\w+\b', full_text.lower())
    indices = [word2idx.get(w, word2idx.get('<UNK>', 1)) for w in tokens]
    
    # Padding / Truncating
    if len(indices) < max_len:
        indices = indices + [word2idx.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor([indices], dtype=torch.long).to(CONFIG["device"])

def predict_bilstm(title, article):
    """Dự đoán bằng model BiLSTM"""
    input_tensor = text_to_tensor(title, article, word2idx_lstm, CONFIG["max_seq_length"])
    with torch.no_grad():
        output = model_bilstm(input_tensor)
        prob = F.softmax(output, dim=1)
        conf, idx = torch.max(prob, dim=1)
    
    label = le_lstm.inverse_transform([idx.item()])[0]
    return label, conf.item()

def predict_textcnn(title, article):
    """Dự đoán bằng model TextCNN"""
    input_tensor = text_to_tensor(title, article, word2idx_cnn, CONFIG["max_seq_length"])
    with torch.no_grad():
        output = model_textcnn(input_tensor)
        prob = F.softmax(output, dim=1)
        conf, idx = torch.max(prob, dim=1)
    
    label = le_cnn.inverse_transform([idx.item()])[0]
    return label, conf.item()
