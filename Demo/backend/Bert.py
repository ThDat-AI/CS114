import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from pathlib import Path

# 1. Đường dẫn đến thư mục chứa model đã lưu
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Weights" / "Bert" / "final_model"

# 2. Load Tokenizer và Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_PATH))
model = BertForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(device)
model.eval() # Chuyển sang chế độ đánh giá (không tính gradient)

def predict_news_category(title: str, article: str):
    """
    Hàm nhận vào title và article, trả về nhãn dự đoán và độ tự tin (confidence)
    Format: {"category": label_name, "confidence": confidence}
    """
    # 3. Tiền xử lý dữ liệu đầu vào (giống hệt lúc train)
    # Kết hợp title và article theo chuẩn của BERT (có thể dùng [SEP] giữa 2 phần)
    inputs = tokenizer(
        title,
        article,
        truncation='only_second',
        padding="max_length",
        max_length=512,
        return_tensors="pt" # Trả về định dạng PyTorch tensor
    )

    # Đưa dữ liệu lên cùng device với model (GPU/CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Thực hiện dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Lấy xác suất bằng hàm Softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Lấy class ID có xác suất cao nhất
        conf, pred_id = torch.max(probs, dim=-1)
        
        # Chuyển ID thành tên nhãn (label name)
        label_name = model.config.id2label[pred_id.item()]
        confidence = conf.item()

    return {"category": label_name, "confidence": confidence}

