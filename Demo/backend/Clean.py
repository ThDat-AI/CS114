import re
import unicodedata
import emoji



def clean_text(text):
    if not isinstance(text, str) or text == "":
        return ""

    # --- 1. THAY THẾ CỤM TỪ ĐẶC BIỆT ---
    # BẰNG CHỨNG: Trigram 'double quotation mark' chiếm 10.6%.
    # Đây là lỗi khi crawl các đoạn trích dẫn nổi bật (pull quotes) của The Guardian.
    text = text.replace("double quotation mark", '"')

    # --- 2. XỬ LÝ THEO ĐOẠN (Dựa trên \n\n) ---
    # Việc tách theo \n\n giúp lọc bỏ chính xác các block rác như Caption hoặc Newsletter
    # mà không lo lắng về việc xóa nhầm nội dung trong câu.
    blocks = text.split('\n\n')
    cleaned_blocks = []

    # Các mẫu nhiễu được xác định từ thống kê Trigrams và phân tích thủ công
    noise_patterns = [
        r"View image in fullscreen", # Top 1 Trigram (89.4%)
        r"Photograph:.*?(Getty|Alamy|Reuters|AP|Guardian|PA|Images)", # Top từ phổ biến
        r"(?i)Sign up (to|for) (the )?.*?newsletter", # Newsletter CTA
        r"(?i)Sign up to (the )?.*?email",
        r"(?i)The Breakdown: sign up",
        r"(?i)Follow .*? on Twitter",
        r"Was this helpful\? Thank you for your feedback", # UI Web
        r"If you.re having trouble using the form", # Form liên hệ
        r"(?i)Reuters and Associated Press contributed", # Tín dụng thông tấn
        r"Include an address and phone number" # Yêu cầu tòa soạn
    ]

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Lọc nhiễu theo Pattern
        is_noise = False
        for pattern in noise_patterns:
            if re.search(pattern, block):
                is_noise = True
                break

        if is_noise:
            continue

        # Lọc theo độ dài: Các block quá ngắn (< 30 ký tự) không có dấu kết thúc
        # thường là tên phóng viên hoặc địa danh (London, Sydney...).
        if len(block) < 30 and not block.endswith(('.', '!', '?', '"', "'",':')):
            continue

        cleaned_blocks.append(block)

    # Gộp lại bằng dấu cách hoặc \n tùy vào mục đích (ở đây dùng dấu cách để đồng nhất text)
    text = " ".join(cleaned_blocks)

    # --- 3. XÓA URL, EMAIL, SOCIAL ---
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\S+', '', text) # Xóa các handle như @BusinessDesk
    text = re.sub(r'#(\w+)', r'\1', text)  # #ClimateChange → ClimateChange


    # --- 4. CHUẨN HÓA UNICODE VÀ ASCII ---
    # Chuyển đổi Smart Quotes về ASCII để tránh việc 'don't' bị hiểu thành 'don' và 't'
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-').replace('…', '...')

    # Xóa Emoji
    text = emoji.replace_emoji(text, replace='')

    # Chuẩn hóa Unicode + Bỏ accents (NFKD)
    text = unicodedata.normalize('NFKD', text)

    allowed_symbols = r'[€£¥₿]'
    text = re.sub(rf'[^\x00-\x7F{allowed_symbols}]', ' ', text)

    # Chuẩn hóa khoảng trắng và lowercase (không bắt buộc vì uncased sẽ lowercase)
    text = re.sub(r'\s+', ' ', text).strip()

    return text