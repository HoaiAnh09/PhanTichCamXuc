import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = text.translate(str.maketrans('', '', string.punctuation))  # Xóa dấu câu
    return text

# Đọc file CSV
file_path = 'amazon.csv'  # Đường dẫn đến tệp dữ liệu của bạn
data = pd.read_csv(file_path)

# Kiểm tra xem có cột 'Text' và 'label' không
if 'Text' not in data.columns or 'label' not in data.columns:
    raise ValueError("Tệp dữ liệu cần có các cột 'Text' và 'label'.")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% - 20%)
X = data['Text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In ra kích thước của tập huấn luyện và tập kiểm tra
print("Kích thước tập huấn luyện:", X_train.shape[0])
print("Kích thước tập kiểm tra:", X_test.shape[0])

# Tiền xử lý văn bản
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Sử dụng TF-IDF để chuyển đổi văn bản thành dạng số
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Áp dụng thuật toán Naive Bayes để xây dựng mô hình phân loại
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_tfidf)

# Đánh giá mô hình bằng độ chính xác và F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# In ra kết quả
print(f'Độ chính xác của mô hình: {accuracy:.2f}')
print(f'F1-score của mô hình: {f1:.2f}')

