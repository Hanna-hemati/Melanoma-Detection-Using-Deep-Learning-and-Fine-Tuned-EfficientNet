import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# ===== تنظیم مسیرها =====
CSV_PATH = 'metadata.csv'    # فایل CSV (تغییر بده اگر اسمش فرق داره)
IMG_DIR = 'data'             # پوشه‌ای که عکس‌ها توش هستن
OUTPUT_DIR = 'dataset'       # مسیر خروجی برای train/val/test

# ===== خواندن فایل CSV =====
df = pd.read_csv(CSV_PATH)

# ===== فیلتر کردن داده‌ها: فقط benign و malignant =====
df = df[df['benign_malignant'].isin(['benign', 'malignant'])].copy()
df['label'] = df['benign_malignant'].map({'benign': 0, 'malignant': 1})

# ===== تقسیم‌بندی داده‌ها =====
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# ===== تابع برای کپی کردن تصاویر به فولدر مناسب =====
def copy_images(split_df, split_name):
    for _, row in split_df.iterrows():
        img_id = row['isic_id']
        label = 'malignant' if row['label'] == 1 else 'benign'

        src = os.path.join(IMG_DIR, f"{img_id}.jpg")
        dest_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{img_id}.jpg")

        if os.path.exists(src):
            shutil.copy(src, dest)
        else:
            print(f"⚠️ File not found: {src}")

# ===== اجرای کپی =====
print("📂 در حال کپی فایل‌های train ...")
copy_images(train_df, 'train')

print("📂 در حال کپی فایل‌های val ...")
copy_images(val_df, 'val')

print("📂 در حال کپی فایل‌های test ...")
copy_images(test_df, 'test')

print("✅ آماده‌سازی دیتاست با موفقیت انجام شد.")
