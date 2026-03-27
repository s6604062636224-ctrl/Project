import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2 # นำเข้า MobileNetV2

# ==========================================
# ส่วนที่ 1: โหลดและทำความสะอาดข้อมูล Automobile_dirty.csv
# ==========================================
print("--- เริ่มส่วนที่ 1: เทรนโมเดลข้อมูลรถยนต์ ---")
df = pd.read_csv('Automobile_dirty.csv')

df = df.replace(r'\?', '', regex=True) 
df = df.replace(['nan', ''], np.nan)

numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

X = df.drop(['name', 'origin', 'mpg'], axis=1) 
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_1 = LinearRegression()
model_2 = DecisionTreeRegressor(max_depth=5, random_state=42)
model_3 = SVR(kernel='rbf', C=100, gamma=0.1)

ensemble_model = VotingRegressor(estimators=[
    ('lr', model_1),
    ('dt', model_2),
    ('svr', model_3)
])

print("กำลังเทรน Ensemble Model...")
ensemble_model.fit(X_train_scaled, y_train)

predictions = ensemble_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"ผลเทรนรถยนต์ -> MSE: {mse:.2f}, R2 Score: {r2:.2f}")
print("------------------------------------------\n")


# ==========================================
# ส่วนที่ 2: โมเดลแยกหมาแมวด้วย Transfer Learning (MobileNetV2)
# ==========================================
print("--- เริ่มส่วนที่ 2: เทรนโมเดลรูปภาพหมาแมว ---")

train_dir = r'C:\Users\sakkarin\Downloads\Project\train' 
test_dir = r'C:\Users\sakkarin\Downloads\Project\test'

BATCH_SIZE = 32
IMG_SIZE = (150, 150)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# โหลดสมองกล MobileNetV2 (ไม่เอาเลเยอร์สุดท้ายมาด้วย)
print("กำลังโหลดสมองกล MobileNetV2...")
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False # ล็อกน้ำหนักไว้ ไม่ให้มันเปลี่ยนความรู้เดิม

# สร้างโครงสร้างโมเดลใหม่
cnn_model = models.Sequential([
    # ทำ Data Augmentation
    layers.RandomFlip("horizontal", input_shape=(150, 150, 3)),
    layers.RandomRotation(0.1),
    
    # MobileNetV2 รับค่าพิกเซลระหว่าง -1 ถึง 1
    layers.Rescaling(1./127.5, offset=-1), 
    
    # เสียบสมองกล MobileNetV2
    base_model,
    
    # ย่อข้อมูลให้เล็กลง
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # กันท่องจำนิดหน่อย
    
    # เลเยอร์สุดท้าย ทายผล 2 คลาส (หมา=1, แมว=0)
    layers.Dense(1, activation='sigmoid') 
])

# ใช้ Learning Rate ปกติได้เลย (0.001)
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

print("เริ่มเทรนโมเดลแยกหมาแมว...")
# ตั้งเทรนแค่ 5 รอบก็พอแล้วครับ สำหรับ Transfer Learning
history = cnn_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10 
)
print("เทรนโมเดลรูปภาพเสร็จสมบูรณ์!")

import joblib

# บันทึกโมเดลรถยนต์และตัวแปลงสเกล (Scaler)
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# บันทึกโมเดลหมาแมว
cnn_model.save('dog_cat_model.h5')

print("\nบันทึกไฟล์โมเดลสำเร็จ! พร้อมนำไปใช้ใน app.py แล้วครับ")