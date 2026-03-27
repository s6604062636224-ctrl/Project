import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import os

# --- นำเข้า Library สำหรับทำ ML โมเดลรถยนต์ (MPG) ---
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- นำเข้า Library สำหรับทำ Deep Learning หมาแมว (CNN) ---
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# ตั้งค่าหน้าเว็บหลัก
# ==========================================
st.set_page_config(page_title="AI & Data Science Project Portfolio", page_icon="🤖", layout="wide")

# ==========================================
# ฟังก์ชันโหลดโมเดล (Cache ไว้จะได้ไม่โหลดซ้ำ)
# ==========================================
@st.cache_resource
def load_models():
    try:
        ensemble_model = joblib.load('ensemble_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        ensemble_model, scaler = None, None
        
    try:
        cnn_model = tf.keras.models.load_model('dog_cat_model.h5')
    except Exception as e:
        cnn_model = None
        
    return ensemble_model, scaler, cnn_model

ensemble_model, scaler, cnn_model = load_models()

# ==========================================
# สร้าง Sidebar เมนู (4 หน้า)
# ==========================================
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("เลือกส่วนของโครงงาน:", [
    "📖 ส่วนที่ 1: ทฤษฎี Machine Learning",
    "📖 ส่วนที่ 2: ทฤษฎี Deep Learning (CNN)",
    "🚗 ส่วนที่ 3: ระบบพยากรณ์อัตราสิ้นเปลือง (MPG)",
    "🐶🐱 ส่วนที่ 4: ระบบจำแนกรูปภาพ (Image Classification)"
])

st.sidebar.markdown("---")
st.sidebar.info("พัฒนาระบบโดยใช้ Streamlit, Scikit-learn และ TensorFlow")

# ==========================================
# หน้าที่ 1: ทฤษฎี Machine Learning
# ==========================================
if app_mode == "📖 ส่วนที่ 1: ทฤษฎี Machine Learning":
    st.title("🚗 โครงงาน Machine Learning: การพยากรณ์อัตราการสิ้นเปลืองเชื้อเพลิง (MPG)")
    
    st.header("1. หลักการและเหตุผล (Introduction)")
    st.write("""
    Machine Learning (การเรียนรู้ของเครื่อง) คือกระบวนการสร้างโมเดลทางคณิตศาสตร์เพื่อให้คอมพิวเตอร์สามารถเรียนรู้รูปแบบ (Patterns) จากข้อมูลในอดีต 
    และนำไปสู่การพยากรณ์ผลลัพธ์ของข้อมูลชุดใหม่ ในโครงงานนี้ได้ประยุกต์ใช้เทคนิค **Ensemble Learning** ซึ่งเป็นการผสานการทำงานของอัลกอริทึมหลายตัว (ได้แก่ Linear Regression, Decision Tree และ Support Vector Regression) เพื่อเพิ่มประสิทธิภาพและความแม่นยำในการพยากรณ์
    """)
    
    st.header("2. วิธีการดำเนินงาน (Methodology)")
    st.markdown("""
    1. **Data Collection (การรวบรวมข้อมูล):** นำเข้าชุดข้อมูลคุณลักษณะของรถยนต์ (Automobile Dataset)
    2. **Data Cleaning (การทำความสะอาดข้อมูล):** จัดการกับค่าสูญหาย (Missing Values) และขจัดอักขระที่ไม่ถูกต้องเพื่อเตรียมความพร้อมให้ข้อมูล
    3. **Data Preprocessing (การประมวลผลข้อมูลเบื้องต้น):** ปรับสเกลข้อมูล (Standard Scaling) เพื่อลดความเหลื่อมล้ำของค่าตัวเลขในแต่ละคุณลักษณะ
    4. **Modeling (การสร้างโมเดล):** พัฒนาโมเดลแบบ Ensemble ด้วยเทคนิค Voting Regressor
    5. **Evaluation (การประเมินประสิทธิภาพ):** วัดผลสัมฤทธิ์ของโมเดลผ่านค่า Mean Squared Error (MSE) และ R-Squared (R²)
    """)
    
    st.header("3. ปัญหาและอุปสรรคในการพัฒนา")
    st.warning("""
    **ปัญหาความไม่สมบูรณ์ของข้อมูล (Dirty Data):** ชุดข้อมูลตั้งต้นมีความบกพร่องสูง เช่น มีอักขระพิเศษปะปน และชนิดข้อมูล (Data Type) คลาดเคลื่อน ส่งผลให้ไม่สามารถนำเข้าสู่กระบวนการ Train ได้ทันที จึงต้องพัฒนากระบวนการทำความสะอาดข้อมูลเชิงลึก (Deep Cleaning) ก่อนนำเข้าสู่ขั้นตอน Preprocessing
    """)
    
    st.header("4. แหล่งอ้างอิงชุดข้อมูล (Dataset Reference)")
    st.info("ชุดข้อมูล Automobile_dirty.csv ในโครงงานนี้ ได้รับการสังเคราะห์และดัดแปลงขึ้นโดย **Generative AI** เพื่อใช้เป็นกรณีศึกษาสำหรับการทำ Data Preparation และ Data Cleaning โดยเฉพาะ")

# ==========================================
# หน้าที่ 2: ทฤษฎี CNN (Neural Network)
# ==========================================
elif app_mode == "📖 ส่วนที่ 2: ทฤษฎี Deep Learning (CNN)":
    st.title("🐶🐱 โครงงาน Deep Learning: การจำแนกประเภทภาพด้วย CNN")
    
    st.header("1. ทฤษฎีที่เกี่ยวข้อง (Theoretical Background)")
    st.write("""
    **Convolutional Neural Network (CNN)** คือสถาปัตยกรรมโครงข่ายประสาทเทียมเชิงลึก (Deep Learning) ที่ถูกออกแบบมาเพื่อประมวลผลข้อมูลประเภทรูปภาพโดยเฉพาะ 
    มีคุณสมบัติเด่นในการสกัดคุณลักษณะ (Feature Extraction) เช่น ขอบภาพ พื้นผิว และรูปทรง ผ่านชั้น Convolutional Layers ก่อนที่จะส่งข้อมูลต่อไปยังชั้น Fully Connected เพื่อตัดสินใจและจำแนกคลาสของรูปภาพ
    """)
    
    st.header("2. สถาปัตยกรรมและการพัฒนา (Architecture & Implementation)")
    st.markdown("""
    1. **Data Preparation:** นำเข้าชุดข้อมูลภาพสุนัขและแมว พร้อมแบ่งสัดส่วนสำหรับ Training และ Testing
    2. **Data Augmentation:** ประยุกต์ใช้เทคนิคการแปลงภาพ (เช่น การสุ่มหมุนภาพ และพลิกภาพ) เพื่อลดความเสี่ยงของการเกิด Overfitting และเพิ่มความหลากหลายของข้อมูลสอน
    3. **Transfer Learning:** นำสถาปัตยกรรม **MobileNetV2** ที่ผ่านการฝึกสอน (Pre-trained) มาแล้ว มาใช้เป็นฐานในการสกัดคุณลักษณะภาพหลัก
    4. **Custom Classification Head:** ออกแบบชั้น Dense Layer บริเวณส่วนปลายของเครือข่าย เพื่อทำหน้าที่จำแนกผลลัพธ์แบบ Binary Classification (สุนัข หรือ แมว)
    5. **Model Training:** ดำเนินการฝึกสอนและบันทึกน้ำหนัก (Weights) ให้อยู่ในรูปแบบไฟล์ `.h5` เพื่อนำไปใช้งานต่อไป
    """)
    
    st.header("3. การแก้ปัญหาและปรับปรุงโมเดล (Optimization)")
    st.warning("""
    **ปัญหา Overfitting และข้อจำกัดด้านปริมาณข้อมูล:** ในระยะแรกของการทดลองสร้างโมเดลจากศูนย์ (Train from scratch) พบว่าโมเดลไม่สามารถลู่เข้าหาคำตอบที่ถูกต้องได้ (Loss ไม่ลดลง) เนื่องจากข้อจำกัดของจำนวนภาพ 
    **แนวทางแก้ไข:** จึงได้ปรับเปลี่ยนสถาปัตยกรรมไปใช้เทคนิค **Transfer Learning** (MobileNetV2) ซึ่งช่วยให้โมเดลมีประสิทธิภาพการเรียนรู้สูงขึ้นอย่างมีนัยสำคัญแม้ฝึกสอนด้วยชุดข้อมูลขนาดเล็ก
    """)
    
    st.header("4. แหล่งอ้างอิงชุดข้อมูล (Dataset Reference)")
    st.info("ชุดข้อมูลภาพถ่ายสำหรับการฝึกสอน (Cats and Dogs Image Classification) อ้างอิงจากฐานข้อมูลสาธารณะ Kaggle: \n\n🔗 [Samuel Cortinhas - Kaggle Dataset](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification?select=train)")

# ==========================================
# หน้าที่ 3: แอปทำนาย MPG (+ ระบบเทรนโมเดล)
# ==========================================
elif app_mode == "🚗 ส่วนที่ 3: ระบบพยากรณ์อัตราสิ้นเปลือง (MPG)":
    st.title("🚗 ระบบพยากรณ์อัตราการสิ้นเปลืองเชื้อเพลิงรถยนต์ (MPG Prediction)")
    
    # --- ส่วนของการเทรนโมเดล ---
    st.markdown("---")
    st.subheader("⚙️ ส่วนการฝึกสอนโมเดล (Model Training Module)")
    st.write("โมดูลสำหรับการประมวลผลข้อมูล `Automobile_dirty.csv` และสร้างออบเจกต์โมเดล Machine Learning")
    
    if st.button("🚀 เริ่มการประมวลผลและฝึกสอนโมเดล (Train Model)"):
        with st.spinner("ระบบกำลังดำเนินการทำความสะอาดข้อมูลและฝึกสอนโมเดล..."):
            try:
                # โหลดและคลีนข้อมูล
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
                
                new_scaler = StandardScaler()
                X_train_scaled = new_scaler.fit_transform(X_train)
                X_test_scaled = new_scaler.transform(X_test)
                
                # สร้างและเทรนโมเดล
                model_1 = LinearRegression()
                model_2 = DecisionTreeRegressor(max_depth=5, random_state=42)
                model_3 = SVR(kernel='rbf', C=100, gamma=0.1)
                new_ensemble = VotingRegressor(estimators=[('lr', model_1), ('dt', model_2), ('svr', model_3)])
                
                new_ensemble.fit(X_train_scaled, y_train)
                
                # วัดความแม่นยำ
                predictions = new_ensemble.predict(X_test_scaled)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                # เซฟโมเดลและล้าง Cache
                joblib.dump(new_ensemble, 'ensemble_model.pkl')
                joblib.dump(new_scaler, 'scaler.pkl')
                load_models.clear() # บังคับให้โหลดโมเดลตัวใหม่
                
                st.success(f"✅ การฝึกสอนเสร็จสมบูรณ์ ระบบได้บันทึกโมเดลเข้าสู่ระบบแล้ว")
                st.info(f"📊 **ประสิทธิภาพการพยากรณ์ของโมเดล (Metrics):** R-Squared = **{r2:.2f}** | Mean Squared Error (MSE) = **{mse:.2f}**")
                
                # --- ข้อมูลเชิงลึกและกราฟ ---
                st.write("### 📊 การวิเคราะห์ผลลัพธ์เชิงลึก (Post-Training Analysis)")
                with st.expander("🔍 ตรวจสอบชุดข้อมูลหลังกระบวนการ Data Cleaning"):
                    st.dataframe(df.head(10))
                    st.caption("ตารางแสดงตัวอย่างข้อมูลที่ผ่านการขจัดอักขระขยะและเติมเต็มค่าสูญหายเรียบร้อยแล้ว")

                st.write("**กราฟแสดงความสัมพันธ์ระหว่าง ค่าจริง (Actual) และ ค่าที่พยากรณ์ได้ (Predicted)**")
                chart_data = pd.DataFrame({
                    'Actual MPG (ค่าจริง)': y_test.values,
                    'Predicted MPG (โมเดลพยากรณ์)': predictions
                })
                st.scatter_chart(chart_data)
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดทางระบบ: {e}")

    # --- ส่วนของการใช้งาน (ทำนาย) ---
    st.markdown("---")
    st.subheader("🔮 ส่วนการทดสอบและพยากรณ์ (Inference Module)")
    
    current_ensemble, current_scaler, _ = load_models()
    
    if current_ensemble is None:
         st.warning("🚨 ไม่พบออบเจกต์โมเดลในระบบ กรุณาดำเนินการ 'เริ่มการประมวลผลและฝึกสอนโมเดล' ก่อนเข้าใช้งานส่วนนี้")
    else:
        col1, col2 = st.columns(2)
        with col1:
            cylinders = st.number_input("จำนวนกระบอกสูบ (Cylinders)", min_value=3, max_value=12, value=4, help="พารามิเตอร์จำนวนลูกสูบ (ค่าปกติ: 4, 6, 8)")
            displacement = st.number_input("ปริมาตรกระบอกสูบ (Displacement - cu. inches)", min_value=50, max_value=500, value=150)
            horsepower = st.number_input("กำลังเครื่องยนต์ (Horsepower)", min_value=40, max_value=300, value=100)
        with col2:
            weight = st.number_input("น้ำหนักตัวถัง (Weight - lbs)", min_value=1500, max_value=5500, value=3000)
            acceleration = st.number_input("อัตราเร่ง 0-60 ไมล์ (Acceleration - sec)", min_value=8.0, max_value=25.0, value=15.0)
            model_year = st.number_input("ปีที่ผลิต (Model Year - ex. 70 = 1970)", min_value=70, max_value=82, value=76)
        
        if st.button("คำนวณอัตราสิ้นเปลือง (Predict)", type="primary"):
            input_data = pd.DataFrame([[cylinders, displacement, horsepower, weight, acceleration, model_year]], 
                                      columns=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year'])
            input_scaled = current_scaler.transform(input_data)
            prediction = current_ensemble.predict(input_scaled)
            st.success(f"### 🎯 ผลการพยากรณ์อัตราสิ้นเปลือง: {prediction[0]:.2f} ไมล์ต่อแกลลอน (MPG)")

# ==========================================
# หน้าที่ 4: แอปแยกหมา-แมว (+ ระบบเทรนโมเดล)
# ==========================================
elif app_mode == "🐶🐱 ส่วนที่ 4: ระบบจำแนกรูปภาพ (Image Classification)":
    st.title("🐶🐱 ระบบจำแนกภาพสุนัขและแมวด้วยสถาปัตยกรรม CNN")
    
    # --- ส่วนของการเทรนโมเดล ---
    st.markdown("---")
    st.subheader("⚙️ ส่วนการฝึกสอนโมเดลโครงข่ายประสาทเทียม (Model Training Module)")
    st.write("โมดูลสำหรับการฝึกสอนสถาปัตยกรรม CNN โดยประยุกต์ใช้เทคนิค Transfer Learning จากชุดข้อมูลภาพ")
    
    if st.button("🚀 เริ่มการฝึกสอนโมเดลภาพ (Train CNN)"):
        with st.spinner("ระบบกำลังดำเนินการสกัดฟีเจอร์และฝึกสอนโมเดล (กรุณารอสักครู่)..."):
            try:
                # แก้ Path ให้ตรงกับเครื่องคุณ
                train_dir = r'C:\Users\sakkarin\Downloads\Project\train' 
                test_dir = r'C:\Users\sakkarin\Downloads\Project\test'
                
                BATCH_SIZE = 32
                IMG_SIZE = (150, 150)
                
                train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
                test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
                
                base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
                base_model.trainable = False
                
                new_cnn_model = models.Sequential([
                    layers.RandomFlip("horizontal", input_shape=(150, 150, 3)),
                    layers.RandomRotation(0.1),
                    layers.Rescaling(1./127.5, offset=-1), 
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation='sigmoid') 
                ])
                
                new_cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
                
                # เทรนแค่ 3 รอบพอสำหรับการโชว์ใน Web App เพื่อไม่ให้รอนานเกินไป
                history = new_cnn_model.fit(train_dataset, validation_data=test_dataset, epochs=3)
                
                # เซฟโมเดล
                new_cnn_model.save('dog_cat_model.h5')
                load_models.clear() # บังคับรีโหลดโมเดล
                
                # ดึงค่า Accuracy รอบสุดท้ายมาโชว์
                final_val_acc = history.history['val_accuracy'][-1]
                
                st.success("✅ กระบวนการฝึกสอนเสร็จสมบูรณ์ ระบบบันทึกค่าน้ำหนัก (Weights) เรียบร้อยแล้ว")
                st.info(f"📊 **ประสิทธิภาพของโมเดล (Validation Accuracy): {(final_val_acc * 100):.2f}%**")
                
                # --- กราฟประวัติการเทรน ---
                st.write("### 📈 พลวัตการเรียนรู้ของโมเดล (Training & Validation History)")
                with st.expander("ตรวจสอบกราฟแสดงผล Accuracy และ Loss Function"):
                    hist_df = pd.DataFrame(history.history)
                    
                    st.write("**กราฟแสดงความแม่นยำ (Model Accuracy)**")
                    st.line_chart(hist_df[['accuracy', 'val_accuracy']])
                    
                    st.write("**กราฟแสดงค่าความสูญเสีย (Model Loss)**")
                    st.line_chart(hist_df[['loss', 'val_loss']])
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e} (กรุณาตรวจสอบความถูกต้องของ Directory Path)")

    # --- ส่วนของการใช้งาน (ทำนาย) ---
    st.markdown("---")
    st.subheader("🔍 ส่วนการทดสอบจำแนกภาพ (Image Inference Module)")
    
    _, _, current_cnn = load_models()
    
    if current_cnn is None:
         st.warning("🚨 ไม่พบไฟล์โมเดล `.h5` ในระบบ กรุณาดำเนินการ 'เริ่มการฝึกสอนโมเดลภาพ' ก่อนเข้าใช้งานส่วนนี้")
    else:
        uploaded_file = st.file_uploader("นำเข้าข้อมูลภาพเพื่อทดสอบการจำแนกประเภท (รองรับไฟล์ .JPG, .PNG, .JPEG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='ภาพตัวอย่างที่นำเข้าสู่ระบบ', width=300)
            
            if st.button("🔍 เริ่มกระบวนการวิเคราะห์ภาพ (Analyze Image)", type="primary"):
                with st.spinner("ระบบกำลังประมวลผลเทนเซอร์ (Tensor Processing)..."):
                    img = image.resize((150, 150))
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) 
                    
                    prediction = current_cnn.predict(img_array)
                    score = prediction[0][0]
                    confidence = score if score > 0.5 else 1 - score
                    
                    if score > 0.5:
                        st.success(f"### 🐶 ผลการจำแนก: สุนัข (Dog) | ความน่าจะเป็น: {(confidence * 100):.2f}%")
                    else:
                        st.success(f"### 🐱 ผลการจำแนก: แมว (Cat) | ความน่าจะเป็น: {(confidence * 100):.2f}%")
                    
                    # แถบ Progress Bar โชว์ระดับความมั่นใจ
                    st.write("ระดับความเชื่อมั่นของโมเดล (Confidence Level):")
                    st.progress(float(confidence))