"""
Titanic Survival Prediction — Streamlit App
พัฒนาโดย: นางสาวระพีพันธ์ สุวรรณประเสริฐ

หมายเหตุสำคัญ (โปรดอ่าน):
โมเดล titanic_mlp.keras ที่ให้มา รับ input 5 ค่า (Pclass, Sex, Age, Fare, Embarked)
แต่ไฟล์โมเดลไม่มีไฟล์ scaler/encoder แนบมาด้วย ผู้พัฒนาจึงตั้งสมมติฐานการ encode ดังนี้
    - Sex:      male = 0, female = 1
    - Embarked: C = 0, Q = 1, S = 2
ถ้าตอนเทรนโมเดลใช้การเข้ารหัสหรือการ scale ข้อมูลต่างจากนี้ (เช่น StandardScaler,
one-hot encoding) กรุณาแก้ไขฟังก์ชัน preprocess_input() ด้านล่างให้ตรงกับตอนเทรนจริง
มิฉะนั้นผลการทำนายอาจไม่แม่นยำ

ตัวแปร MODEL_ACCURACY ด้านล่างเป็นค่าตั้งต้น (placeholder) กรุณาแก้ไขให้ตรงกับค่า
accuracy จริงที่วัดได้จาก test set ตอนเทรนโมเดล
"""

import numpy as np
import streamlit as st
import tensorflow as tf

# ---------- ตั้งค่าคงที่ ----------
MODEL_PATH = "titanic_mlp.keras"
MODEL_ACCURACY = 0.82  # TODO: แก้ไขให้ตรงกับค่า accuracy จริงจากการเทรนโมเดล
DEVELOPER_NAME = "นางสาวระพีพันธ์ สุวรรณประเสริฐ"

SEX_MAP = {"ชาย (Male)": 0, "หญิง (Female)": 1}
EMBARKED_MAP = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}

# ---------- ตั้งค่าหน้าเว็บ ----------
st.set_page_config(
    page_title="ระบบทำนายการรอดชีวิตผู้โดยสารไททานิค",
    page_icon="🚢",
    layout="centered",
)

# ---------- ธีมมินิมอล (CSS) ----------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Helvetica Neue', 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #FAFAFA;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }
    .app-title {
        text-align: center;
        font-size: 1.9rem;
        font-weight: 700;
        color: #1F2933;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #6B7280;
        margin-bottom: 1.6rem;
    }
    .accuracy-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        text-align: center;
        margin-bottom: 1.6rem;
    }
    .accuracy-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .accuracy-label {
        font-size: 0.85rem;
        color: #6B7280;
    }
    .result-card {
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-survive {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
    }
    .result-not {
        background-color: #FEF2F2;
        border: 1px solid #EF4444;
        color: #991B1B;
    }
    .developer-footer {
        text-align: center;
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    div.stButton > button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 0;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- โหลดโมเดล (cache ไว้ไม่ให้โหลดซ้ำ) ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_input(pclass: int, sex_label: str, age: float, fare: float, embarked_label: str) -> np.ndarray:
    """แปลงข้อมูลจากฟอร์มให้อยู่ในรูปแบบเดียวกับตอนเทรนโมเดล (5 features)."""
    sex_val = SEX_MAP[sex_label]
    embarked_val = EMBARKED_MAP[embarked_label]
    features = np.array([[pclass, sex_val, age, fare, embarked_val]], dtype="float32")
    return features


# ---------- ส่วนหัวของหน้าเว็บ ----------
st.markdown('<div class="app-title">🚢 ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิค</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Titanic Survival Prediction System (Neural Network / MLP)</div>', unsafe_allow_html=True)

# ---------- แสดงค่าความแม่นยำของระบบ ----------
st.markdown(
    f"""
    <div class="accuracy-card">
        <div class="accuracy-value">{MODEL_ACCURACY * 100:.2f}%</div>
        <div class="accuracy-label">ความแม่นยำของโมเดล (Model Accuracy บนชุดข้อมูลทดสอบ)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- โหลดโมเดล ----------
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}\nกรุณาตรวจสอบว่าไฟล์ {MODEL_PATH} อยู่ในโฟลเดอร์เดียวกับ app.py")

# ---------- ฟอร์มกรอกข้อมูลผู้โดยสาร ----------
st.subheader("กรอกข้อมูลผู้โดยสาร")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("ชั้นโดยสาร (Pclass)", options=[1, 2, 3], index=2,
                               help="1 = ชั้น 1 (ราคาแพงสุด), 3 = ชั้น 3 (ประหยัด)")
        sex_label = st.selectbox("เพศ (Sex)", options=list(SEX_MAP.keys()))
        age = st.slider("อายุ (Age)", min_value=0, max_value=100, value=30)

    with col2:
        fare = st.number_input("ค่าโดยสาร (Fare, £)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        embarked_label = st.selectbox("ท่าเรือที่ขึ้นเรือ (Embarked)", options=list(EMBARKED_MAP.keys()))

    submitted = st.form_submit_button("🔮 ทำนายผล")

# ---------- ทำนายผลและแสดงผลลัพธ์ ----------
if submitted and model_loaded:
    X = preprocess_input(pclass, sex_label, age, fare, embarked_label)
    prob = float(model.predict(X, verbose=0)[0][0])
    survived = prob >= 0.5

    if survived:
        st.markdown(
            f"""
            <div class="result-card result-survive">
                <h3>✅ คาดว่า “รอดชีวิต”</h3>
                <p>ความน่าจะเป็นในการรอดชีวิต: <b>{prob * 100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-not">
                <h3>❌ คาดว่า “ไม่รอดชีวิต”</h3>
                <p>ความน่าจะเป็นในการรอดชีวิต: <b>{prob * 100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("ผลการทำนายนี้อ้างอิงจากโมเดล Machine Learning เพื่อการศึกษาเท่านั้น ไม่ใช่ข้อเท็จจริงทางประวัติศาสตร์")

# ---------- Footer: ชื่อผู้พัฒนา ----------
st.markdown(
    f'<div class="developer-footer">พัฒนาโดย: {DEVELOPER_NAME}</div>',
    unsafe_allow_html=True,
)
