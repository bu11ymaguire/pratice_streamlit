import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 0. 경로 및 기본 설정
# ==========================================
# 여기 경로 본인 환경에 맞게 설정해주셔야 합니다!!
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    os.chdir(BASE_PATH)
except FileNotFoundError:
    st.error(f"❌ 경로를 찾을 수 없습니다: {BASE_PATH}")
    st.info("streamlit_website.py 파일이 있는 위치에서 실행해주세요.")
    st.stop()

st.set_page_config(page_title="신용카드 연체 예측", page_icon="💳", layout="wide")

# ==========================================
# 1. 모델 및 데이터 로드 함수
# ==========================================
@st.cache_resource
def load_resources():
    # 1. 데이터 로드 (Scaler, Encoder 세팅용)
    try:
        df = pd.read_csv('credit_card_dataset.csv')
    except FileNotFoundError:
        st.error("❌ 'credit_card_dataset.csv' 파일을 찾을 수 없습니다.")
        return None, None, None, None, None

    # 컬럼명 대문자 변환 (에러 방지)
    df.columns = df.columns.str.upper()

    # 전처리 (ID 제거)
    if 'ID' in df.columns: df = df.drop(columns=['ID'])
    if 'UNNAMED: 0' in df.columns: df = df.drop(columns=['UNNAMED: 0'])
    
    df = df.fillna(0)
    
    # Target 제거
    target_col = 'DEFAULT_PAYMENT_NEXT_MONTH'
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
    else:
        X = df.copy()
    
    # 범주형/수치형 정의
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # 데이터 검증
    missing_cols = [col for col in categorical_features if col not in X.columns]
    if missing_cols:
        st.error(f"❌ 데이터 파일에 다음 컬럼이 없습니다: {missing_cols}")
        st.stop()

    numeric_cols = [col for col in X.columns if col not in categorical_features]

    # Scaler & Encoder 학습
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        le.fit(X[col])
        le_dict[col] = le
    
    scaler = StandardScaler()
    scaler.fit(X[numeric_cols])

    # 2. 모델 파일 로드
    models = {}
    
    # (1) XGBoost 로드
    if os.path.exists('optimal_xgboost_model.pkl'):
        try:
            models['XGBoost'] = joblib.load('optimal_xgboost_model.pkl')
        except Exception as e:
            st.warning(f"⚠️ XGBoost 모델 로드 실패: {e}")

    # (2) LightGBM 로드
    if os.path.exists('optimal_lightgbm_model.pkl'):
        try:
            models['LightGBM'] = joblib.load('optimal_lightgbm_model.pkl')
        except Exception as e:
            st.warning(f"⚠️ LightGBM 모델 로드 실패: {e}")

    # (3) Random Forest 로드
    if os.path.exists('optimal_random_forest_model.pkl'):
        try:
            models['Random Forest'] = joblib.load('optimal_random_forest_model.pkl')
        except Exception as e:
            st.warning(f"⚠️ Random Forest 모델 로드 실패: {e}")
    
    # 모델이 하나도 없을 때 경고
    if not models:
        st.error("⚠️ 폴더에 모델 파일(.pkl)이 하나도 없습니다. 주피터 노트북을 실행해 모델을 생성해주세요.")

    return scaler, le_dict, models, list(X.columns), numeric_cols

# 리소스 로드 실행
scaler, le_dict, models, feature_names, numeric_cols = load_resources()

# 모델이 없으면 중단
if not models:
    st.stop()

# ==========================================
# 2. 사용자 입력 (사이드바)
# ==========================================
st.sidebar.header("📝 정보 입력")

def user_input_features():
    
    sex = st.sidebar.selectbox("성별 (SEX)", ["male", "female"]) 
    education = st.sidebar.selectbox("교육 (EDUCATION)", ["graduate school", "university", "high school", "others"])
    marriage = st.sidebar.selectbox("결혼 (MARRIAGE)", ["married", "single", "others"])
    age = st.sidebar.number_input("나이 (AGE)", min_value=20, max_value=80, value=30)
    child_num = st.sidebar.number_input("자녀 수 (CHILDREN)", min_value=0, max_value=10, value=0)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("상환 기록 (-2:무사용, -1:정상, 1~8:연체개월)")
    pay_0 = st.sidebar.slider("9월 상환 (PAY_0)", -2, 8, 0)
    pay_2 = st.sidebar.slider("8월 상환 (PAY_2)", -2, 8, 0)
    pay_3 = st.sidebar.slider("7월 상환 (PAY_3)", -2, 8, 0)
    pay_4 = st.sidebar.slider("6월 상환 (PAY_4)", -2, 8, 0)
    pay_5 = st.sidebar.slider("5월 상환 (PAY_5)", -2, 8, 0)
    pay_6 = st.sidebar.slider("4월 상환 (PAY_6)", -2, 8, 0)
    
    bill_amt = 0 
    pay_amt = 0
    
    # 딕셔너리 키 대문자 통일
    data = {
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'CHILDREN': child_num,
        'PAY_0': pay_0,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'PAY_4': pay_4,
        'PAY_5': pay_5,
        'PAY_6': pay_6,
        'BILL_AMT1': bill_amt,
        'BILL_AMT2': bill_amt,
        'BILL_AMT3': bill_amt, 
        'BILL_AMT4': bill_amt,
        'BILL_AMT5': bill_amt,
        'BILL_AMT6': bill_amt,
        'PAY_AMT1': pay_amt,
        'PAY_AMT2': pay_amt,
        'PAY_AMT3': pay_amt, 
        'PAY_AMT4': pay_amt,
        'PAY_AMT5': pay_amt,
        'PAY_AMT6': pay_amt
    }
    return data

input_data = user_input_features()

# ==========================================
# 3. 예측 실행
# ==========================================
st.title("💳 신용카드 연체 예측")

col1, col2 = st.columns(2)
with col1:
    st.subheader("입력 확인")
    st.dataframe(pd.DataFrame([input_data]).T, height=300)

with col2:
    st.subheader("결과 예측")
    
    # 여기서 모델을 선택합니다 (Random Forest가 있다면 목록에 뜹니다)
    model_name = st.selectbox("사용할 모델 선택", list(models.keys()))
    
    if st.button("예측하기", type="primary"):
        input_df = pd.DataFrame([input_data])
        
        # 컬럼 순서 맞추기
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        try:
            # 인코딩
            for col, le in le_dict.items():
                if input_df[col][0] not in le.classes_:
                    st.error(f"입력값 오류: {col}에 '{input_df[col][0]}' 값은 학습 데이터에 없습니다.")
                    st.stop()
                input_df[col] = le.transform(input_df[col])
            
            # 스케일링
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            # 예측 수행
            model = models[model_name]
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.write("---")
            st.write(f"**선택된 모델:** {model_name}")
            
            if pred == 1:
                st.error(f"🚨 **연체 위험** (확률: {prob:.1%})")
                st.write("다음 달 연체 가능성이 높습니다.")
            else:
                st.success(f"✅ **정상 납부** (연체 확률: {prob:.1%})")
                st.write("안전한 고객으로 예측됩니다.")

        except Exception as e:
            st.error(f"예측 중 에러 발생: {e}") 