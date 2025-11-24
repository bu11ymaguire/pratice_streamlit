import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time

# 경로 설정
BASE_PATH = "/home/jwkim628/hello"
os.chdir(BASE_PATH)

print("=" * 80)
print("신용카드 연체 예측 모델 벤치마크")
print("=" * 80)

# 1. 데이터 로드 및 전처리
print("\n[1] 데이터 로드 중...")
df = pd.read_csv('credit_card_dataset.csv')
df.columns = df.columns.str.upper()

# ID 컬럼 제거
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])
if 'UNNAMED: 0' in df.columns:
    df = df.drop(columns=['UNNAMED: 0'])

df = df.fillna(0)

# Target 분리
target_col = 'DEFAULT_PAYMENT_NEXT_MONTH'
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"데이터 크기: {X.shape}")
print(f"클래스 분포: {y.value_counts().to_dict()}")

# 2. 범주형/수치형 컬럼 정의
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numeric_cols = [col for col in X.columns if col not in categorical_features]

# 3. 인코딩
print("\n[2] 범주형 변수 인코딩 중...")
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# 4. Train/Test 분할
print("\n[3] 데이터 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. 스케일링
print("\n[4] 스케일링 중...")
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 6. 모델 파일 목록
model_files = [
    'optimal_xgboost_model.pkl',
    'optimal_lightgbm_model.pkl',
    'optimal_random_forest_model.pkl'
]

# 7. 벤치마크 실행
print("\n" + "=" * 80)
print("모델 성능 벤치마크")
print("=" * 80)

results = []

for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"\n⚠️  {model_file} 파일을 찾을 수 없습니다. 건너뜁니다.")
        continue
    
    print(f"\n📊 {model_file} 평가 중...")
    
    # 모델 로드
    try:
        model = joblib.load(model_file)
        model_name = model_file.replace('optimal_', '').replace('_model.pkl', '').upper()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        continue
    
    # 예측 시간 측정
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    prediction_time = time.time() - start_time
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 결과 저장
    result = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp,
        'Prediction Time (s)': prediction_time
    }
    results.append(result)
    
    # 결과 출력
    print(f"  ✓ Accuracy:  {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall:    {recall:.4f}")
    print(f"  ✓ F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")
    print(f"  ✓ Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  ✓ Prediction Time: {prediction_time:.4f}s")

# 8. 결과 요약
print("\n" + "=" * 80)
print("벤치마크 결과 요약")
print("=" * 80)

if results:
    df_results = pd.DataFrame(results)
    
    # 소수점 포맷팅
    format_dict = {
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'ROC-AUC': '{:.4f}',
        'Prediction Time (s)': '{:.4f}'
    }
    
    print("\n", df_results.to_string(index=False))
    
    # 최고 성능 모델 찾기
    print("\n" + "-" * 80)
    print("🏆 최고 성능 모델:")
    print(f"  - Accuracy:  {df_results.loc[df_results['Accuracy'].idxmax(), 'Model']}")
    print(f"  - Precision: {df_results.loc[df_results['Precision'].idxmax(), 'Model']}")
    print(f"  - Recall:    {df_results.loc[df_results['Recall'].idxmax(), 'Model']}")
    print(f"  - F1-Score:  {df_results.loc[df_results['F1-Score'].idxmax(), 'Model']}")
    if df_results['ROC-AUC'].notna().any():
        print(f"  - ROC-AUC:   {df_results.loc[df_results['ROC-AUC'].idxmax(), 'Model']}")
    
    # 결과를 CSV로 저장
    df_results.to_csv('benchmark_results.csv', index=False)
    print("\n💾 결과가 'benchmark_results.csv' 파일로 저장되었습니다.")
else:
    print("\n⚠️  벤치마크할 모델이 없습니다.")

print("\n" + "=" * 80)