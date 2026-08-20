import os
import pandas as pd
import pyreadstat

super_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_db = os.path.join(super_dir, "database")

df = pd.read_csv(os.path.join(dir_db, "TCGA_COAD_Clinical_Cleaned.csv"), sep="\t")

def load_and_merge_data(onco_file):
    print(f"--- 1. 데이터 처리 및 병합 시작 (ageG) ---")
    
    print("  [진단 로그] 1-1. OncoLnc CSV 파일 로드 시도 중...")
    df_onco = pd.read_csv(onco_file, usecols=['age_at_initial_pathologic_diagnosis'])

    # --- 추가된 부분 시작 ---
    # 나이 기준(65세 초과 'High', 이하 'Low')으로 'ageG' 열 생성
    df_onco['ageG'] = df_onco['age_at_initial_pathologic_diagnosis'].apply(lambda x: 'High' if x > 65 else 'Low')
    
    # 임시로 df_onco를 df_merged에 할당 (이후 추가적인 병합 로직이 있다면 이 부분에 작성)
    df_merged = df_onco.copy()
    # --- 추가된 부분 끝 ---
    
    print("  [진단 로그] 1-5. 백업용 CSV 파일 저장 진행 중...")
    csv_path = os.path.join(dir_db, f"age_at_initial_pathologic_diagnosis_.csv")
    df_merged.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("  [진단 로그] 1-5. 백업 CSV 저장 완료.")
    
    return df_merged