import pandas as pd
import numpy as np
import pyreadstat
import os
import sys
import warnings

# =====================================================================
# 출력 결과 저장용 로거(Logger) 클래스
# =====================================================================
class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

warnings.filterwarnings('ignore')

def inspect_missing_values(df, mode, output_dir):
    print(f"\n{'-'*60}")
    print(f"[{mode}] 임상 변수 결측치(Missing Values) 정밀 추적 시작")
    print(f"{'-'*60}")
    
    initial_n = len(df)
    print(f"▶ 초기 전체 환자 수(Initial N): {initial_n}명\n")

    # 추적할 핵심 임상 변수 목록 (analysis_v2.py 기반)
    age_col = 'age_at_initial_pathologic_diagnosis' if mode == "LUAD" else 'age'
    
    clinical_vars = [
        age_col, 
        'ageG', 
        'gender', 
        'pathologic_stage', 
        'pathologic_M', 
        'pathologic_N', 
        'pathologic_T', 
        'SmokingStatus',
        'number_pack_years_smoked',
        'OS', 
        'OS.time', 
        'RFS', 
        'RFS.time'
    ]
    
    # 실제 데이터셋에 존재하는 컬럼만 필터링
    valid_vars = [var for var in clinical_vars if var in df.columns]
    
    df_inspect = df[valid_vars].copy()
    
    # =====================================================================
    # [핵심 수술] analysis_v2.py의 절대 무적 정제 엔진 가동
    # =====================================================================
    na_strings = ['not reported', '[not available]', 'unknown', 'na', 'n/a', '', 'none']
    
    for col in valid_vars:
        if df_inspect[col].dtype == object or isinstance(df_inspect[col].dtype, pd.CategoricalDtype):
            # 1. 스페이스바 공백(예: "   ")을 완벽하게 NaN으로 치환
            df_inspect[col] = df_inspect[col].replace(r'^\s*$', np.nan, regex=True)
            # 2. 분석을 방해하는 가짜 문자열을 NaN으로 강제 소각
            df_inspect[col] = df_inspect[col].apply(
                lambda x: np.nan if pd.isna(x) or str(x).strip().lower() in na_strings else x
            )
            
    # =====================================================================
    # 3. 결측치 통계(N수) 집계
    # =====================================================================
    report_data = []
    
    for col in valid_vars:
        missing_n = df_inspect[col].isna().sum()
        valid_n = initial_n - missing_n
        missing_rate = (missing_n / initial_n) * 100
        
        report_data.append({
            'Clinical Variable': col,
            'Total N': initial_n,
            'Valid N (유효)': valid_n,
            'Missing N (결측)': missing_n,
            'Missing Rate (%)': round(missing_rate, 2)
        })
        
    df_report = pd.DataFrame(report_data)
    
    # 콘솔 출력 (가독성 높은 포맷)
    print(df_report.to_string(index=False))
    print(f"\n▶ 결측치 정밀 추적 완료. (총 {len(valid_vars)}개 변수 검증)")
    
    return df_report

# =====================================================================
# 메인 실행 블록 (LUAD, LUSC 순차적 자동 검증)
# =====================================================================
if __name__ == "__main__":
    # 경로 설정
    super_dir = os.path.dirname(os.path.abspath(__file__))
    database_dir = os.path.join(super_dir, "database")
    result_dir = os.path.join(super_dir, "result_missing_check")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 로그 파일 설정
    if isinstance(sys.stdout, PrintLogger):
        sys.stdout.flush()
    sys.stdout = PrintLogger(os.path.join(result_dir, f"Missing_Values_Log.txt"))
    
    all_reports = {}
    
    for mode in ['LUAD', 'LUSC']:
        master_sav_filename = '(LUAD) lung adenocarcinoma.sav' if mode == 'LUAD' else '(LUSC) lung squamous cell carcinoma.sav'
        master_sav_path = os.path.join(database_dir, master_sav_filename)
        
        if not os.path.exists(master_sav_path):
            print(f"\n[오류] 마스터 임상 파일이 존재하지 않습니다: {master_sav_path}")
            continue
            
        # 데이터 로드
        print(f"\n=== {mode} 마스터 SAV 파일 로드 중... ===")
        df_clin, meta = pyreadstat.read_sav(master_sav_path, user_missing=True)
        
        # 결측치 검증 실행
        df_report = inspect_missing_values(df_clin, mode, result_dir)
        all_reports[mode] = df_report
        
    # 결과를 하나의 엑셀 파일(시트 구분)로 통합 저장
    if all_reports:
        excel_save_path = os.path.join(result_dir, "Missing_Values_Report.xlsx")
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            for mode, df_rep in all_reports.items():
                df_rep.to_excel(writer, sheet_name=f'{mode}_Missing_Check', index=False)
        print(f"\n{'='*60}")
        print(f"✅ 결측치 검증 통합 엑셀 리포트가 저장되었습니다: {excel_save_path}")
        print(f"{'='*60}")