import pandas as pd
import os

def convert_tcga_txt_to_csv(input_txt_path, output_csv_path):
    print(f"--- TCGA 임상 텍스트 데이터 -> CSV 변환 시작 ---")
    print(f"  [진단 로그] 타겟 입력 파일: {input_txt_path}")
    
    try:
        # 1. TSV(Tab-Separated Values) 형식의 텍스트 파일 로드
        # TCGA 데이터는 쉼표가 아닌 탭(\t)으로 구분되어 있습니다.
        df = pd.read_csv(input_txt_path, sep='\t', header=0, low_memory=False)
        
        # 2. 외과적 절제술 (불필요한 메타데이터 2줄 제거)
        # TCGA 파일의 두 번째, 세 번째 줄은 변수 설명이므로 분석에 불필요합니다.
        # index 2번부터 끝까지 잘라내어 순수 환자 데이터만 남깁니다.
        df_clean = df.iloc[2:].reset_index(drop=True)
        
        # 3. 순수 CSV 파일로 추출
        # 한글이나 특수문자가 깨지지 않도록 utf-8-sig 인코딩을 적용합니다.
        df_clean.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"  [성공] 데이터 정제 및 변환 완료!")
        print(f"  [진단 로그] 확보된 총 환자 수: {len(df_clean)}명")
        print(f"  [진단 로그] 출력 파일 저장 완료: {output_csv_path}\n")
        
    except Exception as e:
        print(f"  [오류] 데이터 변환 중 치명적 문제가 발생했습니다: {e}")

if __name__ == "__main__":
    # =====================================================================
    # 폴더 구조 자동 인식 (super_dir -> database)
    # =====================================================================
    try:
        super_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        super_dir = os.path.abspath(".")

    # 우리의 데이터베이스 폴더 지정
    db_dir = os.path.join(super_dir, "database")
    
    # 폴더가 없다면 생성 (안전장치)
    os.makedirs(db_dir, exist_ok=True)

    for i in ["nationwidechildrens.org_clinical_follow_up_v1.0_read"]:

        # 타겟 파일 경로 설정 (database 폴더 내부)    
        input_file = os.path.join(db_dir, f"{i}.txt")
        output_file = os.path.join(db_dir, "TCGA_READ_Clinical_FollowUp_Cleaned.csv")
        
        # 실행부
        if os.path.exists(input_file):
            convert_tcga_txt_to_csv(input_file, output_file)
        else:
            print(f"🚨 [경고] '{input_file}' 파일이 존재하지 않습니다.")
            print(f"   '{db_dir}' 폴더 안에 원본 txt 파일이 있는지 다시 한번 확인해 주십시오!")