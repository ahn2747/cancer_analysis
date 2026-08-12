# ... existing code ...
import os
import pandas as pd
import pyreadstat
import re

#directory setting
super_dir = os.path.dirname(os.path.abspath(__file__))
database_dir = os.path.join(super_dir, "database")

def export_sav_to_csv(sav_filepath, db_dir):
    """
    데이터베이스 폴더에 있는 sav 파일을 읽어 열람 가능한 csv 파일로 변환하여 저장합니다.
    """
    print("\n[작업 시작] SAV 파일 CSV 변환...")
    
    try:
        # sav 파일 로드
        df, meta = pyreadstat.read_sav(sav_filepath)
        print(f" - SAV 파일 로드 완료: {sav_filepath}")
    except Exception as e:
        print(f" - [오류] SAV 파일을 읽는 중 문제가 발생했습니다: {e}")
        return None
        
    # 열람 가능한 csv 파일로 저장 (utf-8-sig 인코딩)
    base_name = os.path.basename(sav_filepath).replace('.sav', '')
    csv_filename = f"{base_name}_readable.csv"
    csv_filepath = os.path.join(db_dir, csv_filename)
    
    df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    print(f" - CSV 변환 및 저장 완료: {csv_filepath}")
    
    return csv_filepath

def extract_genes_from_sav(sav_filepath, db_dir, target_genes=None):
    """
    SAV 파일의 컬럼명을 분석하여 데이터셋에 존재하는 유전자 목록을 추출하고 txt로 저장합니다.
    (데이터 로딩 시간을 줄이기 위해 pyreadstat의 메타데이터만 읽어옵니다.)
    """
    print("\n[작업 시작] SAV 파일 내 유전자 목록 별도 추출 및 TXT 저장...")
    try:
        # metadataonly=True 속성을 통해 데이터 전체가 아닌 컬럼명만 매우 빠르게 가져옵니다.
        _, meta = pyreadstat.read_sav(sav_filepath, metadataonly=True)
        df_columns = meta.column_names
    except Exception as e:
        print(f" - [오류] SAV 파일 컬럼을 읽는 중 문제가 발생했습니다: {e}")
        return None
        
    extracted_genes = set()
    
    # 1. 사전에 정의된 target_genes 리스트가 있다면 매칭
    if target_genes:
        matched = [gene for gene in target_genes if gene in df_columns]
        extracted_genes.update(matched)
        
    # 2. 데이터 구조 기반 자동 추출 (스마트 파싱)
    for col in df_columns:
        lower_col = col.lower()
        if any(keyword in lower_col for keyword in ["expression", "expresion", "expressino"]):
            # 대소문자 구분 없이 위 키워드들을 찾아 삭제(공백 치환) 후 유전자명 획득
            gene_name = re.sub(r'(?i)expression|expresion|expressino', '', col).strip()
            if gene_name:
                extracted_genes.add(gene_name)
                
    if not extracted_genes:
        print(" - 추출할 유전자를 데이터셋에서 찾지 못했습니다.")
        return None
        
    # 알파벳 순으로 깔끔하게 정렬
    final_genes = sorted(list(extracted_genes))
    
    # txt 파일로 저장 (이름을 'gene_list.txt'로 고정, 기존 파일 존재 시 덮어쓰기)
    txt_filename = "gene_list.txt"
    txt_filepath = os.path.join(db_dir, txt_filename)
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write("=== 분석 데이터셋 포함 자동 추출 유전자 목록 ===\n")
        f.write(f"총 추출된 유전자 수: {len(final_genes)}개\n")
        f.write("-" * 40 + "\n")
        for gene in final_genes:
            f.write(f"{gene}\n")
            
    print(f" - 유전자 목록 TXT 저장 완료 (총 {len(final_genes)}개 추출, 덮어쓰기 적용): {txt_filepath}")
    return txt_filepath

def export_csv_to_sav(csv_filepath, db_dir, sav_filename=None):
    """
    CSV 파일을 읽어 SPSS 형태의 SAV 파일로 변환하여 저장합니다.
    """
    print("\n[작업 시작] CSV 파일 SAV 변환...")
    try:
        # pandas로 CSV 로드 (utf-8-sig 우선, 실패시 cp949 인코딩 대처)
        try:
            df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_filepath, encoding='cp949')
            
        print(f" - CSV 파일 로드 완료: {csv_filepath}")
        
        # 저장할 파일명 설정
        if sav_filename is None:
            base_name = os.path.basename(csv_filepath).replace('.csv', '')
            sav_filename = f"{base_name}.sav"
            
        out_sav_filepath = os.path.join(db_dir, sav_filename)
        
        # pyreadstat으로 DataFrame을 SAV 파일로 저장
        pyreadstat.write_sav(df, out_sav_filepath)
        print(f" - SAV 변환 및 저장 완료: {out_sav_filepath}")
        
        return out_sav_filepath
    except Exception as e:
        print(f" - [오류] CSV 파일을 SAV로 변환하는 중 문제가 발생했습니다: {e}")
        return None

def main():
    # ... existing code (디렉토리 설정, 더미데이터 생성 등) ...
    
    # [참고용] 타겟 유전자 리스트가 기존에 아래와 같이 정의되어 있다고 가정합니다.
    # target_genes = ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'] 
    
    # ... existing code (데이터 병합 실행) ...
    # merged_sav_path = load_and_merge_data(...)
    
    # =============== [신규 추가 위치] ===============
    # 데이터 병합이 끝난 직후, 해당 sav 파일 경로를 전달하여 csv 변환 및 txt 저장을 수행합니다.
    sav_path = os.path.join(database_dir, "COAD.sav")
    
    # 1. SAV -> CSV 변환 실행
    exported_csv_path = export_sav_to_csv(
        sav_filepath=sav_path, 
        db_dir=database_dir
    )
    
    # 2. 유전자 추출 및 TXT 저장 실행 (활성화 완료, gene_list.txt로 생성)
    extract_genes_from_sav(
        sav_filepath=sav_path, 
        db_dir=database_dir,
        target_genes=[]
    )
    
    # [참고] 방금 생성한 CSV를 다시 새로운 SAV로 변환할 때 사용하는 예시 코드 (필요시 주석 해제)
    # if exported_csv_path:
    #     export_csv_to_sav(
    #         csv_filepath=exported_csv_path, 
    #         db_dir=database_dir, 
    #         sav_filename="converted_back.sav"
    #     )
    # ===============================================
    # ... existing code (이후 통계 분석들: analyze_chi_square, analyze_bivariate_correlation 등) ...


if __name__ == "__main__":
    main()