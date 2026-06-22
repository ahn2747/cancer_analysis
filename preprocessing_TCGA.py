import pandas as pd
import numpy as np
import os
import sys

def preprocess_xena_matrix(tpm_file, probe_map_file=None, output_file=None):
    print("=====================================================================")
    print("            TCGA RNA-seq TPM Matrix 전처리 시작")
    print("=====================================================================")
    
    # 1. UCSC Xena TPM 파일 로드
    print(f"1. Xena TPM tsv 파일 로드 중: {tpm_file}")
    try:
        # Xena 데이터는 탭(\t)으로 구분된 큰 파일입니다.
        df = pd.read_csv(tpm_file, sep='\t', index_col=0)
        print(f"   └─ 원본 데이터 로드 완료! (유전자 수: {df.shape[0]}개, 샘플 수: {df.shape[1]}개)")
    except FileNotFoundError:
        print(f"  [오류] 파일을 찾을 수 없습니다: {tpm_file}")
        print("  [안내] 다운로드받으신 Xena tsv 파일명을 확인해 주시고 database 폴더에 넣어주세요.")
        return
        
    # 2. 환자 샘플 ID 필터링 및 12자리 정제 (Tumor 샘플만 추출)
    # TCGA-38-7271-01A -> Tumor (01~09로 시작하는 코드)
    # TCGA-55-6978-11A -> Normal (10~19로 시작하는 코드) -> 제외 처리
    print("\n2. 정상 조직(-11A 등) 제외 및 종양 조직(-01A 등) 필터링 중...")
    tumor_cols = []
    rename_dict = {}
    
    for col in df.columns:
        parts = col.split('-')
        if len(parts) >= 4:
            sample_type = parts[3] # 예: '01A', '11A'
            # 01~09로 시작하는 샘플 유형만 종양(Tumor) 조직입니다.
            if sample_type.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '09')):
                patient_id = '-'.join(parts[:3]) # 앞의 12자리 'TCGA-XX-XXXX'만 추출
                tumor_cols.append(col)
                rename_dict[col] = patient_id
                
    df_tumor = df[tumor_cols].rename(columns=rename_dict)
    
    # 만약 동일 환자에게서 두 개 이상의 Tumor 샘플이 나온 경우 평균값으로 병합
    df_tumor = df_tumor.groupby(level=0, axis=1).mean()
    print(f"   └─ 정제 완료! (분석 대상 종양 환자 수: {df_tumor.shape[1]}명)")
    
    # 3. Ensembl ID를 Gene Symbol로 변환
    # ENSG00000000003.15 -> ENSG00000000003 (소수점 이하 버전 번호 제거)
    print("\n3. Ensembl ID 소수점 제거 및 Gene Symbol 변환 진행 중...")
    df_tumor.index = df_tumor.index.str.split('.').str[0]
    
    if probe_map_file and os.path.exists(probe_map_file):
        print(f"   [옵션 A] 제공된 UCSC Xena Probe Map 파일({probe_map_file})을 사용한 오프라인 매핑...")
        try:
            df_map = pd.read_csv(probe_map_file, sep='\t')
            df_map.columns = [col.lower() for col in df_map.columns]
            df_map['id_clean'] = df_map['id'].str.split('.').str[0]
            
            mapping_dict = dict(zip(df_map['id_clean'], df_map['gene']))
            df_tumor.index = df_tumor.index.map(mapping_dict)
        except Exception as e:
            print(f"   [오류] Probe Map 파일 매핑 실패: {e}")
            return
            
    else:
        print("   [옵션 B] mygene 라이브러리 API를 이용한 온라인 실시간 매핑...")
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            ensembl_ids = df_tumor.index.tolist()
            
            # API를 사용하여 Ensembl ID를 Gene Symbol로 일괄 변환
            queries = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
            
            mapping_dict = {}
            for q in queries:
                if 'symbol' in q:
                    mapping_dict[q['query']] = q['symbol']
            
            df_tumor.index = df_tumor.index.map(mapping_dict)
        except ImportError:
            print("\n  [경고] 온라인 변환을 위한 'mygene' 라이브러리가 설치되어 있지 않습니다.")
            print("  [안내] 터미널에 'pip install mygene'을 실행하시거나,")
            print("  [안내] UCSC Xena에서 다운로드 가능한 'probeMap' 파일을 두 번째 인자로 전달해 주세요.")
            print("  [경고] 유전자 이름 변환을 실패하여 전처리를 중단합니다.")
            return

    # 매핑에 실패한 NaN 값 제거 및 중복된 Gene Symbol 처리
    df_tumor = df_tumor[df_tumor.index.notna()]
    # 동일한 Gene Symbol이 존재하는 경우 평균값으로 병합
    df_tumor = df_tumor.groupby(level=0).mean()
    print(f"   └─ 최종 매핑 성공한 유전자 수: {df_tumor.shape[0]}개")
    
    # 4. 결과 파일 저장
    if not output_file:
        output_file = tpm_file.replace('.tsv', '_cleaned.csv')
        
    df_tumor.to_csv(output_file)
    print("\n=====================================================================")
    print(f"✅ 전처리 완료! GSEA 입력용 발현량 파일 저장됨: {output_file}")
    print("=====================================================================")

if __name__ == "__main__":
    super_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ⚠️ [설정 사항] UCSC Xena에서 다운받은 실제 파일명에 맞게 아래를 수정해 주세요.
    xena_tpm_file = os.path.join(super_dir, "database", "TCGA-LUSC-tpm.tsv") 
    
    # ⚠️ [옵션] 인터넷 연결이 어려운 경우, UCSC Xena에서 해당 데이터셋 바로 옆에 제공하는
    # 'probeMap' 파일(예: gencode.v23.annotation.gene.probeMap)을 다운받아 아래 경로에 두세요.
    probe_map = os.path.join(super_dir, "database", "gencode.v23.annotation.gene.probeMap")
    
    # 최종 결과물로 생성될 파일명 (GSEA 분석 파이프라인의 expr_file_path와 동일한 이름)
    output_csv = os.path.join(super_dir, "database", "TCGA_LUSC_RNAseq_Expression.csv")
    
    if os.path.exists(xena_tpm_file):
        preprocess_xena_matrix(
            tpm_file=xena_tpm_file, 
            probe_map_file=probe_map if os.path.exists(probe_map) else None, 
            output_file=output_csv
        )
    else:
        print(f"\n[안내] database 폴더 안에 '{os.path.basename(xena_tpm_file)}' 파일이 존재하지 않습니다.")
        print("Xena에서 받으신 tsv 파일 이름을 위의 xena_tpm_file 변수에 설정되어 있는 이름과 일치시켜 주세요.")