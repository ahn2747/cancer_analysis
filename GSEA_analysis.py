import pandas as pd
import numpy as np
import pyreadstat
import gseapy as gp
from gseapy.plot import gseaplot
import matplotlib.pyplot as plt
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

# =====================================================================
# 1. 데이터 전처리 함수 (발현량 데이터와 임상 그룹 정보 병합)
# =====================================================================
def prepare_gsea_data(expr_file, clin_file, group_col='SignatureGroup'):
    print(f"--- 1. GSEA 데이터 준비 시작 ---")
    
    print(f"  [진단 로그] 발현량 데이터 로드 중: {expr_file}")
    try:
        df_expr = pd.read_csv(expr_file, index_col=0) 
    except FileNotFoundError:
        print(f"  [오류] 발현량 파일을 찾을 수 없습니다: {expr_file}")
        return None, None
        
    print(f"  [진단 로그] 임상 그룹 데이터 로드 중: {clin_file}")
    try:
        df_clin, meta = pyreadstat.read_sav(clin_file)
    except FileNotFoundError:
        print(f"  [오류] 임상 파일을 찾을 수 없습니다: {clin_file}")
        return None, None

    if group_col not in df_clin.columns or 'sampleID' not in df_clin.columns:
        print(f"  [오류] 임상 데이터에 '{group_col}' 또는 'sampleID' 컬럼이 없습니다.")
        return None, None

    df_clin = df_clin[['sampleID', group_col]].dropna()
    common_samples = list(set(df_expr.columns).intersection(set(df_clin['sampleID'])))
    
    if not common_samples:
        print("  [오류] 발현량 데이터와 임상 데이터 간에 일치하는 환자 ID(sampleID)가 없습니다.")
        return None, None
        
    print(f"  [진단 로그] 공통 환자 샘플 수: {len(common_samples)} 명")

    df_expr = df_expr[common_samples]
    cls_dict = dict(zip(df_clin['sampleID'], df_clin[group_col]))
    
    valid_samples = []
    for sample in df_expr.columns:
        val = cls_dict.get(sample)
        if val is not None and pd.notna(val) and str(val).strip() != "":
            valid_samples.append(sample)
            
    if len(valid_samples) < 2:
        print("  [오류] 분석 가능한 샘플이 부족합니다.")
        return None, None
        
    for sample in valid_samples:
        cls_dict[sample] = str(cls_dict[sample]).strip()
        
    valid_samples.sort(key=lambda x: cls_dict[x])
    pheno_list = [cls_dict[sample] for sample in valid_samples]
    
    group_counts = pd.Series(pheno_list).value_counts()
    print(f"  [진단 로그] 감지된 환자 그룹 분포:\n{group_counts.to_string()}")
    
    if len(group_counts) > 2:
        print(f"  [경고] 그룹이 3개 이상 감지되었습니다. 상위 2개 그룹만 남깁니다.")
        top_2_groups = group_counts.index[:2].tolist()
        valid_samples = [s for s in valid_samples if cls_dict[s] in top_2_groups]
        pheno_list = [cls_dict[s] for s in valid_samples]
    elif len(group_counts) < 2:
        print(f"  [오류] 비교할 대상이 없어 GSEA를 실행할 수 없습니다.")
        return None, None

    df_expr = df_expr[valid_samples]
    print("--- GSEA 데이터 준비 완료 ---\n")
    return df_expr, pheno_list

# =====================================================================
# 2. CHIP 붕괴(Collapse) 적용 함수
# =====================================================================
def apply_chip_collapse(df_expr, chip_file_path):
    print(f"--- CHIP Collapse 적용 시작 ---")
    print(f"  [안내] CHIP 파일 로드: {chip_file_path}")
    try:
        df_chip = pd.read_csv(chip_file_path, sep='\t')
        
        # 칩 파일 구조: 첫 번째 열이 프로브/기존심볼, 두 번째 열이 매핑될 타겟 심볼로 가정
        col_probe = df_chip.columns[0]
        col_symbol = df_chip.columns[1]
        
        mapping_dict = dict(zip(df_chip[col_probe], df_chip[col_symbol]))
        
        # 원본 데이터 복사 및 인덱스 맵핑
        df_collapsed = df_expr.copy()
        df_collapsed.index = df_collapsed.index.map(mapping_dict)
        
        # 맵핑되지 않은 (CHIP에 없는) 유전자 드랍
        df_collapsed = df_collapsed[df_collapsed.index.notna()]
        
        # 중복 유전자 처리 (Max_probe 방식: 발현량 평균이 가장 높은 행만 남김)
        df_collapsed['__mean'] = df_collapsed.mean(axis=1)
        df_collapsed = df_collapsed.sort_values('__mean', ascending=False)
        df_collapsed = df_collapsed[~df_collapsed.index.duplicated(keep='first')]
        df_collapsed = df_collapsed.drop(columns=['__mean'])
        
        print(f"  [완료] CHIP 파일 기반 매핑 및 중복 제거 완료. 남은 유전자 수: {len(df_collapsed)}개\n")
        return df_collapsed
    except Exception as e:
        print(f"  [오류] CHIP 적용 실패. 원본 데이터로 진행합니다. 에러: {e}\n")
        return df_expr

# =====================================================================
# 3. GSEA 실행 및 플롯 생성 함수
# =====================================================================
def run_gsea_analysis(df_expr, pheno_list, gene_set_name, save_dir, prefix="", remove_genes=None):
    print(f"--- GSEA 분석 실행 (Gene Set: {gene_set_name}) ---")
    
    out_dir = os.path.join(save_dir, f"{prefix}{gene_set_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    print("  [진단 로그] 발현량 데이터 스케일링 생략 (입력값이 이미 Log2 변환된 데이터로 간주함)...")
    df_expr_log = df_expr
    
    gsea_target_sets = gene_set_name
    if remove_genes and isinstance(remove_genes, list) and len(remove_genes) > 0:
        print(f"  [안내] 특정 유전자 제거 진행 중: {remove_genes}")
        try:
            lib_dict = gp.get_library(name=gene_set_name, organism='human')
            modified_dict = {}
            for term, genes in lib_dict.items():
                filtered_genes = [g for g in genes if g not in remove_genes]
                modified_dict[term] = filtered_genes
            gsea_target_sets = modified_dict
            print(f"  [완료] 지정된 유전자들을 분석 데이터베이스에서 제거했습니다.")
        except Exception as e:
            print(f"  [경고] 유전자 제거 실패 (원본 사용): {e}")
            gsea_target_sets = gene_set_name
    
    try:
        gs_res = gp.gsea(data=df_expr_log,
                         gene_sets=gsea_target_sets,
                         cls=pheno_list,
                         permutation_num=1000, 
                         outdir=os.path.join(out_dir, "temp"), 
                         no_plot=True, 
                         method='signal_to_noise', 
                         processes=4, 
                         format='png')
                         
        if gs_res is None:
            print(f"  [경고] GSEA 분석 결과 객체가 NoneType입니다.")
            return pd.DataFrame()
            
        results_df = getattr(gs_res, 'res2d', None)
        if results_df is None or results_df.empty:
            print(f"  [경고] {gene_set_name}에서 유의미한 결과를 찾지 못했습니다.")
            return pd.DataFrame()
            
        report_csv_path = os.path.join(out_dir, "gseapy.phenotype.gsea.report.csv")
        if os.path.exists(report_csv_path):
            try: os.remove(report_csv_path)
            except: pass
        results_df.to_csv(report_csv_path, index=False, encoding='utf-8-sig')
        
        report_df_cleaned = results_df.copy()
        target_cols = ['ES', 'NES', 'NOM p-val', 'FDR q-val']
        for col in target_cols:
            if col in report_df_cleaned.columns:
                report_df_cleaned[col] = pd.to_numeric(report_df_cleaned[col], errors='coerce')
                report_df_cleaned[col] = report_df_cleaned[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
                
        cleaned_report_csv_path = os.path.join(out_dir, "gseapy.phenotype.gsea.report_cleaned.xlsx")
        if os.path.exists(cleaned_report_csv_path):
            try: os.remove(cleaned_report_csv_path)
            except: pass
        report_df_cleaned.to_excel(cleaned_report_csv_path, index=False)
        print(f"  [저장] GSEA Report 파일 처리 완료")
            
        sig_results = results_df[results_df['FDR q-val'] < 2.00].sort_values(by='NES', ascending=False)
        
        excel_path = os.path.join(out_dir, f"{prefix}GSEA_{gene_set_name}_results.xlsx")
        results_df.to_excel(excel_path, index=False)
        
        top_pathways = sig_results['Term'].head(5).tolist()
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        for term in top_pathways:
            safe_term = term.replace('/', '_').replace(':', '_').replace(' ', '_')
            plot_path = os.path.join(out_dir, f"{prefix}GSEA_Plot_{safe_term}.png")
            try:
                if hasattr(gs_res, 'results') and gs_res.results and term in gs_res.results:
                    gseaplot(rank_metric=gs_res.ranking, term=term, **gs_res.results[term], ofname=plot_path)
            except Exception as e:
                pass
                
        return results_df
        
    except Exception as e:
        print(f"  [오류] GSEA 분석 중 에러: {e}")
        return pd.DataFrame()


# =====================================================================
# 4. 메인 실행 블록
# =====================================================================
if __name__ == "__main__":
    try:
        import gseapy
    except ImportError:
        print("\n[오류] gseapy 라이브러리가 없습니다. 'pip install gseapy'를 실행하세요.\n")
        sys.exit(1)

    super_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(super_dir, "result_gsea")
    os.makedirs(result_dir, exist_ok=True)
    
    # [옵션 설정]
    # ANALYSIS_MODE: "raw" (모든 유전자 사용) 또는 "chip" (CHIP 파일 필터링 적용)
    ANALYSIS_MODE = "chip" 
    
    # CHIP 파일 설정 (ANALYSIS_MODE가 "chip"일 때만 사용됩니다)
    CHIP_FILE_NAME = "Human_Gene_Symbol_with_Remapping_MSigDB.v2026.1.Hs.chip"
    
    CUSTOM_FILE_PREFIX = f"FAT1_{ANALYSIS_MODE}_" 
    
    if isinstance(sys.stdout, PrintLogger):
        sys.stdout.flush()
    sys.stdout = PrintLogger(os.path.join(result_dir, f"{CUSTOM_FILE_PREFIX}gsea_log.txt"))
    
    print(f"\n{'='*60}")
    print(f"          자동화 GSEA 시작 (모드: {ANALYSIS_MODE.upper()})")
    print(f"{'='*60}")
    
    mode = 'LUAD' 
    expr_file_path = os.path.join(super_dir, "database", f"TCGA_{mode}_RNAseq_Expression.csv")
    clin_file_path = os.path.join(super_dir, "database", '(LUAD) lung adenocarcinoma.sav' if mode == 'LUAD' else '(LUSC) lung squamous cell carcinoma.sav')
    group_column_name = "FAT1group"
    genes_to_remove = [] 
    
    if not os.path.exists(expr_file_path):
        print(f"\n[오류] 발현량 파일을 찾을 수 없습니다: {expr_file_path}")
    else:
        df_expr, pheno_list = prepare_gsea_data(expr_file_path, clin_file_path, group_col=group_column_name)
        
        if df_expr is not None and pheno_list is not None:
            # CHIP 모드 적용
            if ANALYSIS_MODE.lower() == "chip":
                chip_path = os.path.join(super_dir, "database", CHIP_FILE_NAME)
                if os.path.exists(chip_path):
                    df_expr = apply_chip_collapse(df_expr, chip_path)
                else:
                    print(f"  [경고] 지정된 CHIP 파일이 없습니다: {chip_path}")
                    print("  [안내] Raw 모드로 자동 전환하여 분석을 강행합니다.\n")
            
            gene_sets_to_test = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human']
            r_dir = os.path.join(result_dir, f"{mode}_{CUSTOM_FILE_PREFIX.rstrip('_')}")
            os.makedirs(r_dir, exist_ok=True)

            for gs_name in gene_sets_to_test:
                print(f"\n>> {gs_name} 분석 진행...")
                df_res = run_gsea_analysis(df_expr, pheno_list, gs_name, r_dir, prefix=f"{mode}_", remove_genes=genes_to_remove)
                
    print(f"\n{'='*60}")
    print(f"          GSEA 파이프라인 실행 완료")
    print(f"{'='*60}")