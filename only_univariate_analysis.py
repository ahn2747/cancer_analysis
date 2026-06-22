import pandas as pd
import numpy as np
import pyreadstat
import scipy.stats as stats
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
import os
import sys
import shutil
import glob
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
# 1. 데이터 처리 및 병합
# =====================================================================
def load_and_merge_data(target, onco_file, clin_file):
    print(f"--- 1. 데이터 처리 및 병합 시작 (Target: {target}) ---")
    
    df_onco = pd.read_csv(onco_file, usecols=['Patient', 'Expression', 'Group'])
    df_onco.rename(columns={
        'Patient': 'sampleID',
        'Expression': f'{target}expression',
        'Group': f'{target}group'
    }, inplace=True)
    
    df_clin, meta = pyreadstat.read_sav(clin_file)
    
    new_cols = [f'{target}expression', f'{target}group']
    cols_to_drop = [col for col in new_cols if col in df_clin.columns]
    if cols_to_drop:
        df_clin.drop(columns=cols_to_drop, inplace=True)
    
    df_merged = pd.merge(df_clin, df_onco, on='sampleID', how='left')
    
    pyreadstat.write_sav(df_merged, clin_file)
    csv_path = clin_file.replace('.sav', '.csv')
    df_merged.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return df_merged

# =====================================================================
# 2. 기초 및 다변량 분석 파이프라인 (Excel 표 생성 로직 추가)
# =====================================================================
def analyze_chi_square(df, target_col='gene_group', row_vars=None):
    """카이제곱 분석 수행 및 엑셀용 논문 포맷 Table 1 DataFrame 반환"""
    print("--- 2-1. 예후 분석 (Chi-square Test) ---")
    if row_vars is None:
        row_vars = ['ageG', 'gender', 'pathologic_stage', 'pathologic_M', 'pathologic_N', 'pathologic_T', 'SmokingStatus']
    
    # 그룹 범주 파악 (예: High, Low)
    target_cats = sorted(df[target_col].dropna().unique())
    if len(target_cats) == 0:
        target_cats = ['Group1', 'Group2']
        
    table1_data = []
    
    for var in row_vars:
        crosstab_stat = pd.crosstab(df[var], df[target_col])
        try:
            chi2, p_val, dof, expected = stats.chi2_contingency(crosstab_stat)
            p_val_formatted = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        except ValueError:
            p_val_formatted = "N/A"
            
        crosstab_print = pd.crosstab(df[var], df[target_col], margins=True, margins_name="전체")
        print(f"[{target_col} vs {var}]")
        print(crosstab_print)
        print(f"Chi-square: {chi2:.4f}, p-value: {p_val_formatted}\n")
        
        # 엑셀 표(Table 1) 조립 로직
        table1_data.append([var] + [""] * len(target_cats) + [""]) # 변수명 타이틀 행
        
        for i, (idx, row) in enumerate(crosstab_stat.iterrows()):
            row_data = [f"  {idx}"] # 범주명 (들여쓰기)
            for cat in target_cats:
                row_data.append(row.get(cat, 0))
            
            # 첫 번째 범주 행에만 p-value 표시
            row_data.append(p_val_formatted if i == 0 else "")
            table1_data.append(row_data)

    col_names = ['Clinical Characteristics'] + target_cats + ['P-value']
    df_table1 = pd.DataFrame(table1_data, columns=col_names)
    return df_table1


def analyze_bivariate_correlation(df, gene_list):
    """이변량 분석 수행 및 엑셀용 논문 포맷 Table 2 DataFrame 반환"""
    print("--- 2-2. 이변량 상관 분석 (Pearson 행렬) ---")
    
    valid_data = df[gene_list].dropna()
    
    # 콘솔 출력용
    pearson_matrix = valid_data.corr(method='pearson')
    print("[Pearson 상관계수 행렬]")
    print(pearson_matrix.round(3))
    print("\n")
    
    # 엑셀 표(Table 2) 조립 로직 (R, P 교차 출력)
    table2_data = []
    
    for i in gene_list:
        row_r = [i, 'R']
        row_p = ['', 'P']
        
        for j in gene_list:
            if i == j:
                row_r.append("1")
                row_p.append("")
            else:
                r_val, p_val = stats.pearsonr(valid_data[i], valid_data[j])
                row_r.append(f"{r_val:.3f}")
                row_p.append(f"{p_val:.3f}" if p_val >= 0.001 else "<0.001")
                
        table2_data.append(row_r)
        table2_data.append(row_p)
        
    col_names = ['Variable', 'Stat'] + gene_list
    df_table2 = pd.DataFrame(table2_data, columns=col_names)
    return df_table2


def analyze_glm_multivariate(df, expr_col='target_expression', group_col='gene_group'):
    print("--- 2-3. 다변량 일반선형모형 (GLM / OLS) ---")
    vars_to_use = [expr_col, group_col, 'age_at_initial_pathologic_diagnosis', 'gender', 'pathologic_stage', 'number_pack_years_smoked']
    df_clean = df[vars_to_use].dropna()
    formula = f"{expr_col} ~ C({group_col}) + age_at_initial_pathologic_diagnosis + C(gender) + C(pathologic_stage) + number_pack_years_smoked"
    model = smf.ols(formula=formula, data=df_clean).fit()
    print(model.summary())
    print("\n")


def analyze_kaplan_meier(target, df, mode, save_dir, group_col='gene_group'):
    print("--- 2-4. Kaplan-Meier 생존분석 ---")
    df_km = df.dropna(subset=[group_col])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    kmf = KaplanMeierFitter()
    groups = df_km[group_col].unique()
    
    for group in groups:
        mask = (df_km[group_col] == group)
        time = df_km.loc[mask, 'OS.time'].dropna()
        event = df_km.loc[mask, 'OS'].dropna()
        if len(time) > 0:
            kmf.fit(time, event_observed=event, label=f"{group} (n={len(time)})")
            kmf.plot_survival_function(ax=axes[0], ci_show=False)
            
    axes[0].set_title(f'Overall Survival (OS) by {group_col} - {target}({mode})')
    axes[0].set_xlabel('Time (Months/Days)')
    axes[0].set_ylabel('Survival Probability')
    
    for group in groups:
        mask = (df_km[group_col] == group)
        time = df_km.loc[mask, 'RFS.time'].dropna()
        event = df_km.loc[mask, 'RFS'].dropna()
        if len(time) > 0:
            kmf.fit(time, event_observed=event, label=f"{group} (n={len(time)})")
            kmf.plot_survival_function(ax=axes[1], ci_show=False)
            
    axes[1].set_title(f'Relapse-Free Survival (RFS) by {group_col} - {target}({mode})')
    axes[1].set_xlabel('Time (Months/Days)')
    axes[1].set_ylabel('Relapse-Free Probability')
    
    plt.tight_layout()
    km_plot_path = os.path.join(save_dir, f'{target}_{mode}_Kaplan_Meier_Curves.png')
    plt.savefig(km_plot_path)
    print(f"Kaplan-Meier 생존 곡선 이미지 저장 완료: '{km_plot_path}'\n")


def analyze_cox_regression(target, df, mode, save_dir, group_col='gene_group'):
    """Cox 회귀분석 수행 및 엑셀용 논문 포맷 Table 3 DataFrame 반환"""
    print("--- 2-5. Cox 회귀분석 ---")
    if mode == "LUAD":
        cox_cols = ['OS.time', 'OS', group_col, 'age_at_initial_pathologic_diagnosis', 'gender', 'pathologic_stage', 'number_pack_years_smoked']
    if mode == "LUSC":
        cox_cols = ['OS.time', 'OS', group_col, 'age', 'gender', 'pathologic_stage', 'number_pack_years_smoked']
    df_cox = df[cox_cols].dropna()
    df_cox_dummy = pd.get_dummies(df_cox, columns=[group_col, 'gender', 'pathologic_stage'], drop_first=True)
    cph = CoxPHFitter()
    
    df_table3 = pd.DataFrame() # 엑셀 저장용 빈 데이터프레임 초기화
    
    try:
        cph.fit(df_cox_dummy, duration_col='OS.time', event_col='OS')
        print(cph.summary[['exp(coef)', 'p']]) 
        
        # 엑셀 표(Table 3) 조립 로직: HR, 95% CI, P-value 추출
        summary_df = cph.summary
        df_table3 = pd.DataFrame({
            'Clinical Variable': summary_df.index,
            'Hazard Ratio (HR)': summary_df['exp(coef)'].round(3),
            '95% CI Lower': summary_df['exp(coef) lower 95%'].round(3),
            '95% CI Upper': summary_df['exp(coef) upper 95%'].round(3),
            'P-value': summary_df['p'].apply(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
        })
        
        plt.figure(figsize=(10, 6))
        cph.plot()
        plt.title(f'Cox Regression - Forest Plot - {target}({mode})')
        plt.tight_layout()
        cox_plot_path = os.path.join(save_dir, f'{target}_{mode}_Cox_Forest_Plot.png')
        plt.savefig(cox_plot_path)
        print(f"Cox Forest Plot 이미지 저장 완료: '{cox_plot_path}'\n")
        
    except Exception as e:
         print(f"Cox 분석 중 오류 발생: {e}")
         
    return df_table3

# =====================================================================
# 3. 메인 실행 블록 (Main Execution & Auto Batch)
# =====================================================================
if __name__ == "__main__":
    
    mode = input("분석할 암종을 입력하세요 (LUAD 또는 LUSC): ").strip().upper()
    if mode not in ['LUAD', 'LUSC']:
        print("지원하지 않는 암종입니다. 프로그램을 종료합니다.")
        sys.exit()

    try:
        super_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        super_dir = os.path.abspath(".")

    database_dir = os.path.join(super_dir, "database")
    input_csv_dir = os.path.join(super_dir, f"input_{mode}_csv")
    completed_csv_dir = os.path.join(super_dir, "completed_csv")
    result_dir = os.path.join(super_dir, "result")
    plot_base_dir = os.path.join(super_dir, "plot")

    for d in [database_dir, input_csv_dir, completed_csv_dir, result_dir, plot_base_dir]:
        os.makedirs(d, exist_ok=True)

    result_txt_path = os.path.join(result_dir, f"result_{mode}.txt")
    sys.stdout = PrintLogger(result_txt_path)
    print(f"========== {mode} 자동화 분석 파이프라인 시작 ==========\n")

    if mode == 'LUAD':
        master_sav_filename = '(LUAD) lung adenocarcinoma.sav'
    else:
        master_sav_filename = '(LUSC) lung squamous cell carcinoma.sav'
        
    master_sav_path = os.path.join(database_dir, master_sav_filename)

    if not os.path.exists(master_sav_path):
        print(f"오류: 마스터 임상 파일이 존재하지 않습니다. 경로를 확인하세요: {master_sav_path}")
        sys.exit()

    target_csv_files = glob.glob(os.path.join(input_csv_dir, "*.csv"))
    if len(target_csv_files) == 0:
        print(f"'{input_csv_dir}' 폴더에 분석할 CSV 파일이 없습니다.")
    
    for csv_file in target_csv_files:
        file_name = os.path.basename(csv_file)
        target_gene = file_name.replace('.csv', '').split('_')[-1]
        expr_col_name = f"{target_gene}expression"
        group_col_name = f"{target_gene}group"
        
        print(f"\n[{target_gene} 유전자 분석 시작]")
        
        target_plot_dir = os.path.join(plot_base_dir, f"{target_gene}_plot")
        if os.path.exists(target_plot_dir):
            shutil.rmtree(target_plot_dir)
        os.makedirs(target_plot_dir)

        # 1. 병합
        merged_df = load_and_merge_data(target=target_gene, onco_file=csv_file, clin_file=master_sav_path)
        
        # 2. 분석 및 엑셀 표 데이터 생성
        # df_table1 = analyze_chi_square(merged_df, target_col=group_col_name)
        
        if mode == "LUAD":
            target_genes_for_corr = ["LOXL2expression", "ITGB1expression", "PLAURexpression", "SNAI1expression", "VEGFCexpression", 'KrasExpression', 'TP53Expression', 'ALKExpression', 'BRAFExpression']
        if mode == "LUSC":
            target_genes_for_corr = ["LOXL2expression", "ITGB1expression", "PLAURexpression", "SNAI1expression", "VEGFCexpression", 'TP53Expression', 'CDKN2AExpression', 'SOX2Expression', 'PIK3CAExpression', 'NOTCH1Expression']
        df_table2 = analyze_bivariate_correlation(merged_df, gene_list=target_genes_for_corr)
        
        # 다변량 분석 (선택적 실행)
        # analyze_glm_multivariate(merged_df, expr_col=expr_col_name, group_col=group_col_name)
        
        # 생존 & Cox 분석
        # analyze_kaplan_meier(target_gene, merged_df, mode, target_plot_dir, group_col=group_col_name)
        # df_table3 = analyze_cox_regression(target_gene, merged_df, mode, target_plot_dir, group_col=group_col_name)
        
        # 3. 분석 결과를 Excel 파일로 예쁘게 저장! (저장 시점을 모든 분석이 끝난 후로 이동)
        excel_save_path = os.path.join(result_dir, f"univariate_{mode}_Tables.xlsx")
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            # df_table1.to_excel(writer, sheet_name='Table 1 (Chi-square)', index=False)
            df_table2.to_excel(writer, sheet_name='Table 2 (Correlation)', index=False)
            # if not df_table3.empty:
            #     df_table3.to_excel(writer, sheet_name='Table 3 (Cox)', index=False)
        print(f"✅ 논문용 표 엑셀 파일 저장 완료: '{excel_save_path}'")
        
        # 처리 완료 파일 이동
        completed_path = os.path.join(completed_csv_dir, file_name)
        shutil.move(csv_file, completed_path)
        print(f"'{file_name}' 분석 완료 및 completed_csv 폴더로 이동됨.\n")
        print("="*60)
        
    print(f"\n=== 모든 {mode} 생물통계 파이프라인이 성공적으로 완료되었습니다 ===")
    print(f"전체 결과 로그는 '{result_txt_path}'에 저장되었습니다.")