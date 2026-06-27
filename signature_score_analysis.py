import pandas as pd
import numpy as np
import pyreadstat
import scipy.stats as stats
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
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
# 핵심: Signature Score 계산 및 그룹화
# =====================================================================
def get_signature_group(df, gene_list, sav_path):
    # 5개 유전자의 Z-score 평균을 구해 High/Low 그룹 생성 및 기존 DB 병합
    z_scores = pd.DataFrame()
    for gene in gene_list:
        col = f"{gene}expression"
        if col in df.columns:
            val = df[col].fillna(df[col].mean())
            z_scores[gene] = (val - val.mean()) / val.std()
    
    if z_scores.empty:
        print("[경고] 시그니처를 구성할 수 있는 유전자 데이터가 없습니다.")
        return df

    # 생성할 변수 이름 만들기
    gene_name_list = '_'.join(gene_list)
    score_col = f'SignatureScore_{gene_name_list}'
    group_col = f'SignatureGroup_{gene_name_list}'

    # 기존 SAV 파일에 동일한 유전자 컬럼이 있다면 충돌 방지를 위해 먼저 삭제
    cols_to_drop = [col for col in [score_col, group_col] if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # 데이터가 존재하는 유전자들의 평균 점수 산출 및 새 변수로 할당
    df[score_col] = z_scores.mean(axis=1)
    median_val = df[score_col].median()
    df[group_col] = np.where(df[score_col] >= median_val, 'High', 'Low')

    # 병합된 최신 데이터를 다시 SAV와 CSV로 물리적으로 저장
    pyreadstat.write_sav(df, sav_path)
    csv_path = sav_path.replace('.sav', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return df

# =====================================================================
# 분석 함수들 
# =====================================================================
def analyze_chi_square(df, target_col='SignatureGroup', row_vars=None):
    if target_col not in df.columns:
        print(f"  [경고] '{target_col}' 컬럼이 없어 카이제곱 분석을 건너뜁니다.")
        return pd.DataFrame()
        
    if row_vars is None:
        row_vars = ['ageG', 'gender', 'pathologic_stage', 'pathologic_M', 'pathologic_N', 'pathologic_T', 'SmokingStatus']
    
    target_cats = sorted(df[target_col].dropna().unique())
    if len(target_cats) == 0: 
        return pd.DataFrame()
        
    table1_data = []
    for var in row_vars:
        if var not in df.columns: 
            continue
        crosstab_stat = pd.crosstab(df[var], df[target_col])
        try:
            chi2, p_val, dof, expected = stats.chi2_contingency(crosstab_stat)
            p_val_formatted = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        except ValueError:
            p_val_formatted = "N/A"
            
        table1_data.append([var] + [""] * len(target_cats) + [""])
        for i, (idx, row) in enumerate(crosstab_stat.iterrows()):
            row_data = [f"  {idx}"] + [row.get(cat, 0) for cat in target_cats]
            row_data.append(p_val_formatted if i == 0 else "")
            table1_data.append(row_data)
            
    return pd.DataFrame(table1_data, columns=['Clinical Characteristics'] + target_cats + ['P-value'])

def analyze_bivariate_correlation(df, gene_list):
    valid_genes = [g for g in gene_list if g in df.columns]
    if len(valid_genes) < 2: 
        return pd.DataFrame()
    
    table2_data = []
    for i in valid_genes:
        row_r = [i, 'R']
        row_p = ['', 'P']
        for j in valid_genes:
            if i == j:
                row_r.append("1")
                row_p.append("")
            else:
                pair_data = df[[i, j]].dropna()
                if len(pair_data) >= 2:
                    r_val, p_val = stats.pearsonr(pair_data[i], pair_data[j])
                    row_r.append(f"{r_val:.3f}")
                    row_p.append(f"{p_val:.3f}" if p_val >= 0.001 else "<0.001")
                else:
                    row_r.append("N/A")
                    row_p.append("N/A")
        table2_data.append(row_r)
        table2_data.append(row_p)
        
    return pd.DataFrame(table2_data, columns=['Variable', 'Stat'] + valid_genes)

def analyze_glm_multivariate(df, mode, expr_col='SignatureScore', group_col='SignatureGroup', gene_vars=None):
    print("--- 2-3. 다변량 일반선형모형 (GLM / OLS) - 4개 개별 모델 분석 ---")
    if gene_vars is None:
        gene_vars = []
        
    if mode == "LUAD":
        age_var = "age_at_initial_pathologic_diagnosis"
    elif mode == "LUSC":
        age_var = "age"
        
    base_vars = [age_var, 'gender', 'number_pack_years_smoked'] + gene_vars
    target_vars = [
        ('Pathologic Stage', 'pathologic_stage'),
        ('Pathologic T', 'pathologic_T'),
        ('Pathologic N', 'pathologic_N'),
        ('Pathologic M', 'pathologic_M')
    ]
    
    all_table4_data = []
    
    for model_name, target_var in target_vars:
        # 베이스 변수 중 존재하는 것만 필터링
        valid_base_vars = [v for v in base_vars if v in df.columns]
        
        if target_var not in df.columns:
            print(f"  [경고] '{target_var}' 컬럼이 없어 {model_name} 모델 분석을 건너뜁니다.")
            continue
            
        # 종속변수(Y)와 현재 모델에 필요한 독립변수(X)들만 모아서 결측치 제거
        vars_to_use = [expr_col] + valid_base_vars + [target_var]
        df_clean = df[vars_to_use].dropna()
        
        if df_clean.empty:
            print(f"  [경고] 결측치 제거 후 {model_name} 모델 분석할 데이터가 없습니다.")
            continue
            
        # Formula 작성 (명목형 변수는 C()로 감싸고, gene_vars 같은 연속형 변수는 C() 없이 추가)
        predictors = []
        if age_var in valid_base_vars: predictors.append(age_var)
        if 'gender' in valid_base_vars: predictors.append("C(gender)")
        if 'number_pack_years_smoked' in valid_base_vars: predictors.append("number_pack_years_smoked")
        
        for g in gene_vars:
            if g in valid_base_vars:
                predictors.append(g)  # 연속형 데이터이므로 C() 처리 안함
                
        predictors.append(f"C({target_var})")
        
        formula = f"{expr_col} ~ " + " + ".join(predictors)
        
        try:
            model = smf.ols(formula=formula, data=df_clean).fit()
            print(f"\n[Model: {model_name}]")
            print(model.summary())
            
            # Table 4 데이터 수집
            all_table4_data.append({"Variable": f"=== [ Model: {model_name} ] ===", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            all_table4_data.append({"Variable": "Dependent Variable (Y)", "Coefficient": expr_col, "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            all_table4_data.append({"Variable": "Model Formula", "Coefficient": formula, "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            all_table4_data.append({"Variable": "R-squared", "Coefficient": f"{model.rsquared:.4f}", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            all_table4_data.append({"Variable": "Prob (F-statistic)", "Coefficient": f"{model.f_pvalue:.4e}", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            all_table4_data.append({"Variable": "--- [ Coefficients ] ---", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            
            coef = model.params
            pvalues = model.pvalues
            conf_int = model.conf_int()
            
            for idx in coef.index:
                all_table4_data.append({
                    "Variable": idx,
                    "Coefficient": round(coef[idx], 4),
                    "95% CI Lower": round(conf_int.loc[idx, 0], 4),
                    "95% CI Upper": round(conf_int.loc[idx, 1], 4),
                    "P-value": f"{pvalues[idx]:.3f}" if pvalues[idx] >= 0.001 else "<0.001"
                })
            all_table4_data.append({"Variable": "", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            
        except Exception as e:
            print(f"{model_name} 다변량 분석 중 오류 발생: {e}\n")
            
    print("\n")
    if not all_table4_data:
        return pd.DataFrame()
        
    return pd.DataFrame(all_table4_data)

def analyze_kaplan_meier(df, mode, save_dir, group_col='SignatureGroup', plot_type=1, file_prefix=''):
    if group_col not in df.columns or 'OS.time' not in df.columns or 'OS' not in df.columns:
        print(f"  [경고] 생존 분석에 필요한 컬럼이 없어 KM 분석을 건너뜁니다.")
        return
        
    df_km = df.dropna(subset=[group_col, 'OS.time', 'OS'])
    if df_km.empty: 
        return
        
    # 한글 폰트 및 스타일 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    kmf = KaplanMeierFitter()
    
    # 범례 순서 고정: High가 위(파란색), Low가 아래(빨간색)
    raw_groups = df_km[group_col].unique()
    groups = []
    if 'High' in raw_groups: groups.append('High')
    if 'Low' in raw_groups: groups.append('Low')
    for g in sorted(raw_groups):
        if g not in groups:
            groups.append(g)
            
    colors = ['#00a2e8', '#b51a5e']  # High=파랑, Low=자주색
    color_map = {'High': '#00a2e8', 'Low': '#b51a5e'}
    for i, g in enumerate(groups):
        if g not in color_map:
            color_map[g] = colors[i % len(colors)]
    
    import matplotlib.lines as mlines
    from matplotlib.legend_handler import HandlerLine2D
    
    class StepLine2D(mlines.Line2D):
        pass
        
    class StepHandler(HandlerLine2D):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            xdata = [0, width*0.35, width*0.35, width*0.85, width*0.85]
            ydata = [height*0.2, height*0.2, height*0.8, height*0.8, height*0.2]
            line = mlines.Line2D(xdata, ydata, color=orig_handle.get_color(), lw=1.5)
            line.set_transform(trans)
            return [line]

    # ---------------- OS 분석 ----------------
    plt.figure(figsize=(8, 6)) 
    ax_os = plt.gca()
    
    for group in groups:
        mask = (df_km[group_col] == group)
        if mask.any() and len(df_km.loc[mask, 'OS.time']) > 0:
            color = color_map[group]
            kmf.fit(df_km.loc[mask, 'OS.time'], event_observed=df_km.loc[mask, 'OS'])
            kmf.plot_survival_function(ax=ax_os, ci_show=False, color=color,
                                       show_censors=True, censor_styles={'marker': '+', 'mew': 1, 'ms': 6, 'mec': color})
            
    if plot_type == 1:
        ax_os.set_title('생존함수')
        ax_os.set_xlabel('OS.time')
        ax_os.set_ylabel('누적 생존함수')
    elif plot_type == 2:
        ax_os.set_title('')
        ax_os.set_xlabel('Time(Days)', fontweight='bold', fontsize=12)
        ax_os.set_ylabel('Overall Survival', fontweight='bold', fontsize=12)
        ax_os.spines['top'].set_visible(False)
        ax_os.spines['right'].set_visible(False)

    ax_os.grid(axis='y', color='lightgray', linestyle='-')
    ax_os.set_facecolor('white')
    for spine in ax_os.spines.values():
        if spine.get_visible():
            spine.set_color('black')
            
    handles_os = []
    labels_os = []
    for group in groups:
        handles_os.append(StepLine2D([0], [0], color=color_map[group]))
        labels_os.append(group)
    for group in groups:
        handles_os.append(mlines.Line2D([], [], color=color_map[group], marker='+', linestyle='-', lw=1.5, mew=1, ms=6))
        if plot_type == 1:
            labels_os.append(f'{group}-중도절단')
        elif plot_type == 2:
            labels_os.append(f'{group}-censored')
        
    valid_groups = df_km[group_col].unique()
    if len(valid_groups) > 1:
        res_os = multivariate_logrank_test(df_km['OS.time'], df_km[group_col], df_km['OS'])
        p_val_os = res_os.p_value
        print(f"  [OS Log-rank Test] p-value: {p_val_os:.4f}")
        if plot_type == 1:
            ax_os.legend(handles=handles_os, labels=labels_os, title=f'{group_col}\nLog-rank p={p_val_os:.3f}', handler_map={StepLine2D: StepHandler()})
        elif plot_type == 2:
            legend = ax_os.legend(handles=handles_os, labels=labels_os, title=group_col, handler_map={StepLine2D: StepHandler()}, frameon=False, loc='upper right')
            plt.setp(legend.get_title(), fontweight='bold')
            ax_os.text(0.05, 0.05, f"P = {p_val_os:.3f}", transform=ax_os.transAxes, fontsize=16, fontweight='bold')
    else:
        legend = ax_os.legend(handles=handles_os, labels=labels_os, title=group_col, handler_map={StepLine2D: StepHandler()}, frameon=(plot_type==1))
        if plot_type == 2:
            plt.setp(legend.get_title(), fontweight='bold')
            
    plt.tight_layout()
    km_plot_path_os = os.path.join(save_dir, f'{file_prefix}Signature_{mode}_OS_KM.png')
    if os.path.exists(km_plot_path_os):
        try: os.remove(km_plot_path_os)
        except: pass
    plt.savefig(km_plot_path_os)
    plt.close()

    # ---------------- RFS 분석 ----------------
    if 'RFS.time' in df.columns and 'RFS' in df.columns:
        df_rfs = df_km.dropna(subset=['RFS.time', 'RFS'])
        if not df_rfs.empty:
            plt.figure(figsize=(8, 6)) 
            ax_rfs = plt.gca()
            
            for group in groups:
                mask = (df_rfs[group_col] == group)
                if mask.any():
                    color = color_map[group]
                    kmf.fit(df_rfs.loc[mask, 'RFS.time'], event_observed=df_rfs.loc[mask, 'RFS'])
                    kmf.plot_survival_function(ax=ax_rfs, ci_show=False, color=color,
                                               show_censors=True, censor_styles={'marker': '+', 'mew': 1, 'ms': 6, 'mec': color})
                    
            if plot_type == 1:
                ax_rfs.set_title('생존함수')
                ax_rfs.set_xlabel('RFS.time')
                ax_rfs.set_ylabel('누적 생존함수')
            elif plot_type == 2:
                ax_rfs.set_title('')
                ax_rfs.set_xlabel('Time(Days)', fontweight='bold', fontsize=12)
                ax_rfs.set_ylabel('Relapse-Free Survival', fontweight='bold', fontsize=12)
                ax_rfs.spines['top'].set_visible(False)
                ax_rfs.spines['right'].set_visible(False)

            ax_rfs.grid(axis='y', color='lightgray', linestyle='-')
            ax_rfs.set_facecolor('white')
            for spine in ax_rfs.spines.values():
                if spine.get_visible():
                    spine.set_color('black')
                
            handles_rfs = []
            labels_rfs = []
            for group in groups:
                handles_rfs.append(StepLine2D([0], [0], color=color_map[group]))
                labels_rfs.append(group)
            for group in groups:
                handles_rfs.append(mlines.Line2D([], [], color=color_map[group], marker='+', linestyle='-', lw=1.5, mew=1, ms=6))
                if plot_type == 1:
                    labels_rfs.append(f'{group}-중도절단')
                elif plot_type == 2:
                    labels_rfs.append(f'{group}-censored')
                
            rfs_groups = df_rfs[group_col].unique()
            if len(rfs_groups) > 1:
                res_rfs = multivariate_logrank_test(df_rfs['RFS.time'], df_rfs[group_col], df_rfs['RFS'])
                p_val_rfs = res_rfs.p_value
                print(f"  [RFS Log-rank Test] p-value: {p_val_rfs:.4f}")
                if plot_type == 1:
                    ax_rfs.legend(handles=handles_rfs, labels=labels_rfs, title=f'{group_col}\nLog-rank p={p_val_rfs:.3f}', handler_map={StepLine2D: StepHandler()})
                elif plot_type == 2:
                    legend = ax_rfs.legend(handles=handles_rfs, labels=labels_rfs, title=group_col, handler_map={StepLine2D: StepHandler()}, frameon=False, loc='upper right')
                    plt.setp(legend.get_title(), fontweight='bold')
                    ax_rfs.text(0.05, 0.05, f"P = {p_val_rfs:.3f}", transform=ax_rfs.transAxes, fontsize=16, fontweight='bold')
            else:
                legend = ax_rfs.legend(handles=handles_rfs, labels=labels_rfs, title=group_col, handler_map={StepLine2D: StepHandler()}, frameon=(plot_type==1))
                if plot_type == 2:
                    plt.setp(legend.get_title(), fontweight='bold')
                
            plt.tight_layout()
            km_plot_path_rfs = os.path.join(save_dir, f'{file_prefix}Signature_{mode}_RFS_KM.png')
            if os.path.exists(km_plot_path_rfs):
                try: os.remove(km_plot_path_rfs)
                except: pass
            plt.savefig(km_plot_path_rfs)
            plt.close()

def analyze_cox_regression(df, mode, save_dir, group_col='SignatureGroup', file_prefix='', categorical_vars=None, continuous_vars=None):
    # 1. 분석에 사용할 변수들을 명목형(Categorical)과 연속형(Continuous)으로 명확히 구분
    if categorical_vars is None:
        categorical_vars = [group_col, 'gender', 'pathologic_stage']
    if continuous_vars is None:
        age_col = 'age_at_initial_pathologic_diagnosis' if mode == "LUAD" else 'age'
        continuous_vars = [age_col, 'number_pack_years_smoked']
        
    survival_cols = ['OS.time', 'OS']
    cols = survival_cols + categorical_vars + continuous_vars
    
    valid_cols = [c for c in cols if c in df.columns]
    if group_col not in valid_cols or 'OS' not in valid_cols:
        print(f"  [경고] 필수 컬럼이 없어 Cox 분석을 건너뜁니다.")
        return pd.DataFrame()
        
    df_cox = df[valid_cols].dropna()
    if df_cox.empty: 
        return pd.DataFrame()
    
    # 2. 명목형 변수들만 명시적으로 선택하여 더미 변환 (연속형은 원형 그대로 유지)
    dummy_cols = [c for c in categorical_vars if c in df_cox.columns]
    df_cox_dummy = pd.get_dummies(df_cox, columns=dummy_cols, drop_first=True)
    
    cph = CoxPHFitter()
    
    df_table3 = pd.DataFrame()
    try:
        cph.fit(df_cox_dummy, duration_col='OS.time', event_col='OS')
        # 콘솔 출력에도 95% CI가 보이도록 추가
        print(cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])
        
        summary = cph.summary
        # 엑셀 표(Table 3)에 95% CI 하한선과 상한선을 논문 포맷으로 조립
        df_table3 = pd.DataFrame({
            'Clinical Variable': summary.index,
            'Hazard Ratio (HR)': summary['exp(coef)'].round(3),
            '95% CI Lower': summary['exp(coef) lower 95%'].round(3),
            '95% CI Upper': summary['exp(coef) upper 95%'].round(3),
            'P-value': summary['p'].apply(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
        })
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Cox Regression - Forest Plot - Signature ({mode})') 
        plt.tight_layout()
        
        cox_plot_path = os.path.join(save_dir, f'{file_prefix}Signature_{mode}_Cox.png')
        if os.path.exists(cox_plot_path):
            try: os.remove(cox_plot_path)
            except: pass
        plt.savefig(cox_plot_path)
        plt.close()
        return df_table3
    except: 
        return pd.DataFrame()

# =====================================================================
# 3. 메인 실행 블록 (LUAD, LUSC 순차적 자동 실행)
# =====================================================================
if __name__ == "__main__":
    super_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 분석할 시그니처 유전자 리스트 업데이트 (원하시는 EMT-ECM 축 유전자로 변경 가능)
    sig_genes = ["LOXL2", "ITGB1", "PLAUR", "SNAI1", "VEGFC"]
    
    # 여기서 플롯 출력 타입을 지정합니다. 
    # 1: 기존 한글 스타일(범례에 P-value 포함) / 2: 영문 스타일(그래프 내부에 P-value 삽입)
    CURRENT_PLOT_TYPE = 2
    CUSTOM_FILE_PREFIX = ""
    
    for mode in ['LUAD', 'LUSC']:
        print(f"\n{'='*60}")
        print(f"          {mode} Signature 자동화 분석 시작")
        print(f"{'='*60}")
        
        master_sav_path = os.path.join(super_dir, "database", '(LUAD) lung adenocarcinoma.sav' if mode == 'LUAD' else '(LUSC) lung squamous cell carcinoma.sav')
        
        if not os.path.exists(master_sav_path):
            print(f"오류: 마스터 임상 파일이 존재하지 않습니다: {master_sav_path}")
            continue
            
        result_dir = os.path.join(super_dir, "result")
        plot_base_dir = os.path.join(super_dir, "plot")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(plot_base_dir, exist_ok=True)

        if isinstance(sys.stdout, PrintLogger):
            sys.stdout.flush()
        sys.stdout = PrintLogger(os.path.join(result_dir, f"{CUSTOM_FILE_PREFIX}result_signature_{mode}.txt"))
        
        # 1. 마스터 데이터 로드 및 시그니처 생성
        merged_df, _ = pyreadstat.read_sav(master_sav_path)
        merged_df = get_signature_group(merged_df, sig_genes, master_sav_path)
        
        gene_name_list = '_'.join(sig_genes)
        group_col_name = f'SignatureGroup_{gene_name_list}'
        score_col_name = f'SignatureScore_{gene_name_list}'
        
        if group_col_name not in merged_df.columns:
            print(f"[{mode}] 시그니처 그룹 생성에 실패하여 다음 암종으로 넘어갑니다.")
            continue
            
        # 2. 통계 및 그래프 생성
        df_table1 = analyze_chi_square(merged_df, target_col=group_col_name)
        
        # [수정됨] 암종별로 통제(교란) 변수로 사용할 연속형 유전자 발현량(gene_vars) 분리
        if mode == "LUAD":
            gene_vars = ['KrasExpression', 'TP53Expression', 'ALKExpression', 'BRAFExpression']
            corr_vars = [score_col_name, 'number_pack_years_smoked', 'age_at_initial_pathologic_diagnosis'] + gene_vars
        else:
            gene_vars = ['TP53Expression', 'CDKN2AExpression', 'SOX2Expression', 'PIK3CAExpression', 'NOTCH1Expression']
            corr_vars = [score_col_name, 'number_pack_years_smoked', 'age'] + gene_vars
            
        df_table2 = analyze_bivariate_correlation(merged_df, gene_list=corr_vars)
        
        # 다변량 분석에 gene_vars를 함께 전달하여 연속형 변수로 회귀분석 수행
        df_table4 = analyze_glm_multivariate(merged_df, mode, expr_col=score_col_name, group_col=group_col_name, gene_vars=gene_vars)
        
        analyze_kaplan_meier(merged_df, mode, plot_base_dir, group_col=group_col_name, plot_type=CURRENT_PLOT_TYPE, file_prefix=CUSTOM_FILE_PREFIX)
        
        # [수정됨] Cox 생존 분석 시 명목형/연속형 변수를 명확히 분리하여 주입
        age_col = 'age_at_initial_pathologic_diagnosis' if mode == "LUAD" else 'age'
        
        cox_categorical = [group_col_name, 'gender', 'pathologic_stage']
        cox_continuous = [age_col, 'number_pack_years_smoked'] + gene_vars
        
        df_table3 = analyze_cox_regression(
            merged_df, mode, plot_base_dir, 
            group_col=group_col_name, 
            file_prefix=CUSTOM_FILE_PREFIX, 
            categorical_vars=cox_categorical, 
            continuous_vars=cox_continuous
        )
        
        # 3. 엑셀 저장
        excel_save_path = os.path.join(result_dir, f"{CUSTOM_FILE_PREFIX}Signature_{mode}_Tables.xlsx")
        if os.path.exists(excel_save_path):
            try: os.remove(excel_save_path)
            except: pass
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            if not df_table1.empty:
                df_table1.to_excel(writer, sheet_name='Table 1', index=False)
            if not df_table2.empty:
                df_table2.to_excel(writer, sheet_name='Table 2', index=False)
            if not df_table3.empty:
                df_table3.to_excel(writer, sheet_name='Table 3', index=False)
            if not df_table4.empty:
                df_table4.to_excel(writer, sheet_name='Table 4', index=False)
                
        print(f"✅ {mode} Signature 기반 분석 및 엑셀 저장 완료")