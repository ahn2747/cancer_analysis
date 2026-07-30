import pandas as pd
import numpy as np
import pyreadstat
import scipy.stats as stats
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
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
    
    if not z_scores.empty:
        df['SignatureScore'] = z_scores.mean(axis=1)
        median_score = df['SignatureScore'].median()
        df['SignatureGroup'] = np.where(df['SignatureScore'] > median_score, 'High', 'Low')
        df.loc[df['SignatureScore'].isna(), 'SignatureGroup'] = np.nan
        
        try:
            pyreadstat.write_sav(df, sav_path)
            print(f"  [진단 로그] 마스터 SAV 파일에 시그니처 점수/그룹 변수를 성공적으로 저장했습니다.")
        except Exception as e:
            print(f"  [경고] SAV 파일 업데이트 실패: {e}")
            
    return df

# =====================================================================
# 2. 기초 및 다변량 분석 파이프라인
# =====================================================================
def analyze_chi_square(df, target_col='SignatureGroup', row_vars=None):
    print("--- 2-1. 임상 특성 예후 분석 (Chi-square Test) ---")
    if row_vars is None:
        row_vars = ['ageG', 'gender', 'pathologic_stage', 'pathologic_M', 'pathologic_N', 'pathologic_T', 'SmokingStatus']
    
    target_cats = sorted(df[target_col].dropna().unique())
    if len(target_cats) == 0:
        target_cats = ['Group1', 'Group2']
        
    table1_data = []
    na_strings = ['not reported', '[not available]', 'unknown', 'na', 'n/a', '', 'none']
    
    for var in row_vars:
        if var not in df.columns:
            continue
        
        df_clean = df[[var, target_col]].copy()
        df_clean[var] = df_clean[var].apply(lambda x: np.nan if pd.isna(x) or str(x).strip().lower() in na_strings else x)
        df_clean = df_clean.dropna()
        
        if df_clean[var].nunique() <= 1 or df_clean[target_col].nunique() <= 1:
            continue

        crosstab_stat = pd.crosstab(df_clean[var], df_clean[target_col])
        
        try:
            chi2, p_val, dof, expected = stats.chi2_contingency(crosstab_stat, correction=False)
            min_expected = expected.min()
            warning_msg = f" (⚠️경고: 최소 기대빈도 {min_expected:.1f}<5)" if min_expected < 5 else ""
            p_val_formatted = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        except ValueError:
            p_val_formatted = "N/A"
            warning_msg = ""
            
        crosstab_print = pd.crosstab(df_clean[var], df_clean[target_col], margins=True, margins_name="전체")
        print(f"[{target_col} vs {var}]")
        print(crosstab_print)
        print(f"Chi-square: {chi2:.4f}, p-value: {p_val_formatted}{warning_msg}\n")
        
        table1_data.append([var] + [""] * len(target_cats) + [""])
        
        for i, (idx, row) in enumerate(crosstab_stat.iterrows()):
            row_data = [f"  {idx}"]
            for cat in target_cats:
                row_data.append(row.get(cat, 0))
            row_data.append(p_val_formatted if i == 0 else "")
            table1_data.append(row_data)

    col_names = ['Clinical Characteristics'] + target_cats + ['P-value']
    df_table1 = pd.DataFrame(table1_data, columns=col_names)
    return df_table1

def analyze_bivariate_correlation(df, gene_list):
    print("--- 2-2. 이변량 상관 분석 (Pearson 행렬) ---")
    valid_genes = [g for g in gene_list if g in df.columns]
    if len(valid_genes) < 2:
        return pd.DataFrame()
        
    df_numeric = df[valid_genes].apply(pd.to_numeric, errors='coerce')
    pearson_matrix = df_numeric.corr(method='pearson')
    print("[Pearson 상관계수 행렬 (SPSS Pairwise 방식)]")
    print(pearson_matrix.round(3))
    print("\n")
    
    table2_data = []
    for i in valid_genes:
        row_r = [i, 'R']
        row_p = ['', 'P-value']
        for j in valid_genes:
            if i == j:
                row_r.append("1")
                row_p.append("")
            else:
                pair_data = df_numeric[[i, j]].dropna()
                if len(pair_data) > 1 and pair_data[i].nunique() > 1 and pair_data[j].nunique() > 1:
                    r_val, p_val = stats.pearsonr(pair_data[i], pair_data[j])
                    row_r.append(f"{r_val:.3f}")
                    row_p.append(f"{p_val:.3f}" if p_val >= 0.001 else "<0.001")
                else:
                    row_r.append("N/A")
                    row_p.append("N/A")
        table2_data.append(row_r)
        table2_data.append(row_p)
        
    col_names = ['Variable', 'Stat'] + valid_genes
    df_table2 = pd.DataFrame(table2_data, columns=col_names)
    return df_table2

# =====================================================================
# [수술 완료] 시그니처 OLM 모델 (Base Model 선행 및 Stage 스위치)
# =====================================================================
def analyze_glm_multivariate(df, mode, expr_col='SignatureScore', categorical_vars=None, continuous_vars=None, target_stages=None):
    print("--- 2-3. 다변량 일반선형모형 (GLM / OLS) - Base 모델 및 병기 결합 ---")
    
    age_var = "age_at_initial_pathologic_diagnosis" if mode == "LUAD" else "age"
    
    if categorical_vars is None: categorical_vars = ['gender']
    if continuous_vars is None: continuous_vars = [age_var, 'number_pack_years_smoked']
    
    if target_stages is False:
        target_stages = []
    elif target_stages is None:
        target_stages = [
            ('Pathologic Stage', 'pathologic_stage'),
            ('Pathologic T', 'pathologic_T'),
            ('Pathologic N', 'pathologic_N'),
            ('Pathologic M', 'pathologic_M')
        ]
    
    all_table4_data = []
    
    # -------------------------------------------------------------
    # 1. Base Model 
    # -------------------------------------------------------------
    base_cols_to_check = [expr_col] + categorical_vars + continuous_vars
    base_valid_cols = [c for c in base_cols_to_check if c in df.columns]
    
    df_base = df[base_valid_cols].dropna()
    
    base_predictors = []
    for var in continuous_vars:
        if var in base_valid_cols: base_predictors.append(var)
    for var in categorical_vars:
        if var in base_valid_cols: base_predictors.append(f"C({var})")
            
    if not df_base.empty and base_predictors:
        formula_base = f"{expr_col} ~ " + " + ".join(base_predictors)
        try:
            model_base = smf.ols(formula=formula_base, data=df_base).fit()
            print(f"\n[Model: Base Model (No Stage)]")
            print(model_base.summary())
            
            all_table4_data.extend([
                {"Variable": f"=== [ Model: Base Model (No Stage) ] ===", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                {"Variable": "Dependent Variable (Y)", "Coefficient": expr_col, "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                {"Variable": "Model Formula", "Coefficient": formula_base, "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                {"Variable": "R-squared", "Coefficient": f"{model_base.rsquared:.4f}", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                {"Variable": "Prob (F-statistic)", "Coefficient": f"{model_base.f_pvalue:.4e}", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                {"Variable": "--- [ Coefficients ] ---", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""}
            ])
            
            coef, pvalues, conf_int = model_base.params, model_base.pvalues, model_base.conf_int()
            for idx in coef.index:
                all_table4_data.append({
                    "Variable": idx, "Coefficient": round(coef[idx], 4),
                    "95% CI Lower": round(conf_int.loc[idx, 0], 4), "95% CI Upper": round(conf_int.loc[idx, 1], 4),
                    "P-value": f"{pvalues[idx]:.3f}" if pvalues[idx] >= 0.001 else "<0.001"
                })
            all_table4_data.append({"Variable": "", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
        except Exception as e:
            print(f"기본 모델 분석 중 오류 발생: {e}\n")

    # -------------------------------------------------------------
    # 2. Stage Models 
    # -------------------------------------------------------------
    if target_stages:
        for model_name, target_var in target_stages:
            cols_to_check = base_cols_to_check + [target_var]
            valid_cols = [c for c in cols_to_check if c in df.columns]
            
            if target_var not in valid_cols: continue
                
            df_stage = df[valid_cols].dropna()
            if df_stage.empty: continue
                
            stage_predictors = base_predictors.copy()
            stage_predictors.append(f"C({target_var})")
            
            formula_stage = f"{expr_col} ~ " + " + ".join(stage_predictors)
            
            try:
                model_stage = smf.ols(formula=formula_stage, data=df_stage).fit()
                print(f"\n[Model: Base + {model_name}]")
                print(model_stage.summary())
                
                all_table4_data.extend([
                    {"Variable": f"=== [ Model: Base + {model_name} ] ===", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                    {"Variable": "Dependent Variable (Y)", "Coefficient": expr_col, "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                    {"Variable": "Model Formula", "Coefficient": formula_stage, "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                    {"Variable": "R-squared", "Coefficient": f"{model_stage.rsquared:.4f}", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                    {"Variable": "Prob (F-statistic)", "Coefficient": f"{model_stage.f_pvalue:.4e}", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""},
                    {"Variable": "--- [ Coefficients ] ---", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""}
                ])
                
                coef, pvalues, conf_int = model_stage.params, model_stage.pvalues, model_stage.conf_int()
                for idx in coef.index:
                    all_table4_data.append({
                        "Variable": idx, "Coefficient": round(coef[idx], 4),
                        "95% CI Lower": round(conf_int.loc[idx, 0], 4), "95% CI Upper": round(conf_int.loc[idx, 1], 4),
                        "P-value": f"{pvalues[idx]:.3f}" if pvalues[idx] >= 0.001 else "<0.001"
                    })
                all_table4_data.append({"Variable": "", "Coefficient": "", "95% CI Lower": "", "95% CI Upper": "", "P-value": ""})
            except Exception as e:
                print(f"{model_name} 다변량 분석 중 오류 발생: {e}\n")
                
    return pd.DataFrame(all_table4_data)


def analyze_kaplan_meier(df, mode, save_dir, group_col='SignatureGroup', file_prefix="", sub_group_col=None):
    print(f"--- 2-4. Kaplan-Meier 생존분석 (Sub-group: {sub_group_col if sub_group_col else 'None'}) ---")
    if group_col not in df.columns:
        return pd.DataFrame()

    df_km = df.copy()
    df_km = df_km[df_km[group_col].isin(['High', 'Low'])]
    
    # ==============================================================
    # [수석 데이터 과학자 설계] 4분할 다중 변수 조합 로직
    # ==============================================================
    plot_group_col = group_col
    if sub_group_col and sub_group_col in df_km.columns:
        df_km[sub_group_col] = df_km[sub_group_col].astype(str).str.strip()
        na_strings = ['not reported', '[not available]', 'unknown', 'na', 'n/a', '', 'none', 'nan']
        df_km = df_km[~df_km[sub_group_col].str.lower().isin(na_strings)]
        
        df_km['CombinedGroup'] = df_km[sub_group_col] + " / " + df_km[group_col]
        plot_group_col = 'CombinedGroup'

    # [핵심 수술] OS 결측치 제거 전에 RFS를 위한 베이스 데이터를 복제해 둡니다.
    df_base_for_rfs = df_km.copy()

    df_km = df_km.dropna(subset=['OS.time', 'OS', plot_group_col])
    
    if df_km.empty:
        return pd.DataFrame()
        
    table5_data = []
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    kmf = KaplanMeierFitter()
    groups = sorted(df_km[plot_group_col].unique())
            
    # 그룹 수에 따른 동적 팔레트 할당 (2분할 vs 4분할 이상)
    if set(groups) == {'High', 'Low'}:
        color_map = {'High': '#00a2e8', 'Low': '#b51a5e'}
    else:
        palette = ['#b51a5e', '#00a2e8', '#22b14c', '#ff7f27', '#9932cc', '#8b4513']
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    
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
        mask = (df_km[plot_group_col] == group)
        time = df_km.loc[mask, 'OS.time'].dropna()
        event = df_km.loc[mask, 'OS'].dropna()
        if len(time) > 0:
            color = color_map[group]
            kmf.fit(time, event_observed=event)
            kmf.plot_survival_function(ax=ax_os, ci_show=False, color=color,
                                    show_censors=True, censor_styles={'marker': '+', 'mew': 1, 'ms': 6, 'mec': color})
            
    ax_os.set_title(f'{mode}_OS')
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
        labels_os.append(f'{group}-censored')
            
    if len(groups) > 1:
        if len(groups) == 2:
            mask0, mask1 = (df_km[plot_group_col] == groups[0]), (df_km[plot_group_col] == groups[1])
            res_os = logrank_test(df_km.loc[mask0, 'OS.time'], df_km.loc[mask1, 'OS.time'], event_observed_A=df_km.loc[mask0, 'OS'], event_observed_B=df_km.loc[mask1, 'OS'])
        else:
            res_os = multivariate_logrank_test(df_km['OS.time'], df_km[plot_group_col], df_km['OS'])
            
        p_val_os = res_os.p_value
        chi2_os = res_os.test_statistic
        p_str_os = f"{p_val_os:.3f}" if p_val_os >= 0.001 else "<0.001"
        
        table5_data.append(["Overall Survival (OS)", mode, "Signature", f"{chi2_os:.3f}", p_str_os])
        print(f"  [OS Log-rank Test] Chi-square: {chi2_os:.3f}, p-value: {p_str_os}")
        
        legend = ax_os.legend(handles=handles_os, labels=labels_os, title=plot_group_col, handler_map={StepLine2D: StepHandler()}, frameon=False, loc='upper right')
        plt.setp(legend.get_title(), fontweight='bold')
        ax_os.text(0.05, 0.05, f"P = {p_str_os}", transform=ax_os.transAxes, fontsize=16, fontweight='bold')
    else:
        legend = ax_os.legend(handles=handles_os, labels=labels_os, title=plot_group_col, handler_map={StepLine2D: StepHandler()}, frameon=False)
        plt.setp(legend.get_title(), fontweight='bold')
            
    plt.tight_layout()
    km_plot_path_os = os.path.join(save_dir, f'{file_prefix}Signature_{mode}_OS_KM.png')
    plt.savefig(km_plot_path_os)
    plt.close()
        
    # ---------------- RFS 분석 ----------------
    if 'RFS.time' in df.columns and 'RFS' in df.columns:
        # [핵심 수술] 원본 df가 아닌, CombinedGroup이 포함된 베이스 데이터를 사용합니다.
        df_rfs = df_base_for_rfs.copy()
        df_rfs = df_rfs.dropna(subset=['RFS.time', 'RFS', plot_group_col])
        
        if not df_rfs.empty:
            plt.figure(figsize=(8, 6)) 
            ax_rfs = plt.gca()
            
            for group in groups:
                mask = (df_rfs[plot_group_col] == group)
                time = df_rfs.loc[mask, 'RFS.time'].dropna()
                event = df_rfs.loc[mask, 'RFS'].dropna()
                if len(time) > 0:
                    color = color_map[group]
                    kmf.fit(time, event_observed=event)
                    kmf.plot_survival_function(ax=ax_rfs, ci_show=False, color=color,
                                            show_censors=True, censor_styles={'marker': '+', 'mew': 1, 'ms': 6, 'mec': color})
                    
            ax_rfs.set_title(f'{mode}_RFS')
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
                labels_rfs.append(f'{group}-censored')
                    
            if len(groups) > 1:
                if len(groups) == 2:
                    mask0, mask1 = (df_rfs[plot_group_col] == groups[0]), (df_rfs[plot_group_col] == groups[1])
                    res_rfs = logrank_test(df_rfs.loc[mask0, 'RFS.time'], df_rfs.loc[mask1, 'RFS.time'], event_observed_A=df_rfs.loc[mask0, 'RFS'], event_observed_B=df_rfs.loc[mask1, 'RFS'])
                else:
                    res_rfs = multivariate_logrank_test(df_rfs['RFS.time'], df_rfs[plot_group_col], df_rfs['RFS'])
                    
                p_val_rfs = res_rfs.p_value
                chi2_rfs = res_rfs.test_statistic
                p_str_rfs = f"{p_val_rfs:.3f}" if p_val_rfs >= 0.001 else "<0.001"
                
                table5_data.append(["Relapse-Free Survival (RFS)", mode, "Signature", f"{chi2_rfs:.3f}", p_str_rfs])
                
                legend = ax_rfs.legend(handles=handles_rfs, labels=labels_rfs, title=plot_group_col, handler_map={StepLine2D: StepHandler()}, frameon=False, loc='upper right')
                plt.setp(legend.get_title(), fontweight='bold')
                ax_rfs.text(0.05, 0.05, f"P = {p_str_rfs}", transform=ax_rfs.transAxes, fontsize=16, fontweight='bold')
            else:
                legend = ax_rfs.legend(handles=handles_rfs, labels=labels_rfs, title=plot_group_col, handler_map={StepLine2D: StepHandler()}, frameon=False)
                plt.setp(legend.get_title(), fontweight='bold')
                    
            plt.tight_layout()
            km_plot_path_rfs = os.path.join(save_dir, f'{file_prefix}Signature_{mode}_RFS_KM.png')
            plt.savefig(km_plot_path_rfs)
            plt.close()

    return pd.DataFrame(table5_data, columns=["Survival Type", "Cancer Type", "Target", "Chi-square", "P-value"])

def analyze_cox_regression(df, mode, save_dir, group_col='SignatureGroup', file_prefix="", categorical_vars=None, continuous_vars=None):
    print("--- 2-5. Cox 회귀분석 ---")
    age_col = 'age_at_initial_pathologic_diagnosis' if mode == "LUAD" else 'age'
    
    if categorical_vars is None:
        categorical_vars = [group_col, 'gender', 'pathologic_stage']
    if continuous_vars is None:
        continuous_vars = [age_col, 'number_pack_years_smoked']
        
    cols = ['OS.time', 'OS'] + categorical_vars + continuous_vars
    
    valid_cols = [c for c in cols if c in df.columns]
    if group_col not in valid_cols or 'OS' not in valid_cols:
        print("  [경고] 필수 컬럼이 없어 Cox 분석을 건너뜁니다.")
        return pd.DataFrame()
        
    # 바로 결측치를 제거하지 않고, 사본을 떠서 정제 작업을 먼저 수행
    df_cox = df[valid_cols].copy()
    
    # [핵심 수술] Pandas가 결측치로 인식하지 못하는 가짜 공백 및 문자열 찌꺼기 완벽 정제
    na_strings = ['not reported', '[not available]', 'unknown', 'na', 'n/a', '', 'none']
    for col in valid_cols:
        # 문자열(object) 또는 범주형 데이터일 경우에만 텍스트 검사
        if df_cox[col].dtype == object or isinstance(df_cox[col].dtype, pd.CategoricalDtype):
            # 1. 선생님의 아이디어: 정규식으로 완전 공백(Space) 문자열을 NaN으로 강제 치환
            df_cox[col] = df_cox[col].replace(r'^\s*$', np.nan, regex=True)
            # 2. 지정된 찌꺼기 텍스트를 NaN으로 치환 (대소문자 무시)
            df_cox[col] = df_cox[col].apply(lambda x: np.nan if pd.isna(x) or str(x).strip().lower() in na_strings else x)
    
    # 찌꺼기가 완벽하게 NaN으로 치환된 후, 순수 데이터만 남기기
    df_cox = df_cox.dropna()
    
    if df_cox.empty:
        print("  [경고] 결측치 제거 후 Cox 분석할 데이터가 없습니다.")
        return pd.DataFrame()
        
    dummy_cols = [c for c in categorical_vars if c in df_cox.columns]
    df_cox_dummy = pd.get_dummies(df_cox, columns=dummy_cols, drop_first=True)
    
    cph = CoxPHFitter()
    
    df_table3 = pd.DataFrame() 
    
    try:
        cph.fit(df_cox_dummy, duration_col='OS.time', event_col='OS')
        
        print(cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]) 
        
        summary = cph.summary
        
        df_table3 = pd.DataFrame({
            'Clinical Variable': summary.index,
            'Hazard Ratio (HR)': summary['exp(coef)'].round(3),
            '95% CI Lower': summary['exp(coef) lower 95%'].round(3),
            '95% CI Upper': summary['exp(coef) upper 95%'].round(3),
            'P-value': summary['p'].apply(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
        })
        
        plt.figure(figsize=(10, 6))
        cph.plot()
        plt.title(f'Cox Regression - Forest Plot - Signature({mode})')
        plt.tight_layout()
        cox_plot_path = os.path.join(save_dir, f'{file_prefix}Signature_{mode}_Cox_Forest_Plot.png')
        plt.savefig(cox_plot_path)
        plt.close()
        print(f"Cox Forest Plot 이미지 저장 완료: '{cox_plot_path}'\n")
        
    except Exception as e:
         print(f"Cox 분석 중 오류 발생: {e}")
         
    return df_table3

# =====================================================================
# 3. 메인 실행 블록
# =====================================================================
if __name__ == "__main__":
    
    try:
        super_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        super_dir = os.path.abspath(".")

    database_dir = os.path.join(super_dir, "database")
    result_dir = os.path.join(super_dir, "result_signature")
    plot_base_dir = os.path.join(super_dir, "plot_signature")

    for d in [database_dir, result_dir, plot_base_dir]:
        os.makedirs(d, exist_ok=True)

    CUSTOM_FILE_PREFIX = "Signature_2차_"

    gene_list = ["LOXL2", "ITGB1", "PLAUR", "SNAI1", "VEGFC"]
    group_col_name = "SignatureGroup"
    expr_col_name = "SignatureScore"

    for mode in ['LUAD', 'LUSC']:
        print(f"\n{'='*60}")
        print(f"          {mode} Signature 자동화 분석 시작")
        print(f"{'='*60}")

        if mode == 'LUAD':
            master_sav_filename = '(LUAD) lung adenocarcinoma.sav'
        else:
            master_sav_filename = '(LUSC) lung squamous cell carcinoma.sav'
            
        master_sav_path = os.path.join(database_dir, master_sav_filename)

        if not os.path.exists(master_sav_path):
            print(f"오류: 마스터 임상 파일이 존재하지 않습니다: {master_sav_path}")
            continue

        result_txt_path = os.path.join(result_dir, f"{CUSTOM_FILE_PREFIX}result_Signature_{mode}.txt")
        sys.stdout = PrintLogger(result_txt_path)
        
        print(f"--- 1. 기존 데이터 로드 및 Signature 계산 ---")
        df_clin, meta = pyreadstat.read_sav(master_sav_path, user_missing=True)
        merged_df = get_signature_group(df_clin, gene_list, master_sav_path)
        
        if group_col_name not in merged_df.columns:
            print(f"  [경고] Signature 계산에 실패했습니다. 이 암종은 건너뜁니다.")
            print("="*60)
            continue
            
        # 2. 분석 및 표 데이터 생성
        df_table1 = analyze_chi_square(merged_df, target_col=group_col_name)
        
        if mode == "LUAD":
            gene_vars = ['KrasExpression', 'TP53Expression', 'ALKExpression', 'BRAFExpression', 'EGFRExpression']
            corr_continuous = [expr_col_name, 'number_pack_years_smoked', 'age_at_initial_pathologic_diagnosis'] + gene_vars
            corr_binary_targets = ['EGFR']
        else:
            gene_vars = ['TP53Expression', 'CDKN2AExpression', 'SOX2Expression', 'PIK3CAExpression', 'NOTCH1Expression']
            corr_continuous = [expr_col_name, 'number_pack_years_smoked', 'age'] + gene_vars
            corr_binary_targets = []
            
        corr_processed_binary = []
        
        def encode_mutation(val):
            if pd.isna(val):
                return np.nan
            v_str = str(val).strip().lower()
            if v_str in ['none', 'wt', 'wildtype']: return 0
            elif v_str in ['not reported', '[not available]', 'unknown', 'na', 'n/a', '']: return np.nan
            else: return 1

        for mut_var in corr_binary_targets:
            if mut_var in merged_df.columns:
                bin_col_name = f"{mut_var}_Mut_Binary"
                merged_df[bin_col_name] = merged_df[mut_var].apply(encode_mutation)
                corr_processed_binary.append(bin_col_name)
                print(f"  [안내] {mut_var} 데이터를 이진수(0=WT, 1=Mut, 제외=NaN)로 정밀 변환 완료.")
        
        final_corr_genes = corr_continuous #+ corr_processed_binary
        
        df_table2 = analyze_bivariate_correlation(merged_df, gene_list=final_corr_genes)
        
        # ==============================================================
        # [블록화 적용] OLM 변수 분리 및 Stage 스위치 (True/False/None)
        # ==============================================================
        age_col = 'age_at_initial_pathologic_diagnosis' if mode == "LUAD" else 'age'
        glm_categorical = ['gender']
        glm_continuous = [age_col, 'number_pack_years_smoked']
        
        df_table4 = analyze_glm_multivariate(
            merged_df, mode, 
            expr_col=expr_col_name, 
            categorical_vars=glm_categorical, 
            continuous_vars=glm_continuous,
            target_stages=None  # 병기 분석을 끄고 Base 모델만 보려면 target_stages=False 옵션을 부여하세요.
        )
        
        # [플롯 스위치] 여기에 'ageG', 'gender', 'pathologic_stage' 등을 입력하면 자동으로 다중 플롯이 생성됩니다.
        # 원상복구(2분할)를 원하시면 KM_SUB_GROUP = None 으로 두시면 됩니다.
        KM_SUB_GROUP = 'ageG' 
        
        df_table5 = analyze_kaplan_meier(
            merged_df, mode, plot_base_dir, 
            group_col=group_col_name, 
            file_prefix=CUSTOM_FILE_PREFIX,
            sub_group_col=KM_SUB_GROUP
        )
        
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
                df_table1.to_excel(writer, sheet_name='Table 1 (Chi-square)', index=False)
            if not df_table2.empty:
                df_table2.to_excel(writer, sheet_name='Table 2 (Correlation)', index=False)
            if not df_table3.empty:
                df_table3.to_excel(writer, sheet_name='Table 3 (Cox)', index=False)
            if not df_table4.empty:
                df_table4.to_excel(writer, sheet_name='Table 4 (Multivariate OLS)', index=False)
            if not df_table5.empty:
                df_table5.to_excel(writer, sheet_name='Table 5 (KM Log-rank)', index=False)
        print(f"✅ 논문용 시그니처 표 엑셀 파일 저장 완료: '{excel_save_path}'")
        
        print("="*60)
        
    print(f"\n=== 모든 Signature 생물통계 파이프라인이 성공적으로 완료되었습니다 ===")