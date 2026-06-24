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
# 1. 데이터 처리 및 병합 (원본 로직 복구: 중복 컬럼 방지 및 SAV/CSV 저장)
# =====================================================================
def load_and_merge_data(target, onco_file, clin_file):
    print(f"--- 1. 데이터 처리 및 병합 시작 (Target: {target}) ---")
    
    print("  [진단 로그] 1-1. OncoLnc CSV 파일 로드 시도 중...")
    df_onco = pd.read_csv(onco_file, usecols=['Patient', 'Expression', 'Group'])
    df_onco.rename(columns={
        'Patient': 'sampleID',
        'Expression': f'{target}expression',
        'Group': f'{target}group'
    }, inplace=True)
    print("  [진단 로그] 1-1. CSV 파일 로드 완료.")
    
    print("  [진단 로그] 1-2. 마스터 SAV 파일 로드 시도 중 (pyreadstat)...")
    df_clin, meta = pyreadstat.read_sav(clin_file)
    print("  [진단 로그] 1-2. SAV 파일 로드 완료.")
    
    # [중요 복구] 기존 SAV 파일에 동일한 유전자 컬럼이 있다면 충돌 방지를 위해 먼저 삭제 (x, y 접미사 생성 방지)
    new_cols = [f'{target}expression', f'{target}group']
    cols_to_drop = [col for col in new_cols if col in df_clin.columns]
    if cols_to_drop:
        print(f"  [진단 로그] 기존 '{target}' 관련 컬럼 중복 발견하여 구버전 데이터 삭제 진행...")
        df_clin.drop(columns=cols_to_drop, inplace=True)
    
    print("  [진단 로그] 1-3. 두 데이터프레임 병합(Merge) 진행 중...")
    df_merged = pd.merge(df_clin, df_onco, on='sampleID', how='left')
    print("  [진단 로그] 1-3. 병합 완료.")
    
    # [중요 복구] 병합된 최신 데이터를 다시 SAV와 CSV로 물리적으로 저장
    print("  [진단 로그] 1-4. 병합본을 마스터 SAV 파일로 저장 진행 중...")
    pyreadstat.write_sav(df_merged, clin_file)
    print("  [진단 로그] 1-4. SAV 파일 물리적 저장 완료.")
    
    print("  [진단 로그] 1-5. 백업용 CSV 파일 저장 진행 중...")
    csv_path = clin_file.replace('.sav', '.csv')
    df_merged.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("  [진단 로그] 1-5. 백업 CSV 저장 완료.")
    
    return df_merged

# =====================================================================
# 2. 기초 및 다변량 분석 파이프라인 (원본 로직 복구)
# =====================================================================
def analyze_chi_square(df, target_col='gene_group', row_vars=None):
    print("--- 2-1. 예후 분석 (Chi-square Test) ---")
    if target_col not in df.columns:
        print(f"  [경고] '{target_col}' 컬럼이 존재하지 않습니다.")
        return pd.DataFrame()
        
    if row_vars is None:
        row_vars = ['ageG', 'gender', 'pathologic_stage', 'pathologic_M', 'pathologic_N', 'pathologic_T', 'SmokingStatus']
    
    target_cats = sorted(df[target_col].dropna().unique())
    if len(target_cats) == 0:
        target_cats = ['Group1', 'Group2']
        
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
            
        crosstab_print = pd.crosstab(df[var], df[target_col], margins=True, margins_name="전체")
        print(f"[{target_col} vs {var}]")
        print(crosstab_print)
        print(f"Chi-square: {chi2:.4f}, p-value: {p_val_formatted}\n")
        
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
        
    # [기존 로직 완벽 복구] 상관관계 분석 변수 전체에 대해 한 번에 결측치 제거 (Listwise Deletion)
    valid_data = df[valid_genes].dropna()
    
    pearson_matrix = valid_data.corr(method='pearson')
    print("[Pearson 상관계수 행렬]")
    print(pearson_matrix.round(3))
    print("\n")
    
    table2_data = []
    for i in valid_genes:
        row_r = [i, 'R']
        row_p = ['', 'P']
        for j in valid_genes:
            if i == j:
                row_r.append("1")
                row_p.append("")
            else:
                r_val, p_val = stats.pearsonr(valid_data[i], valid_data[j])
                row_r.append(f"{r_val:.3f}")
                row_p.append(f"{p_val:.3f}" if p_val >= 0.001 else "<0.001")
        table2_data.append(row_r)
        table2_data.append(row_p)
        
    col_names = ['Variable', 'Stat'] + valid_genes
    df_table2 = pd.DataFrame(table2_data, columns=col_names)
    return df_table2

def analyze_glm_multivariate(df, mode, expr_col='target_expression', group_col='gene_group'):
    print("--- 2-3. 다변량 일반선형모형 (GLM / OLS) - 4개 개별 모델 분석 ---")
    if mode == "LUAD":
        age_var = "age_at_initial_pathologic_diagnosis"
    elif mode == "LUSC":
        age_var = "age"
        
    base_vars = [age_var, 'gender', 'number_pack_years_smoked']
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
            
        # Formula 작성 (Group 변수는 순환 논리 방지를 위해 제외)
        predictors = []
        if age_var in valid_base_vars: predictors.append(age_var)
        if 'gender' in valid_base_vars: predictors.append("C(gender)")
        if 'number_pack_years_smoked' in valid_base_vars: predictors.append("number_pack_years_smoked")
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

def analyze_kaplan_meier(target, df, mode, save_dir, group_col='gene_group', plot_type=1):
    print("--- 2-4. Kaplan-Meier 생존분석 ---")
    if group_col not in df.columns or 'OS.time' not in df.columns or 'OS' not in df.columns:
        print("  [경고] KM 생존 분석에 필요한 데이터가 없습니다.")
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
    
    # SPSS 방식 범례(Legend) 핸들 커스텀 설정
    import matplotlib.lines as mlines
    from matplotlib.legend_handler import HandlerLine2D
    
    class StepLine2D(mlines.Line2D):
        pass
        
    class StepHandler(HandlerLine2D):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # 사진과 완벽히 일치하는 _|‾| (위로 꺾이고 오른쪽으로 가다가 아래로 꺾이는) 모양
            xdata = [0, width*0.35, width*0.35, width*0.85, width*0.85]
            ydata = [height*0.2, height*0.2, height*0.8, height*0.8, height*0.2]
            line = mlines.Line2D(xdata, ydata, color=orig_handle.get_color(), lw=1.5)
            line.set_transform(trans)
            return [line]

    # ---------------- OS 분석 ----------------
    plt.figure(figsize=(8, 6)) # 가로세로 비율을 사진과 동일하게(4:3) 설정
    ax_os = plt.gca()
    
    for group in groups:
        mask = (df_km[group_col] == group)
        time = df_km.loc[mask, 'OS.time'].dropna()
        event = df_km.loc[mask, 'OS'].dropna()
        if len(time) > 0:
            color = color_map[group]
            kmf.fit(time, event_observed=event)
            kmf.plot_survival_function(ax=ax_os, ci_show=False, color=color,
                                       show_censors=True, censor_styles={'marker': '+', 'mew': 1, 'ms': 6, 'mec': color})
            
    if plot_type == 1:
        ax_os.set_title(f'{mode}_OS 생존함수')
        ax_os.set_xlabel('OS.time')
        ax_os.set_ylabel('누적 생존함수')
    elif plot_type == 2:
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
        if group in color_map:
            # 커스텀 _|‾| 꺾인 선 마커 추가
            handles_os.append(StepLine2D([0], [0], color=color_map[group]))
            labels_os.append(group)
    for group in groups:
        if group in color_map:
            # 십자가(+) 마커 추가 (사진처럼 -+- 모양이 되도록 실선 속성 추가)
            handles_os.append(mlines.Line2D([], [], color=color_map[group], marker='+', linestyle='-', lw=1.5, mew=1, ms=6))
            if plot_type == 1:
                labels_os.append(f'{group}-중도절단')
            elif plot_type == 2:
                labels_os.append(f'{group}-censored')
            
    if len(groups) > 1:
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
    # 기존 결과 대체 형식 (timestamp 제거) 및 덮어쓰기 강제
    km_plot_path_os = os.path.join(save_dir, f'{target}_{mode}_OS_KM.png')
    if os.path.exists(km_plot_path_os):
        try: os.remove(km_plot_path_os)
        except: pass
    plt.savefig(km_plot_path_os)
    plt.close()
    print(f"Kaplan-Meier (OS) 이미지 저장 완료: '{km_plot_path_os}'")
        
    # ---------------- RFS 분석 ----------------
    if 'RFS.time' in df.columns and 'RFS' in df.columns:
        df_rfs = df_km.dropna(subset=['RFS.time', 'RFS'])
        if not df_rfs.empty:
            plt.figure(figsize=(8, 6)) # 가로세로 비율을 사진과 동일하게(4:3) 설정
            ax_rfs = plt.gca()
            
            for group in groups:
                mask = (df_rfs[group_col] == group)
                time = df_rfs.loc[mask, 'RFS.time'].dropna()
                event = df_rfs.loc[mask, 'RFS'].dropna()
                if len(time) > 0:
                    color = color_map[group]
                    kmf.fit(time, event_observed=event)
                    kmf.plot_survival_function(ax=ax_rfs, ci_show=False, color=color,
                                               show_censors=True, censor_styles={'marker': '+', 'mew': 1, 'ms': 6, 'mec': color})
                    
            if plot_type == 1:
                ax_rfs.set_title(f'{mode}_RFS 생존함수')
                ax_rfs.set_xlabel('RFS.time')
                ax_rfs.set_ylabel('누적 생존함수')
            elif plot_type == 2:
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
                if group in color_map:
                    handles_rfs.append(StepLine2D([0], [0], color=color_map[group]))
                    labels_rfs.append(group)
            for group in groups:
                if group in color_map:
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
            # 기존 결과 대체 형식 (timestamp 제거) 및 덮어쓰기 강제
            km_plot_path_rfs = os.path.join(save_dir, f'{target}_{mode}_RFS_KM.png')
            if os.path.exists(km_plot_path_rfs):
                try: os.remove(km_plot_path_rfs)
                except: pass
            plt.savefig(km_plot_path_rfs)
            plt.close()
            print(f"Kaplan-Meier (RFS) 이미지 저장 완료: '{km_plot_path_rfs}'\n")

def analyze_cox_regression(target, df, mode, save_dir, group_col='gene_group', categorical_vars=None, continuous_vars=None):
    print("--- 2-5. Cox 회귀분석 ---")
    
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
        print("  [경고] 필수 컬럼이 없어 Cox 분석을 건너뜁니다.")
        return pd.DataFrame()
        
    df_cox = df[valid_cols].dropna()
    
    if df_cox.empty:
        print("  [경고] 결측치 제거 후 Cox 분석할 데이터가 없습니다.")
        return pd.DataFrame()
        
    # 2. 명목형 변수들만 명시적으로 선택하여 더미 변환 (연속형은 원형 그대로 유지)
    dummy_cols = [c for c in categorical_vars if c in df_cox.columns]
    df_cox_dummy = pd.get_dummies(df_cox, columns=dummy_cols, drop_first=True)
    
    cph = CoxPHFitter()
    
    df_table3 = pd.DataFrame()
    try:
        cph.fit(df_cox_dummy, duration_col='OS.time', event_col='OS')
        print(cph.summary[['exp(coef)', 'p']])
        
        summary = cph.summary
        df_table3 = pd.DataFrame({
            'Clinical Variable': summary.index,
            'Hazard Ratio (HR)': summary['exp(coef)'].round(3),
            'P-value': summary['p'].apply(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
        })
        
        plt.figure(figsize=(10, 6))
        cph.plot()
        plt.title(f'Cox Regression - Forest Plot - {target}({mode})')
        plt.tight_layout()
        # 기존 결과 대체 형식 (timestamp 제거) 및 덮어쓰기 강제
        cox_plot_path = os.path.join(save_dir, f'{target}_{mode}_Cox.png')
        if os.path.exists(cox_plot_path):
            try: os.remove(cox_plot_path)
            except: pass
        plt.savefig(cox_plot_path)
        plt.close()
        print(f"Cox Forest Plot 저장 완료: '{cox_plot_path}'\n")
        
    except Exception as e:
        print(f"Cox 분석 중 오류 발생: {e}")
        
    return df_table3

# =====================================================================
# 3. 메인 실행 블록 (LUAD, LUSC 순차적 자동 실행)
# =====================================================================
if __name__ == "__main__":
    super_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 여기서 플롯 출력 타입을 지정합니다. 
    # 1: 기존 한글 스타일(범례에 P-value 포함) / 2: 영문 스타일(그래프 내부에 P-value 삽입)
    CURRENT_PLOT_TYPE = 2
    
    for mode in ['LUAD', 'LUSC']:
        print(f"\n{'='*60}")
        print(f"          {mode} 자동화 분석 파이프라인 시작")
        print(f"{'='*60}")
        
        master_sav_path = os.path.join(super_dir, "database", '(LUAD) lung adenocarcinoma.sav' if mode == 'LUAD' else '(LUSC) lung squamous cell carcinoma.sav')
        
        if not os.path.exists(master_sav_path):
            print(f"오류: 마스터 임상 파일이 존재하지 않습니다: {master_sav_path}")
            continue

        input_csv_dir = os.path.join(super_dir, f"input_{mode}_csv")
        completed_csv_dir = os.path.join(super_dir, "completed_csv")
        result_dir = os.path.join(super_dir, "result")
        plot_base_dir = os.path.join(super_dir, "plot")

        for d in [input_csv_dir, completed_csv_dir, result_dir, plot_base_dir]:
            os.makedirs(d, exist_ok=True)

        if isinstance(sys.stdout, PrintLogger):
            sys.stdout.flush()
        # 기존 결과 대체 형식 (timestamp 제거)
        result_txt_path = os.path.join(result_dir, f"result_single_{mode}.txt")
        sys.stdout = PrintLogger(result_txt_path)

        target_csv_files = glob.glob(os.path.join(input_csv_dir, "*.csv"))
        analysis_targets = []
        
        if len(target_csv_files) > 0:
            for csv_file in target_csv_files:
                file_name = os.path.basename(csv_file)
                target_gene = file_name.replace('.csv', '').split('_')[-1]
                analysis_targets.append((target_gene, csv_file, file_name))
        else:
            print(f"'{input_csv_dir}' 폴더에 병합할 CSV 파일이 없습니다.")
            target_input = input(f"[{mode}] 마스터 파일(SAV)에 이미 존재하는 유전자로 분석하려면 이름을 입력하세요\n(여러 개는 쉼표 구분, 없으면 그냥 엔터): ").strip()
            if not target_input:
                print(f"[{mode}] 입력된 유전자가 없어 다음 암종으로 넘어갑니다.")
                continue
            for g in target_input.split(','):
                if g.strip():
                    analysis_targets.append((g.strip(), None, None))
        
        for target_gene, csv_file, file_name in analysis_targets:
            group_col_name = f"{target_gene}group"
            expr_col_name = f"{target_gene}expression"
            
            print(f"\n[{target_gene} 유전자 분석 시작]")
            
            # 1. 병합 또는 기존 데이터 로드
            if csv_file:
                merged_df = load_and_merge_data(target=target_gene, onco_file=csv_file, clin_file=master_sav_path)
            else:
                print(f"--- 1. 기존 데이터 로드 (Target: {target_gene}, 병합 과정 건너뜀) ---")
                merged_df, _ = pyreadstat.read_sav(master_sav_path)
            
            # 방어 코드 추가 (KeyError 방지)
            if group_col_name not in merged_df.columns:
                print(f"  [경고] 병합 후 '{group_col_name}' 데이터가 없습니다. 이 유전자는 분석을 건너뜁니다.")
                print("="*60)
                continue
            
            # 2. 분석 및 엑셀 표 데이터 생성
            df_table1 = analyze_chi_square(merged_df, target_col=group_col_name)
            
            # [수정됨] 암종별로 통제(교란) 변수로 사용할 연속형 유전자 발현량(gene_vars) 분리
            if mode == "LUAD":
                gene_vars = ['KrasExpression', 'TP53Expression', 'ALKExpression', 'BRAFExpression']
                target_genes_for_corr = [expr_col_name, 'number_pack_years_smoked', 'age_at_initial_pathologic_diagnosis'] + gene_vars
            else:
                gene_vars = ['TP53Expression', 'CDKN2AExpression', 'SOX2Expression', 'PIK3CAExpression', 'NOTCH1Expression']
                target_genes_for_corr = [expr_col_name, 'number_pack_years_smoked', 'age'] + gene_vars
            
            df_table2 = analyze_bivariate_correlation(merged_df, gene_list=target_genes_for_corr)
            
            # 다변량 분석 (GLM / OLS) - Table 4 반환
            df_table4 = analyze_glm_multivariate(merged_df, mode, expr_col=expr_col_name, group_col=group_col_name)
            
            # 생존 & Cox 분석 (plot_type 전달)
            analyze_kaplan_meier(target_gene, merged_df, mode, plot_base_dir, group_col=group_col_name, plot_type=CURRENT_PLOT_TYPE)
            
            # [수정됨] Cox 생존 분석 시 명목형/연속형 변수를 명확히 분리하여 주입
            age_col = 'age_at_initial_pathologic_diagnosis' if mode == "LUAD" else 'age'
            cox_categorical = [group_col_name, 'gender', 'pathologic_stage']
            cox_continuous = [age_col, 'number_pack_years_smoked'] + gene_vars
            
            df_table3 = analyze_cox_regression(
                target_gene, merged_df, mode, plot_base_dir, 
                group_col=group_col_name,
                categorical_vars=cox_categorical, 
                continuous_vars=cox_continuous
            )
            
            # 3. 분석 결과를 Excel 파일로 예쁘게 저장 (Table 4 추가)
            # 기존 결과 대체 형식 (timestamp 제거) 및 덮어쓰기 강제
            excel_save_path = os.path.join(result_dir, f"{target_gene}_{mode}_Tables.xlsx")
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
            print(f"✅ 논문용 표 엑셀 파일 저장 완료: '{excel_save_path}'")
            
            # 처리 완료 파일 이동 또는 패스
            if csv_file:
                completed_path = os.path.join(completed_csv_dir, file_name)
                shutil.move(csv_file, completed_path)
                print(f"'{file_name}' 분석 완료 및 completed_csv 폴더로 이동됨.\n")
            else:
                print(f"'{target_gene}' 기존 데이터 재분석 및 결과 출력 완료.\n")
            print("="*60)
            
        print(f"\n=== {mode} 파이프라인 완료 ===")