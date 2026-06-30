import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# [커스텀 범례 객체 정의] - 생존 분석 그래프 특유의 계단형(Step) 선을 범례에 표시하기 위함
class StepLine2D(Line2D):
    pass

class StepHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # 범례 안에 계단(step) 모양의 선을 그리기 위한 좌표 설정
        x = [0, width / 2, width / 2, width]
        y = [height / 2 + height / 4, height / 2 + height / 4, height / 2 - height / 4, height / 2 - height / 4]
        line = plt.Line2D(x, y, 
                          color=orig_handle.get_color(), 
                          linewidth=orig_handle.get_linewidth(),
                          linestyle=orig_handle.get_linestyle())
        line.set_transform(trans)
        return [line]


def analyze_custom_kaplan_meier(df, target, group_col, time_col, event_col, save_dir, 
                                cancer_type='Unknown', sample_id_col='sampleID', 
                                exclude_samples=None, replace_dict=None):
    """
    사용자 정의형 Kaplan-Meier 생존 분석 파이프라인 함수
    
    Parameters:
    - df: 분석할 전체 데이터프레임
    - target: 분석 대상 유전자명 또는 타겟명 (예: 'EGFR')
    - group_col: 타겟 그룹 컬럼명 (예: 'gene_group') - 'High'와 'Low'만 필터링됨
    - time_col: 생존 시간 컬럼명 (예: 'OS.time')
    - event_col: 이벤트 여부 컬럼명 (예: 'OS')
    - save_dir: 그래프를 저장할 디렉토리 경로
    - cancer_type: 암종 이름 (결과 반환 DataFrame용)
    - sample_id_col: 샘플 ID가 기록된 컬럼명
    - exclude_samples: 분석에서 제외할 샘플 ID 리스트
    - replace_dict: 특정 샘플의 값을 변경할 딕셔너리 (예: {'sample_1': {'gene_group': 'High'}})
    """
    
    # 1. 폰트 및 시각화 기본 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 2. 데이터 전처리 로직
    data = df.copy()
    
    # (선택 사항) 특정 샘플 값 강제 변경 로직
    if replace_dict and sample_id_col in data.columns:
        for s_id, changes in replace_dict.items():
            for col, new_val in changes.items():
                data.loc[data[sample_id_col] == s_id, col] = new_val
                
    # (선택 사항) 특정 샘플 제외 로직
    if exclude_samples and sample_id_col in data.columns:
        data = data[~data[sample_id_col].isin(exclude_samples)]
        
    # 결측치 제거
    data = data.dropna(subset=[time_col, event_col])
    
    # 'High'와 'Low' 그룹 데이터만 필터링
    data = data[data[group_col].isin(['High', 'Low'])]

    #전처리
    data[event_col] = data[event_col].map({'Alive': 0, 'Dead': 1})
    
    # 3. 통계 검정 (Log-rank Test)
    groups = data[group_col].unique()
    
    # 그룹 개수에 따른 분기 (현재 필터링 기준으로는 2개지만 파이프라인 확장성을 위해 다변량 통계검정 유지)
    if len(groups) == 2:
        group1, group2 = groups[0], groups[1]
        mask1 = data[group_col] == group1
        mask2 = data[group_col] == group2
        
        test_result = logrank_test(data.loc[mask1, time_col], data.loc[mask2, time_col],
                                   event_observed_A=data.loc[mask1, event_col],
                                   event_observed_B=data.loc[mask2, event_col])
    elif len(groups) > 2:
        test_result = multivariate_logrank_test(data[time_col], data[group_col], data[event_col])
    else:
        raise ValueError("분석 가능한 그룹이 2개 미만입니다 (모두 같은 그룹이거나 데이터가 부족함).")

    p_val = test_result.p_value
    chi2 = test_result.test_statistic

    # P-value 포맷팅
    if p_val < 0.001:
        p_text = "P < 0.001"
    else:
        p_text = f"P = {p_val:.3f}"

    # 4. 시각화 (Matplotlib)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    colors = {'High': '#00a2e8', 'Low': '#b51a5e'}
    kmf_dict = {}
    
    for group_name in ['High', 'Low']:
        mask = data[group_col] == group_name
        if mask.sum() > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(data.loc[mask, time_col], data.loc[mask, event_col], label=group_name)
            
            # KM 곡선 및 중도절단(Censored) 표시 (십자 마커 사용 및 디테일 설정)
            # TypeError 해결: censor_styles 안의 'color' 속성 제거 (메인 color를 자동으로 따라감)
            kmf.plot_survival_function(ax=ax, color=colors[group_name], ci_show=False, 
                                       show_censors=True, 
                                       censor_styles={'marker': '+', 'mew': 1, 'ms': 6},
                                       drawstyle='steps-post', linewidth=1.5)
            kmf_dict[group_name] = kmf

    # 축 라벨 디자인
    ax.set_xlabel('Time(Days)', fontsize=12, fontweight='bold')
    
    # 이벤트 컬럼명을 기반으로 Y축 라벨 자동 유추
    if 'OS' in event_col.upper():
        ylabel = 'Overall Survival'
    elif 'RFS' in event_col.upper():
        ylabel = 'Relapse-Free Survival'
    else:
        ylabel = f'{event_col} Probability'
        
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_ylim(0.0, 1.05)

    # 테두리(Spines) 숨기기
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # P-value 텍스트 그래프 내부(좌측 하단) 삽입
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='bottom', ha='left')

    # 범례 커스텀 생성 (계단형 렌더러 적용)
    custom_lines = []
    labels = []
    for g_name in ['High', 'Low']:
        if g_name in kmf_dict:
            n_count = len(data[data[group_col] == g_name])
            custom_lines.append(StepLine2D([0], [0], color=colors[g_name], lw=1.5))
            labels.append(f"{g_name} (n={n_count})")
            
    leg = ax.legend(custom_lines, labels, 
                    handler_map={StepLine2D: StepHandler()},
                    frameon=False, loc='upper right', title=f'{target}')
    
    # 범례 제목 볼드체 처리
    plt.setp(leg.get_title(), fontweight='bold')

    # 5. 그래프 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'KaplanMeier_{target}_{event_col}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False)
    plt.close(fig) # 메모리 관리

    # 6. 분석 결과 반환 (1줄 DataFrame)
    result_df = pd.DataFrame([{
        'Cancer Type': cancer_type,
        'Target': target,
        'Survival Type': ylabel,
        'Chi-square': round(chi2, 4),
        'P-value': p_val,
        'High_N': len(data[data[group_col] == 'High']),
        'Low_N': len(data[data[group_col] == 'Low'])
    }])
    
    return result_df

# ==========================================
# [사용 예시]
if __name__ == "__main__":
    
    super_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(super_dir, 'database')
    csv_dir = os.path.join(super_dir, 'completed_csv')
    df1 = pd.read_csv(os.path.join(csv_dir, 'LUAD_50_50_XRCC3.csv'))
    # 예외 처리 조건 세팅
    exclude_list = [] #예시 exclude_list = ['S1', 'S2'] 
    replace_info = {} #예시 replace_info = {'S3': {'gene_group': 'High'}} 


    result = analyze_custom_kaplan_meier(
        df=df1, 
        target='XRCC3', 
        group_col='Group', 
        time_col='Days', 
        event_col='Status', 
        save_dir='./km_results',
        cancer_type='LUAD',
        exclude_samples=exclude_list,
        replace_dict=replace_info
    )
    print(result)