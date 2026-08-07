import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import re


# 엑셀 컬럼이 'Cohort', 'Pathway', 'NES', 'FDR' 로 구성되어 있다고 가정합니다.
# dir setting
super_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(super_dir, 'database')
result_dir = os.path.join(super_dir, 'GSEA_figure')

# path setting
def plot_gsea_dotplot(file):
    # ---------------------------------------------------------
    # 테스트용 가상 데이터 (실제 데이터 사용 시 이 부분은 지우세요)
    # data = {
    #     'Cohort': ['LUAD']*8 + ['LUSC']*8,
    #     'Enriched_group': ['High']*4 + ['Low']*4 + ['High']*4 + ['Low']*4,
    #     'Pathway': [
    #         'MTORC1 SIGNALING', 'MITOTIC SPINDLE', 'G2M CHECKPOINT', 'KRAS SIGNALING DN',
    #         'HEDGEHOG SIGNALING', 'P53 PATHWAY', 'COAGULATION', 'APICAL JUNCTION',
    #         # LUSC
    #         'MTORC1 SIGNALING', 'MYC TARGETS V1', 'E2F TARGETS', 'DNA REPAIR',
    #         'APICAL JUNCTION', 'IL6 JAK STAT3 SIGNALING', 'KRAS SIGNALING UP', 'APOPTOSIS'
    #     ],
    #     'NES': [
    #         2.2, 2.1, 1.9, -1.2, -1.25, -1.3, -1.35, -1.4,
    #         2.3, 2.2, 2.15, 1.9, -1.3, -1.35, -1.4, -1.45
    #     ],
    #     'FDR': [
    #         0.01, 0.015, 0.03, 0.3, 0.35, 0.4, 0.45, 0.5,
    #         0.005, 0.01, 0.02, 0.08, 0.3, 0.35, 0.4, 0.45
    #     ]
    # }
    # df = pd.DataFrame(data)
    # ---------------------------------------------------------
    
    # 전처리
    df = pd.read_excel(file)
    df = df.rename(columns={
        'Cancer_Type': 'Cohort',
        'Enriched_group': 'Enriched_group',
        'Hallmark_gene_set': 'Pathway',
        'NES': 'NES',
        'FDR_q-val': 'FDR'
    }, inplace=True)

    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)  # 컬럼명 공백 제거 및 '_'로 대체
    df['Cohort'] = df['Cohort'].ffill()
    df['Enriched_group'] = df['Enriched_group'].ffill()

    # 이미지를 따라 Y축(Pathway)을 NES 값 기준으로 위에서 아래로 정렬
    df = df.sort_values(by=['Cohort', 'NES'], ascending=[True, True])

    # 2. 1행 2열의 서브플롯(그래프 공간) 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=False)
    fig.suptitle('GSEA Pathways Enriched by ECT2 Expression Status', fontsize=18, fontweight='bold', y=1.05)

    # 사용할 색상 지정
    colors = {'High': '#d9534f', 'Low': '#428bca'} 

    # LUAD와 LUSC 각각 그리기
    for i, cohort in enumerate(['LUAD', 'LUSC']):
        ax = axes[i]
        cohort_data = df[df['Cohort'] == cohort]
        
        # 배경 X축 눈금선 (회색 실선) 및 0 기준선 (검은 점선) 세팅
        ax.xaxis.grid(True, linestyle='-', color='#e9ecef', zorder=0)
        ax.set_axisbelow(True) # 눈금선을 점 뒤로 보냄
        ax.axvline(0, color='black', linestyle='--', linewidth=1, zorder=1)
        
        # 각 Pathway 데이터를 한 줄씩 읽어오면서 점 찍기
        for _, row in cohort_data.iterrows():
            x = row['NES']
            y = row['Pathway']
            fdr = row['FDR']
            
            # 조건 1: 색상 (NES가 양수면 빨강, 음수면 파랑)
            color = colors['High'] if re.search(r'\b[hH]igh\b', row['Enriched_group']) else colors['Low']
            
            # 조건 2: 속 채움 (FDR < 0.25면 색상 채우기, 아니면 흰색 바탕에 테두리만)
            facecolor = color if fdr < 0.25 else 'white'
            
            # 조건 3: 크기 (-log10(FDR) 값에 비례하여 크기 조절, 상수는 적절히 조정 가능)
            # FDR이 0인 경우 에러 방지를 위해 아주 작은 값을 더해줍니다.
            size = -np.log10(fdr + 1e-10) * 120 
            
            ax.scatter(x, y, s=size, facecolors=facecolor, edgecolors=color, linewidths=1.5, zorder=2)
        
        # 서브플롯 제목 및 X축 라벨 설정
        ax.set_title(f'{cohort}: Enriched Hallmark Pathways', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=11, labelpad=10)
        
        # X축 범위 고정 (데이터에 따라 수정하세요)
        ax.set_xlim(-2.5, 2.5)
        
        # 차트 외곽선(Spine) 정리 (위, 오른쪽 선 숨기기)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')
        
        # Y축 눈금선(tick)은 안 보이게 지우고 텍스트만 남김
        ax.tick_params(axis='y', length=0)

    plt.tight_layout()

    # 3. 하단 커스텀 범례(Legend) 만들기
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='High ECT2', markerfacecolor=colors['High'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Low ECT2', markerfacecolor=colors['Low'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label='FDR q < 0.25', markerfacecolor='#888888', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='FDR q ≥ 0.25', markerfacecolor='w', markeredgecolor='#888888', markeredgewidth=1.5, markersize=9),
    ]

    # 범례 위치 조정 (가운데 하단)
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.4, -0.12), ncol=4, frameon=False, fontsize=10)

    # 크기 설명 텍스트 추가
    plt.text(0.7, -0.11, 'Circle area increases with -log10(FDR q-value)', transform=fig.transFigure, fontsize=9, verticalalignment='center')

    # 4. 고해상도 SVG로 저장
    output_path = os.path.join(result_dir, 'GSEA_Dotplot_custom.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"[{output_path}] 파일 생성 완료!")

if __name__ == "__main__":
    plot_gsea_dotplot(os.path.join(db_dir, 'ITGB1_GSEA.xlsx'))