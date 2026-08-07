import pandas as pd
import os

super_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(super_dir, 'database')
csv_dir = os.path.join(super_dir, 'completed_csv')

df1 = pd.read_csv(os.path.join(db_dir, '(LUAD) lung adenocarcinoma.csv'))
df2 = pd.read_csv(os.path.join(csv_dir, 'LUAD_50_50_XRCC3.csv'))


def check_patient_match(df1, df2):
    list1 = []
    list2 = []
    for Patient,Days,Status,Expression,Group in df2[['Patient','Days','Status','Expression','Group']].values:
        print(Patient, Days, Status, Expression, Group)
        val1 = df1[df1['sampleID'] == Patient]
        val2 = df2[df2['Patient'] == Patient]

        if val1['OS.time'].values[0] == val2['Days'].values[0]:
            print(f"Match found for Patient: {Patient}, {Days}")
            list1.append((Patient, Days))
        else:
            print(f"No match for Patient: {Patient}, {Days}")
            list2.append((Patient, Days, val1['OS.time'].values[0], val2['Status'].values[0], val1['OS'].values[0]))

    print(list1)
    print('--------------------')
    print(list2)

def gene_group_Ntest(df, tartget_col):
    data = df.copy()
    target = data[tartget_col]
    print(target)
    a1 = [i for i in target if i == 'High']
    a2 = [i for i in target if i == 'Low']
    print (len(a1), len(a2))


def gsea_df_test(rawdata):
    df = pd.read_excel(rawdata)
    
    # 빈칸(NaN)이 있다면 앞의 값으로 쭉 채워주는 함수 (ffill)
    # 만약 기존 로직처럼 정확히 LUAD, LUSC만 타겟팅해야 한다면 다른 처리가 필요할 수 있습니다.
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)  # 컬럼명 공백 제거 및 '_'로 대체
    df['Cancer_Type'] = df['Cancer_Type'].ffill()
    df['Enriched_group'] = df['Enriched_group'].ffill()
    return df



if __name__ == "__main__":
    # check_patient_match(df1, df2)
    # gene_group_Ntest(df1, 'ADAM10group')
    # pass
    print(gsea_df_test(os.path.join(db_dir, 'ITGB1_GSEA.xlsx')))

    