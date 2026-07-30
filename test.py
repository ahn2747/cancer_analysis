import pandas
import os

super_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(super_dir, 'database')
csv_dir = os.path.join(super_dir, 'completed_csv')

df1 = pandas.read_csv(os.path.join(db_dir, '(LUAD) lung adenocarcinoma.csv'))
df2 = pandas.read_csv(os.path.join(csv_dir, 'LUAD_50_50_XRCC3.csv'))

print(df1.head())
print(df2.head())

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



if __name__ == "__main__":
    # check_patient_match(df1, df2)
    gene_group_Ntest(df1, 'ADAM10group')
    pass

    