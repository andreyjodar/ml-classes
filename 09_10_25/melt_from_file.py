import os
import pandas as pd

def melt_from_file(input_file, output_file):
    origin_dataframe = pd.read_csv(input_file)

    melt_dataframe = pd.melt(
        origin_dataframe,
        id_vars=['dataset', 'classifier', 'metric', 'author'],
        value_vars=[f'v{i}' for i in range(1, 21)],
        var_name='value_name',
        value_name='value'
    )

    output_dir = '09_10_25'
    output_path = os.path.join(output_dir, f'{output_file}.csv')

    os.makedirs(output_dir, exist_ok=True)
    melt_dataframe.to_csv(output_path, index=False)
    return melt_dataframe


if __name__ == '__main__':
    df_melted = melt_from_file('09_10_25/all-test.csv', 'melt-result')
    print("Arquivo salvo com sucesso!")
