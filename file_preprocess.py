import pandas as pd

# read csv
# df = pd.read_csv('./yelp/2022-12-13-10-17-log.csv')
# df = pd.read_csv('./mnli/2022-12-13-11-12-log.csv')
df = pd.read_csv('./imdb/2022-12-14-15-14-log.csv')
# check column 'result_type' = 'Successful'
successful_df = df[df['result_type'] == 'Successful']
# print(successful_df.shape[0])
# check column 'perturbed_text', delete '[[' and ']]'
print(successful_df['perturbed_text'].iloc[0])

# successful_df['perturbed_text']=successful_df['perturbed_text'].str.replace('\[\[\[\[Premise\]\]\]\]\: ','')
# successful_df['perturbed_text']=successful_df['perturbed_text'].str.replace('\<SPLIT\>',' .')
# successful_df['perturbed_text']=successful_df['perturbed_text'].str.replace('\[\[\[\[Hypothesis\]\]\]\]:','')

successful_df['perturbed_text']=successful_df['perturbed_text'].str.replace('\[\[','')
successful_df['perturbed_text']=successful_df['perturbed_text'].str.replace('\]\]','')



print(successful_df['perturbed_text'].iloc[0])

# successful_df['perturbed_text'].to_csv('./yelp/processed.txt', sep='\t', index=False, header=False)
successful_df['perturbed_text'].to_csv('./imdb/processed.txt', sep='\t', index=False, header=False)
# # test
# from utils.helpers import read_lines
# test = read_lines('./mr/processed.txt')
# print(len(test))