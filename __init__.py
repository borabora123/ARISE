import torch
import pandas as pd
print(torch.cuda.is_available())
print(torch.__version__)
submit = pd.read_csv('/submit_res_net_50_64_42.csv')
submit['ID'] = submit['patient_id'].astype(str)+'_'+submit['joint_id'].astype(str)
submit['PAD'] = 0
print(submit)
submit.to_csv('C:\\Users\\User\\PycharmProjects\\ARISE\\submit.csv', index=False)