# process and extract targeted json results into csv
# covenient for copy
import argparse
import pandas as pd
import json


def load_line_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            yield json.loads(line)


def main(args):

    metrics = load_line_json(args["source_file"])
    result_dict_list = {
        'time': [], 
        'subname': [],
        "dataset": [],
        'seed': [],
        'arch': [],
        'mode': [],
        'hit10(active)': [],
        'hit10(static)': [], 
        'hit_diff': [],
        'hit10(all)': [],
        'ndcg10(active)': [],
        'ndcg10(static)': [], 
        'ndcg_diff': [],
        'ndcg10(all)': []
    }

    for metric in metrics:
        result_dict_list['time'].append(metric['time'])
        result_dict_list['subname'].append(metric['subname'])
        result_dict_list['dataset'].append(metric['dataset'])
        result_dict_list['seed'].append(metric['seed'])
        result_dict_list['arch'].append(metric['architecture'])
        result_dict_list['mode'].append(metric['mode'])
        result_dict_list['hit10(active)'].append(metric["metric"]['hit_ratio@10(active)'])
        result_dict_list['hit10(static)'].append(metric["metric"]['hit_ratio@10(static)'])
        result_dict_list['hit10(all)'].append(metric["metric"]['hit_ratio@10(all)'])
        result_dict_list['hit_diff'].append(None)
        result_dict_list['ndcg10(active)'].append(metric["metric"]['ndcg@10(active)'])
        result_dict_list['ndcg10(static)'].append(metric["metric"]['ndcg@10(static)'])
        result_dict_list['ndcg_diff'].append(None)
        result_dict_list['ndcg10(all)'].append(metric["metric"]['ndcg@10(all)'])
        
        

    df = pd.DataFrame(result_dict_list)
    print(df.columns)
    print(df.info())

    order = [ 'dataset','time', 'arch','subname', 'seed',  'mode',
             'hit10(active)', 'hit10(static)', 'hit_diff', 'hit10(all)', 'ndcg10(active)', 'ndcg10(static)', 'ndcg_diff', 'ndcg10(all)'
             ]

    df_ordered = df[order]
    df_ordered.to_csv("result_summation.csv")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
