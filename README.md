## II-GOOT

This is the Pytorch implementation of “Intra- and Inter-Group Optimal Transport for User-Oriented Fairness in Recommender Systems”, which has published in AAAI 2024.

## Environment Requirement

```shell
pip install -r requirements.txt
```

## Dataset

- Beauty_2014
- Grocery_2014
- Health_2018

see more: http://jmcauley.ucsd.edu/data/amazon/

## Setup

For the shell scripts in bash, set `data_dir` to the directory with data in your device. Set `output_dir` to a directory where you want to save the model checkpoints, performance and so on.

## Run the code

```shell
# run baseline
bash bash/train.sh -d amazon -e Health_2018 -f remove_100bigger -s 42 -c 0 -b Health_normal -a LightGCN -m normal

# run II-GOOT
bash bash/train.sh -d amazon -e Health_2018 -f remove_100bigger -s 42 -c 0 -b Health_normal -a LightGCN -m fair

# test the above model
bash bash/test.sh -d amazon -e Health_2018 -f remove_100bigger -s 42 -c 0 -b Health_normal -a LightGCN -m fair

# save the above model's embedding after t-sne
bash bash/test_tsne.sh -d amazon -e Health_2018 -f remove_100bigger -s 42 -c 0 -b Health_normal -a LightGCN -m fair

# if you want to designate the ratio of advantaged user, say you want the top 15% user as the advantaged user
bash bash/train_p.sh -d amazon -e Health_2018 -f remove_100bigger -s 42 -c 0 -b Health_normal -a LightGCN -m fair -p 0.15
```

#### options

- -d data_subfolder
- -e data_subsubfolder
- -f data_subsubsubfolder
- -s seed
- -c cuda_id
- -b unique_name_used_to_store_results
- -a architecture, select from `LightGCN`, `MF`, `NeuMF`
- -m mode, select from `normal`, `fair`, `inter`, `intra`

#### Gather results

```she
python metric_summation.py --source_file <output_dir(in setup)/performance.txt>
```

## project structure

```
 ─myCode_ot_final
    │  metric_summation.py  summarize training results into csv
    │  train.py              main file for training
    │  train_tsne.py         similar to train.py, commented test metric computaion,
    |                        compute and save user embeddings after t-sne
    │  
    ├─bash
    │      test.sh           test on a given checkpoint
    │      test_tsne.sh      test to generate and save user embedding after t-sne
    │      train.sh          train model given hyperparameters and datasets
    │      train_p.sh        train model with addtional specification of the
    |                        ratio of advantaged users
    │      
    ├─model
    │      LightGCN.py       LightGCN backbone
    │      MF.py             MF backbone
    │      model_configs.py  model configs for all models in this dir
    │      NeuMF.py          NeuMF backbone
    │      __init__.py
    │      
    ├─utils
    │      datamodule.py    load and process data
    │      metrics.py       functions to compute NDCG and HIT
    │      result.py        process model's prediction results and save to file
    │      visualize.py     function to generate low dimension embeddings using
    |                       tsne given original embeddings
    │      __init__.py
    │      
    └─visualize
        │  dataset_stat_core.xlsx  excel with results and  data to draw $\xi$-UOF and 
        |                          performance change in SHEET `variational percentage`
        │  visualize_loss2.ipynb   draw L_{inter} change
        │  visualize_trend.ipynb   draw $\xi$-UOF and performance change with 
        |                          different ratio for advantaged user
        │  visualize_tsne.ipynb    visualize user embeddings
        │  
        ├─loss                     model loss for visualize_loss2.ipynb
        │          
        └─tsne
            ├─Beauty_2014         data and figure for visualize_tsne.ipynb 
            │      
            ├─Grocery_2014        data and figure for visualize_tsne.ipynb
            │      
            └─Health_2018         data and figure for visualize_tsne.ipynb
  
                    

```

