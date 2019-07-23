# GDP-KPRN

***Last edit : 22 July 2019***

Recommender system referencing [KPRN](https://arxiv.org/pdf/1811.04540.pdf), [original github](https://github.com/eBay/KPRN), trained using custom [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset.

> The model implemented has slight difference where no pooling layer added at the end of LSTM.

# Domain of problems
*Given a path between user and an item, predict how likely the user will interact with the item*

# Contents
- cache : temporary files used in training
- data : contains dataset (custom ml-20m dataset where only movies shows up in Ripple-Net's knowledge graph used) used in training.
- **log** : contains training result stored in single folder named after training timestamp.
- **test** : contains jupyter notebook used in testing the trained models

- KPRN-LSTM.ipynb : notebook to train model

### Note
    *italic* means this folder is ommited from git, but necessary if you need to run experiments
    **bold** means this folder has it's own README, check it for detailed information :)

# How to run
1. Unzip `ratings_re.zip` and `ratings_re.z01` in `/data`
2. To preprocess, run `Preprocess.ipynb` notebook or `preprocess.py`
    ~~~
    python3 data/preprocess.py
    ~~~

3. To train, run `KPRN-LSTM.ipynb` notebook

# Preparing 
## Installing dependencies 

    pip3 install -r requirements.txt

## **! Caching warning !**
To start using new dataset, or if you wish to generate new dataset, please delete all items inside `/cache`

# Training
## How to change hyper parameter
Open `KPRN-LSTM.ipynb` and change the model parameters

# Testing / Evaluation
## How to check training result
1. Find the training result folder inside `/log` (find the latest), copy the folder name.
2. Create copy of latest jupyter notebook inside `/test` folder.
3. Rename folder to match a folder in `/log` (for traceability purpose).
4. Replace `TESTING_CODE` at the top of the notebook.
5. Run the notebook

# Final result
| Metric             | Value       |
|--------------------|-------------|
| Average prec@10    | +- 0.06     |
| Diversity@10 n=10  | 0.60 - 0.80 |
| Evaluated on       | 1.5k users  |

# Other findings

**KPRN relies heavily upon paths**, and those paths are *handcrafted* by using the knowledge-graph. The paths are also sampled from hundreds of million possible paths.

- To find paths  
    ```
    (user -> seed item (eg: Castle on The Hill) -> entity (eg: Ed Sheeran) -> suggestion (eg: Perfect)) 
    ```
    from each seed, we can extract millions of paths (if not sampled), even after sampled using only one relation (eg: same artist, same albums, etc per seed, it still generates around 8k-10k path per seed.

- Each user has multiple item work as seed (typically 20+), this need to be sampled again to reduce paths generated and reduce computational cost. 

- We do make sure each item in suggestion has about 4 - 7 paths

- At this point, we only evaluate on around 75 - 150 path per user, out of possible hundred million possible paths  

- That's a huge possible source of sampling bias, but at the same time, it's kinda impossible to search through all paths.

- Looking from the result of KPRN, the usage of KG might turn out to be quite promising, especially to improve the diversity of suggestion.
  
- The downside of using KPRN is that the result rely heavily on 'handcrafted' paths, which undergoes a lot of downsampling steps. 
  
- Summary compared to non-KG RecSys: **Big improvement in terms of diversity, similar/worse Prec@k result**

# Experiment notes
- path generation & predict time : about 4k path / second
- different sampling method and sampling parameter has insignificant effect

# Author
- Jessin Donnyson - jessinra@gmail.com

# Contributors
- Michael Julio - michael.julio@gdplabs.id
- Fallon Candra - fallon.candra@gdplabs.id