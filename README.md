# GDP-KPRN

***Last edit : 29 July 2019***

Recommender system referencing [KPRN](https://arxiv.org/pdf/1811.04540.pdf), [original github](https://github.com/eBay/KPRN), trained using custom [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset.

> The model implemented has slight difference where no pooling layer added at the end of LSTM.

# Domain of problems
*Given a path between user and an item, predict how likely the user will interact with the item*

# Contents
- `/cache` : temporary files used in training
- `/data` : contains dataset (custom ml-20m dataset where only movies shows up in Ripple-Net's knowledge graph used) used in training.
- **`/log`** : contains training result stored in single folder named after training timestamp.
- **`/test`** : contains jupyter notebook used in testing the trained models

- `KPRN-LSTM.ipynb` : notebook to train model
- `main.py` : python3 version of KPRN-LSTM.ipynb

### Note
    *italic* means this folder is ommited from git, but necessary if you need to run experiments
    **bold** means this folder has it's own README, check it for detailed information :)

# Preparing 
## Installing dependencies 

    pip3 install -r requirements.txt

# How to run
1. Unzip `ratings_re.zip` and `ratings_re.z01` in `/data`
2. To preprocess, run `Preprocess.ipynb` notebook or `preprocess.py`
    ~~~
    python3 data/preprocess.py
    ~~~

3. To train, run `KPRN-LSTM.ipynb` notebook or `main.py`
    ~~~
    python3 main.py
    ~~~

## **! Caching warning !**
To start using new dataset, or if you wish to generate new dataset, please delete all items inside `/cache`

# Training
## How to change hyper parameter
Open `KPRN-LSTM.ipynb` or `main.py` and change the model parameters

# Testing / Evaluation
## How to check training result
1. Find the training result folder inside `/log` (find the latest), copy the folder name.
2. Create copy of latest jupyter notebook inside `/test` folder.
3. Rename folder to match a folder in `/log` (for traceability purpose).
4. Replace `TESTING_CODE` at the top of the notebook.
5. Run the notebook

# Final result

### KPRN - pool_size = 1 (no pooling)
| Evaluation size    | Prec@10 | Distinct Rate |  Unique items |
|--------------------|---------|---------------|---------------|
| Eval on  10 user   | 0.12028 |  0.70000      |    70         |
| Eval on  30 user   | 0.16667 |  0.60667      |   182         |
| Eval on 100 user   | 0.17471 |  0.38600      |   386         |

### KPRN - pool_size = 3
| Prec@10            | Value   | Distinct Rate |  Unique items |
|--------------------|---------|---------------|---------------|
| Eval on  10 user   | 0.20000 |  0.32000      |    32         |
| Eval on  30 user   | 0.24333 |  0.21000      |    63         |
| Eval on 100 user   | 0.25864 |  0.13400      |   134         |

### KPRN - pool_size = 5
| Prec@10            | Value   | Distinct Rate |  Unique items |
|--------------------|---------|---------------|---------------|
| Eval on  10 user   | 0.18667 |  0.31000      |    31         |
| Eval on  30 user   | 0.18000 |  0.14667      |    44         |
| Eval on 100 user   | 0.23453 |  0.08400      |    84         |


# Findings

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
  
- Summary compared to non-KG RecSys: **Big improvement in terms of Prec@k and distinct rate**

# Pros
- Able to incorporate Knowledge Graph as another source of information
- Able to infer why a user is given such suggestions (based on path scores)
- Able to adjust between 'exploration and optimization' by applying result pooling (the model doesn't require to be re-trained)
- During training, the model converge really fast (< 10 epochs)

# Cons
- No original implementation usable
- Relies heavily upon paths, and those paths are 'handcrafted' by using the knowledge-graph and also sampled from hundreds of million possible paths.
- Huge possible sampling bias introduced from preprocessing step and path generation step.
- The model remember the user, the model need to be re-trained for every new user and  item addition.
- Require relatively slow preprocessing
- Super slow train and prediction time
- Loss function and metric used in training is not Prec@K, instead it uses accuracy.

# Experiment notes
- At the cost of slightly different implementation, it's easier to implement using high-level libraries such as Keras instead of using original version.
- path generation & predict time : about 4k path / second
- different sampling method and sampling parameter has insignificant effect
- Using more items as path generation 'seed' (for predicting suggestion), should lead to more personalized suggestions. (i.e. consider the suggestion by using more user history)
- By pooling path prediction-score for the same items, the model should be able to give a better suggestion since it considers multiple reasons instead of just a single reason.


# Author
- Jessin Donnyson - jessinra@gmail.com

# Contributors
- Michael Julio - michael.julio@gdplabs.id
- Fallon Candra - fallon.candra@gdplabs.id