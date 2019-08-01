#!/usr/bin/env python
# coding: utf-8

# # <b><i> Preprocessing for KPRN </i></b>

# # > Preparation

# ## >> Import

# In[ ]:


from numpy.random import randint
from numpy.random import random_sample

from tqdm import tqdm
import pickle


# ## >> Read data

# In[ ]:


file_ratings_re = open("ratings_re.csv").readlines()
file_triples_idx = open("triples_idx.txt").readlines()
file_moviesIdx = open("moviesIdx.txt").readlines() 
file_types = open("types.txt").readlines() 
file_entities = open("entities.txt").readlines()
file_relations = open("relations.txt").readlines()


# ## >> Prepare vocab generation function

# In[ ]:


def _get_entity_to_name():

    entity_id_to_name = {}
    for line in file_moviesIdx:
        movie_title, entity_id = line.strip().split()
        entity_id_to_name[entity_id] = movie_title

    for line in file_entities:
        entity_name, entity_id = line.strip().split()
        entity_id_to_name[entity_id] = entity_name
        
    return entity_id_to_name


# In[ ]:


def _get_movie_title_to_entity_type():

    movie_title_to_entity_type = {}
    for line in file_types:
        
        entity, entity_type = line.strip().split('\t')
        movie_title_to_entity_type[entity] = entity_type
        
    return movie_title_to_entity_type


# In[ ]:


def _get_entity_list_with_type():

    entity_list_with_type = {}
    for line in file_types:
        
        entity, entity_type = line.strip().split('\t')
        if entity_type not in entity_list_with_type:
            entity_list_with_type[entity_type] = []
        entity_list_with_type[entity_type].append(entity)
        
    return entity_list_with_type


# In[ ]:


def _get_relation_to_name():

    # Create relation id to name mapping
    relation_id_to_name = {}
    for line in file_relations:
        relation_name, relation_id = line.strip().split()
        relation_id = int(relation_id)
        relation_id += 200000

        # last 2 relation : spouse and relative has no inverse
        if relation_id < 200023:
            relation_id_to_name[str(relation_id + 1)] = relation_name + "_inverse"

        relation_id_to_name[str(relation_id)] = relation_name
        
    return relation_id_to_name


# Let's run those script

# In[ ]:


movie_title_to_entity_type = _get_movie_title_to_entity_type()
entity_list_with_type = _get_entity_list_with_type()
entity_id_to_name = _get_entity_to_name()
relation_id_to_name = _get_relation_to_name()


# Some minor value

# ## >> Prepare KG path functions

# In[ ]:


USER_ENTITY_ID_PADDING = 500000
REL_ID_RATED_GOOD_BY = '200026'
REL_ID_GIVEN_GOOD_RATING = '200027'
RATING_THRESHOLD = 4


# In[ ]:


def _generate_kg_path_on_entity(kg_path):
    
    for line in tqdm(file_triples_idx):

        head, relation, tail = line.strip().split()
        if head not in kg_path:
            kg_path[head] = {}

        if relation not in kg_path[head]:
            kg_path[head][relation] = []

        kg_path[head][relation].append(tail)
        
    return kg_path

def _generate_kg_path_on_collaborative_filtering(kg_path):
    
    for line in tqdm(file_ratings_re):
        
        user_id, movie_id, rating = line.split(',')[:3]
        user_entity_id = str(USER_ENTITY_ID_PADDING + int(user_id))

        # skip if movie is bad
        if float(rating) < RATING_THRESHOLD :
            continue

        # Add given-good-rating list for each user
        if user_entity_id not in kg_path:
            kg_path[user_entity_id] = {}
            kg_path[user_entity_id][REL_ID_GIVEN_GOOD_RATING] = []

        # Add rated-good-by rating for the movie
        if REL_ID_RATED_GOOD_BY not in kg_path[movie_id]:
            kg_path[movie_id][REL_ID_RATED_GOOD_BY] = []

        kg_path[user_entity_id][REL_ID_GIVEN_GOOD_RATING].append(movie_id)
        kg_path[movie_id][REL_ID_RATED_GOOD_BY].append(user_entity_id)

    return kg_path


# In[ ]:


CACHE_KG_FILENAME = "cache_kg_path"

def create_kg_path():
    
    kg_path = dict()
    kg_path = _generate_kg_path_on_entity(kg_path)
    kg_path = _generate_kg_path_on_collaborative_filtering(kg_path)
    
    pickle.dump(kg_path, open(CACHE_KG_FILENAME, "wb"))
    return kg_path

def load_kg_path():
    
    try:
        kg_path = pickle.load(open(CACHE_KG_FILENAME, "rb"))
        
    except:
        kg_path = create_kg_path()
        
    finally:
        return kg_path


# In[ ]:


kg_path = load_kg_path()


# ## >> Some helper function

# In[ ]:


def save_list_to_file(list_obj, filename):
    with open(filename, 'w') as writer:
        for item in list_obj:
            if "\n" in item:
                writer.write(item)
            else:
                writer.write("%s\n" % item)


# In[ ]:


def is_user_rated_potential_movie(user_id, potential_id):
    return potential_id in kg_path[user_id][REL_ID_GIVEN_GOOD_RATING]


# ## >> Path generation function

# In[ ]:


def generate_path_from_entity(entity_id, depth, n_sample_for_each_relation=2):
    
    generated_paths = []
    for relation in kg_path[entity_id]:
        
        tails = kg_path[entity_id][relation]        
        sample_idx = randint(0, len(tails), n_sample_for_each_relation)
        sampled_tails = [tails[x] for x in sample_idx]
        
        for sampled_tail in sampled_tails:
            
            # Recurse until last layer - 1 is enough
            if depth > 2 :
                future_paths = generate_path_from_entity(sampled_tail, depth-1, n_sample_for_each_relation)
                generated_paths += ["{} {} {}".format(relation, sampled_tail, x) for x in future_paths]
            else:
                generated_paths.append("{} {}".format(relation, sampled_tail))
    
    return generated_paths


# In[ ]:


def _split_dataset_positive_negative(path_dataset, sampling_negative_ratio=0.15):
    
    positive_dataset = []
    negative_dataset = []
    
    for line in (path_dataset):
        split = line.split()
        user_id = split[0]
        potential_movie = split[-1]

        if is_user_rated_potential_movie(user_id, potential_movie) :
            positive_dataset.append(line.strip() + " 1")
        else:
            if random_sample() < sampling_negative_ratio:
                negative_dataset.append(line.strip() + " 0")
    
    return sorted(positive_dataset), sorted(negative_dataset)


# In[ ]:


def _save_batch(generated_paths, batch_num):
    generated_paths = list(dict.fromkeys(generated_paths))        
    positive_dataset, negative_dataset = _split_dataset_positive_negative(generated_paths)

    save_list_to_file(positive_dataset, "positive_path_{}.txt".format(batch_num))
    save_list_to_file(negative_dataset, "negative_path_{}.txt".format(batch_num))
    print("saved batch {}".format(batch_num))


# # > User-item path generations

# In[ ]:


batch_num = 0
generated_paths = []
for line in tqdm(file_ratings_re):
    
    # Checkpoint every 5m paths
    if len(generated_paths) > 5000000:        
        _save_batch(generated_paths, batch_num)
        batch_num += 1
        generated_paths = []
    
    start_user_id, start_movie_id, rating = line.split(',')[:3]
    start_user_entity_id = str(USER_ENTITY_ID_PADDING + int(start_user_id))
    
    # skip if movie is bad
    if float(rating) < RATING_THRESHOLD :
        continue
        
    future_paths = generate_path_from_entity(start_movie_id, depth=3)
    generated_paths += ["{} {} {} {}".format(start_user_entity_id, REL_ID_GIVEN_GOOD_RATING, start_movie_id, x) for x in future_paths]


# -----

# # > Save other vocabulary things required

# ### all_entity_id.txt

# In[ ]:


all_entity = [x.strip().split() for x in file_moviesIdx]
all_entity += [x.strip().split() for x in file_entities]
all_entity = ["\t".join(x) for x in all_entity]

save_list_to_file(all_entity, "all_entity_id.txt")


# ### all_relation_id.txt

# In[ ]:


relation = sorted(relation_id_to_name.items()) # sorted by key, return a list of tuples
all_relation = ["{}\t{}".format(x[1], x[0]) for x in relation] # unpack a list of pairs into two tuples

all_relation += [
    "#UNK_RELATION\t200028",
    "#PAD_TOKEN\t200029",
    "#END_RELATION\t200030",
]

save_list_to_file(all_relation, "all_relation_id.txt")


# ### entity_to_type.txt

# In[ ]:


# All entity type (including movie alr)
all_types = [x.strip() for x in file_types]

# All user
n_users = int(file_ratings_re[-1].split(',')[0])
all_types += ["u{}\tUser".format(i) for i in range(1, n_users + 1)]

save_list_to_file(all_types, "entity_to_type.txt")


# ### entity_type_id.txt

# In[ ]:


list_of_type = [
    "Category",
    "Company",
    "Country",
    "Genre",
    "Movie",
    "Person",
    "User",
    
    "#PAD_TOKEN",
    "#UNK_ENTITY_TYPE",
]

entity_type_to_id = {}
entity_type_with_id = []

for i in range(0, len(list_of_type)):
    entity_type = list_of_type[i]
    entity_type_to_id[entity_type] = i
    entity_type_with_id.append("{}\t{}".format(entity_type, i))
    
save_list_to_file(entity_type_with_id, "entity_type_id.txt")


# # > Format data for fit own-KPRN model

# In[ ]:


def get_type_from_entity_id(entity_id):
    
    if int(entity_id) > USER_ENTITY_ID_PADDING:
        return "User"
    
    elif entity_id_to_name[entity_id] in movie_title_to_entity_type:
        return movie_title_to_entity_type[entity_id_to_name[entity_id]]
    
    else:
        return "#UNK_ENTITY_TYPE"


# In[ ]:


import pickle

END_RELATION_ID = 200030
BATCH_COUNT = 72

def _reformat_paths(paths):

    result_paths = []
    for line in paths:
        e1, r1, e2, r2, e3, r3, e4, label = line.strip().split()

        t1 = entity_type_to_id[get_type_from_entity_id(e1)]
        t2 = entity_type_to_id[get_type_from_entity_id(e2)]
        t3 = entity_type_to_id[get_type_from_entity_id(e3)]
        t4 = entity_type_to_id[get_type_from_entity_id(e4)]

        r4 = END_RELATION_ID

        output = [
            "{} {} {}".format(e1, t1, r1),
            "{} {} {}".format(e2, t2, r2),
            "{} {} {}".format(e3, t3, r3),
            "{} {} {}".format(e4, t4, r4),
            label
        ]

        output = "#".join(output)
        result_paths.append(output)
        
    return result_paths


def create_own_format():
    for p in ['positive', 'negative']:
        for i in tqdm(range(0, BATCH_COUNT)):
            sample_paths = open("{}_path_{}.txt".format(p, i), 'r').readlines()
            formatted_paths = _reformat_paths(sample_paths)
            save_list_to_file(formatted_paths, "new_{}_path_{}.txt".format(p, i))
            


# In[ ]:


create_own_format()


# ## >> Delete temporary files

# In[ ]:


import os

for p in ['positive', 'negative']:
    for i in range(0, BATCH_COUNT):
        try:
            os.remove("{}_path_{}.txt".format(p, i))
        except:
            pass


# # > Reasoning part (extra, unrelated)
# 
# Reading paths as a string version

# In[ ]:


sample_paths_filename = "negative_path_0.txt"
sample_paths = open(sample_paths_filename, 'r').readlines()


# In[ ]:


def get_entity_name(entity_id):
    
    if entity_id in entity_id_to_name:
        return entity_id_to_name[entity_id]
    elif entity_id in relation_id_to_name:
        return relation_id_to_name[entity_id]
    else:
        return "user_{}".format(entity_id)


# In[ ]:


reasoning_path = []
for line in tqdm(sample_paths):
    string_line = " -> ".join([get_entity_name(x) for x in line.split()[:-1]])
    reasoning_path.append(string_line)


# In[ ]:


reasoning_path
