#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


from tqdm import tqdm
import pickle


# # Read data

# In[2]:


file_ratings_re = open("ratings_re.csv").readlines()
file_triples_idx = open("triples_idx.txt").readlines()

file_moviesIdx = open("moviesIdx.txt").readlines() 
file_types = open("types.txt").readlines() 
file_entities = open("entities.txt").readlines()
file_relations = open("relations.txt").readlines()


# In[3]:


# create entity id -> name mapping
entity_id_to_name = {}
for line in file_moviesIdx:
    movie_title, entity_id = line.strip().split()
    entity_id_to_name[entity_id] = movie_title
    
for line in file_entities:
    entity_name, entity_id = line.strip().split()
    entity_id_to_name[entity_id] = entity_name


# In[4]:


# create movie title -> entity type mapping and list of entity for each type

movie_title_to_entity_type = {}
entity_list_with_type = {}

for line in file_types:
    
    # movie title -> entity type
    entity, entity_type = line.strip().split('\t')
    movie_title_to_entity_type[entity] = entity_type
    
    # entity for each type
    if entity_type not in entity_list_with_type:
        entity_list_with_type[entity_type] = []
    
    entity_list_with_type[entity_type].append(entity)


# In[5]:


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


# In[7]:


user_entity_id_padding = 500000
relation_rated_good_by_id = '200026'
relation_given_good_rating_id = '200027'
good_movie_rating_threshold = 4


# In[ ]:


def create_kg_path():

    ### Create entity_id path in knowledge graph
    kg_path = {}
    for line in file_triples_idx:

        head, relation, tail = line.strip().split()
        if head not in kg_path:
            kg_path[head] = {}

        if relation not in kg_path[head]:
            kg_path[head][relation] = []

        kg_path[head][relation].append(tail)
        
    ### Insert user rated relation into knowledge graph (collab filtering effect)
    for line in tqdm(file_ratings_re):
        user_id, movie_id, rating = line.split(',')[:3]
        user_entity_id = str(user_entity_id_padding + int(user_id))

        # skip if movie is bad
        if float(rating) < good_movie_rating_threshold :
            continue

        # Add given-good-rating for each user
        if user_entity_id not in kg_path:
            kg_path[user_entity_id] = {}
            kg_path[user_entity_id][relation_given_good_rating_id] = []

        # Add rated-good-by for the movie
        if relation_rated_good_by_id not in kg_path[movie_id]:
            kg_path[movie_id][relation_rated_good_by_id] = []

        kg_path[user_entity_id][relation_given_good_rating_id].append(movie_id)
        kg_path[movie_id][relation_rated_good_by_id].append(user_entity_id)
        
    pickle.dump(kg_path, open("cache_kg_path", "wb"))
    return kg_path


# In[ ]:


def load_kg_path():
    try:
        kg_path = pickle.load(open("cache_kg_path", "rb"))
    except:
        kg_path = create_kg_path()
    finally:
        return kg_path


# In[ ]:


kg_path = load_kg_path()


# # Utility function

# In[8]:


from numpy.random import randint

def generate_path_from_entity(entity_id, depth, n_sample_for_each_relation=2):
    
    generated_paths = []
    
    for relation in kg_path[entity_id]:
        tails = kg_path[entity_id][relation]
        
        sample_idx = randint(0, len(tails), n_sample_for_each_relation)
        sampled_tails = [tails[x] for x in sample_idx]
        
        for sampled_tail in sampled_tails:
            
            # RECURSE until last layer - 1 is enough
            if depth > 2 :
                future_paths = generate_path_from_entity(sampled_tail, depth-1, n_sample_for_each_relation)
                generated_paths += ["{} {} {}".format(relation, sampled_tail, x) for x in future_paths]
                
            else:
                generated_paths.append("{} {}".format(relation, sampled_tail))
    
    return generated_paths


# In[9]:


def save_list_to_file(list_obj, filename):
    with open(filename, 'w') as writer:
        for item in list_obj:
            
            if "\n" in item:
                writer.write(item)
            else:
                writer.write("%s\n" % item)


# In[10]:


def is_user_rated_potential_movie(user_id, potential_id):
    return potential_id in kg_path[user_id][relation_given_good_rating_id]


# # Create path from user - movies

# In[11]:


from numpy.random import random_sample

def split_dataset_positive_negative(path_dataset, sampling_negative_ratio=0.15):
    
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
    
    positive_dataset = sorted(positive_dataset)
    negative_dataset = sorted(negative_dataset)
    
    return positive_dataset, negative_dataset


# In[ ]:


import pickle

n_sample_for_each_relation = 2
batch_count = 0
path_depth = 3

generated_paths = []
for line in tqdm(file_ratings_re):
    
    # Save as batch
    if len(generated_paths) > 5000000:
        
        generated_paths = list(dict.fromkeys(generated_paths))        
        positive_dataset, negative_dataset = split_dataset_positive_negative(generated_paths)
            
        save_list_to_file(positive_dataset, "positive_path_{}.txt".format(batch_count))
        save_list_to_file(negative_dataset, "negative_path_{}.txt".format(batch_count))
        print("saved batch {}".format(batch_count))
        
        batch_count += 1
        generated_paths = []
    
    start_user_id, start_movie_id, rating = line.split(',')[:3]
    start_user_entity_id = str(user_entity_id_padding + int(start_user_id))
    
    # skip if movie is bad
    if float(rating) < good_movie_rating_threshold :
        continue
        
    future_paths = generate_path_from_entity(start_movie_id, path_depth)
    generated_paths += ["{} {} {} {}".format(start_user_entity_id, relation_given_good_rating_id, start_movie_id, x) for x in future_paths]


# # Reasoning part (extra)
# 
# Reading path as a string

# In[ ]:


sample_paths = open("negative_path_1.txt", 'r').readlines()


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


reasoning_path[:5]


# # Prepare other vocabulary things

# ## All entities

# In[104]:


all_entity = [x.strip().split() for x in file_moviesIdx]
all_entity += [x.strip().split() for x in file_entities]
all_entity = ["\t".join(x) for x in all_entity]


# In[ ]:


save_list_to_file(all_entity, "all_entity_id.txt")


# ## All relation

# In[11]:


relation = sorted(relation_id_to_name.items()) # sorted by key, return a list of tuples
all_relation = ["{}\t{}".format(x[1], x[0]) for x in relation] # unpack a list of pairs into two tuples

all_relation += [
    "#UNK_RELATION\t200028",
    "#PAD_TOKEN\t200029",
    "#END_RELATION\t200030",
]


# In[13]:


save_list_to_file(all_relation, "all_relation_id.txt")


# ## All type

# In[ ]:


# All entity type (including movie alr)
all_types = [x.strip() for x in file_types]

# All user
n_users = int(file_ratings_re[-1].split(',')[0])
all_types += ["u{}\tUser".format(i) for i in range(1, n_users + 1)]


# In[ ]:


save_list_to_file(all_types, "entity_to_type.txt")


# ## Type to id

# In[7]:


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


# In[8]:


save_list_to_file(entity_type_with_id, "entity_type_id.txt")


# # Format data for own dataset

# In[ ]:


def get_type_from_entity_id(entity_id):
    
    # user
    if int(entity_id) > user_entity_id_padding:
        return "User"
    elif entity_id_to_name[entity_id] in movie_title_to_entity_type:
        return movie_title_to_entity_type[entity_id_to_name[entity_id]]
    else:
        return "#UNK_ENTITY_TYPE"


# In[ ]:


import pickle

END_RELATION = 200030
BATCH_COUNT = 72

def create_own_format():
    
    for p in ['positive', 'negative']:
        for i in tqdm(range(0, BATCH_COUNT)):

            sample_paths = open("{}_path_{}.txt".format(p, i), 'r').readlines()

            dataset = []
            for line in sample_paths:
                e1, r1, e2, r2, e3, r3, e4, label = line.strip().split()

                t1 = entity_type_to_id[get_type_from_entity_id(e1)]
                t2 = entity_type_to_id[get_type_from_entity_id(e2)]
                t3 = entity_type_to_id[get_type_from_entity_id(e3)]
                t4 = entity_type_to_id[get_type_from_entity_id(e4)]

                r4 = END_RELATION

                output = [
                    "{} {} {}".format(e1, t1, r1),
                    "{} {} {}".format(e2, t2, r2),
                    "{} {} {}".format(e3, t3, r3),
                    "{} {} {}".format(e4, t4, r4),
                    label
                ]

                output = "#".join(output)
                dataset.append(output)
        
            save_list_to_file(dataset, "new_{}_path_{}.txt".format(p, i))


# In[ ]:


create_own_format()


# # Delete temporary files

# In[12]:


import os

for p in ['positive', 'negative']:
    for i in range(0, BATCH_COUNT):
        try:
            os.remove("{}_path_{}.txt".format(p, i))
        except:
            pass
