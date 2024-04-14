import Levenshtein
import jieba
import utils
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

def match(query, ipc, model, topn=20, epochs=50):
    normal_match_result = normal_match(query, ipc)
    if len(normal_match_result) == 0:
        return [ipc_id[0] for ipc_id in word2vec_similarity_match(query, ipc, model)]
    elif len(normal_match_result) < topn:
        word2vec_similarity_match_result = word2vec_similarity_match(query, ipc, model)[: topn - len(normal_match_result)]
        mixed_result = normal_match_result.extend(word2vec_similarity_match_result)
        return [ipc_id[0] for ipc_id in mixed_result]
    else:
        return [ipc_id[0] for ipc_id in normal_match[: 20]]

def normal_match(word, ipc):
    assert type(ipc) == utils.ipc
    word = list(jieba.cut(word, cut_all=True))
    
    score = []
    target_list = [[target_id, ipc.get_all_decs(target_id)] for target_id in ipc.leaf_node]
    for target in target_list:
        tmp_score = 0
        for c in word:
            if c in target[1]:
                tmp_score += 1
        score.append([target[0], tmp_score])
    
    return list(filter(lambda x:x[1] > 1, sorted(score, key=lambda x:x[1], reverse=True)))

def levenshtein_distance_match(word, ipc):
    assert type(ipc) == utils.ipc
    target_list = [[target_id, list(jieba.cut(ipc.get_all_decs(target_id), cut_all=False))] for target_id in ipc.leaf_node]
    
    score = []
    for target in tqdm(target_list, desc="levenshtein_distance_match"):
        tmp_score = []
        for target_piece in target[1]:
            tmp_score.append(Levenshtein.distance(word, target_piece))
        score.append([target[0], sum(tmp_score) / len(tmp_score)])
    
    return sorted(score, key=lambda x:x[1], reverse=False)

def word2vec_similarity_match(word, ipc, model, epochs=50):
    assert type(model) == str
    assert type(ipc) == utils.ipc
    # more_sentences = [list(jieba.cut(word, cut_all=False))]
    more_sentences = [[word]]
    tmp_model = Word2Vec.load(model)
    tmp_model.build_vocab(corpus_iterable=more_sentences, update=True)
    tmp_model.train(corpus_iterable=more_sentences, total_examples=tmp_model.corpus_count, epochs=epochs)
    # return tmp_model.wv.most_similar(positive=word, topn=10)

    score = []
    for leaf_node in tqdm(ipc.leaf_node, desc="word2vec_similarity_match"):
        max_cosine_similarity = 0
        target_cut_words = list(jieba.cut(ipc.get_all_decs(leaf_node)))
        for cut_word in target_cut_words:
            cosine = utils.cosine(tmp_model.wv[word], tmp_model.wv[cut_word])
            if max_cosine_similarity < cosine:
                max_cosine_similarity = cosine
        score.append([leaf_node, max_cosine_similarity])

    return sorted(score, key=lambda x:x[1], reverse=True)

def cosine_similarity_match(word, ipc, model, epochs=50):
    assert type(model) == str
    assert type(ipc) == utils.ipc
    more_sentences = [list(jieba.cut(word, cut_all=False))]
    tmp_model = Word2Vec.load(model)
    tmp_model.build_vocab(corpus_iterable=more_sentences, update=True)
    tmp_model.train(corpus_iterable=more_sentences, total_examples=tmp_model.corpus_count, epochs=epochs)

    query_embedding = np.mean([tmp_model.wv[cut_word] for cut_word in more_sentences[0]], axis=0)
    
    score = []
    for leaf_node in tqdm(ipc.leaf_node, desc="matching"):
        target_desc = ipc.get_all_decs(leaf_node)
        target_embedding = np.mean([tmp_model.wv[cut_target] for cut_target in jieba.cut(target_desc, cut_all=False)], axis=0)
        score.append([leaf_node, utils.cosine(query_embedding, target_embedding)])
    
    return sorted(score, key=lambda x:x[1], reverse=True)
