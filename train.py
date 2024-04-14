from gensim.models import Word2Vec
import utils
import jieba
import pickle

def main():
    ipc = utils.ipc("ipc_2019.txt")

    # sentences = [list(jieba.cut(sentence[1], cut_all=False)) for sentence in ipc.ipc_list]
    sentences = [list(jieba.cut(ipc.get_all_decs(leaf_node))) for leaf_node in ipc.leaf_node]
    id2ipc = [ipc_id[0] for ipc_id in ipc.ipc_list]

    # with open("sentences.pickle", "wb") as f1:
    #     pickle.dump(sentences, f1)
    # with open("id2ipc.pickle", "wb") as f2:
    #     pickle.dump(id2ipc, f2)
    # f1.close()
    # f2.close()

    model = Word2Vec(min_count=1, epochs=50)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=50)
    model.save("word2vec.model")
    print("done")

if __name__ == "__main__":
    main()