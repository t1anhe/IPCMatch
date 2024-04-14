from pymilvus import DataType, FieldSchema, CollectionSchema, Collection, connections
import random
import jieba
import numpy as np

class ipc:
    def __init__(self, ipc_file_path):
        assert ipc_file_path[-3:] == 'txt'
        self.ipc_file_path = ipc_file_path
        self.ipc_list = []
        self.ipc_dict = {}

        self.ipc_tree = {}
        self.leaf_node = []
        self.search_path = []

        with open(self.ipc_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(" ")
                line[1] = line[1].strip("*")
                self.ipc_list.append(line)

        for ipc in self.ipc_list:
            put_node(ipc[0], self.ipc_tree)

        for ipc in self.ipc_list:
            self.ipc_dict[ipc[0]] = ipc[1]

        self.traverse_leaf_nodes(self.ipc_tree)

    def get_ipc_list(self):
        return self.ipc_list

    def get_ipc_dict(self):
        # for ipc in self.ipc_list:
        #     self.ipc_dict[ipc[0]] = ipc[1]
        return self.ipc_dict
    
    def char_match(self, word):
        result = []       
        for ipc in self.ipc_list:
            if word in ipc[1]:
                result.append(ipc[1])
        random.shuffle(result)
        return result
    
    def traverse_leaf_nodes(self, tree):
        # visit_leaf_node(self.leaf_node, self.ipc_tree)
        # print(tree)
        for node in tree:
            # print(node)
            if tree[node]:
                # print(node)
                self.traverse_leaf_nodes(tree[node])
            else:
                # print(node)
                self.leaf_node.append(node)

    def __search(self, target, tree):
        for node in tree:
            if node == target:
                self.search_path.append(node)
                return self.search_path
            elif (node in target) and tree[node]:
                self.search_path.append(node)
                self.__search(target, tree[node])
                break

    def get_all_decs(self, ipc_id):
        assert len(self.search_path) == 0
        self.__search(ipc_id, self.ipc_tree)
        ipc_decs = "".join([self.ipc_dict[i] for i in self.search_path])
        self.search_path.clear()
        return ipc_decs
    
    def search2(self, target):
        assert len(self.search_path) == 0
        self.__search(target, self.ipc_tree)
        search_path = list(self.search_path)
        self.search_path.clear()
        return search_path
        
def put_node(node, tree):
    flag = 1
    for parent in tree:
        if parent in node:
            flag = 0
            put_node(node, tree[parent])
            break
    if flag:
        tree[node] = {}

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# class ipc_set:
#     def __init__(self, ipc_file_path):
#         self.ipc_file_path = ipc_file_path
#         self.ipc_list = []
#         self.ipc_dict = {}

#         with open(self.ipc_file_path, 'r', encoding="utf-8") as f:
#             for line in f:
#                 # self.ipc_list.append(line.strip())
#                 line = line.strip().split(" ")
#                 line[1] = line[1].strip("*")
#                 # print(line)
#                 self.ipc_list.append(line)

#     def get_ipc_list(self):
#         return self.ipc_list

#     def get_ipc_dict(self):
#         for ipc in self.ipc_list:
#             self.ipc_dict[ipc[0]] = ipc[1]
#         return self.ipc_dict
    
#     def char_match(self, word):
#         # assert type(word) == str
#         # word = jieba.cut(word, cut_all=False)
#         # print(word)
#         result = []
#         # for word_piece in word:
#         #     for ipc in self.ipc_list:
#         #         if word_piece in ipc[1]:
#         #             result.append(ipc[1])
        
#         for ipc in self.ipc_list:
#             if word in ipc[1]:
#                 result.append(ipc[1])
#         random.shuffle(result)
#         return result

# class ipc_tree():
#     def __init__(self, ipc_list):
#         self.ipc_list = ipc_list
#         self.ipc_dict = {}
#         self.ipc_tree = {}
#         self.leaf_node = []
#         self.search_path = []

#         for ipc in self.ipc_list:
#             put_node(ipc[0], self.ipc_tree)

#         for ipc in self.ipc_list:
#             self.ipc_dict[ipc[0]] = ipc[1]

#         self.traverse_leaf_nodes(self.ipc_tree)

#     def traverse_leaf_nodes(self, tree):
#         # visit_leaf_node(self.leaf_node, self.ipc_tree)
#         # print(tree)
#         for node in tree:
#             # print(node)
#             if tree[node]:
#                 # print(node)
#                 self.traverse_leaf_nodes(tree[node])
#             else:
#                 # print(node)
#                 self.leaf_node.append(node)

#     def search(self, target, tree):
#         for node in tree:
#             if node == target:
#                 self.search_path.append(node)
#                 return self.search_path
#             elif node in target and tree[node]:
#                 self.search_path.append(node)
#                 self.search(target, tree[node])
#                 break

#     def get_all_decs(self, ipc_id):
#         self.search(ipc_id, self.ipc_tree)
#         ipc_decs = "".join([self.ipc_dict[i] for i in self.search_path])
#         self.search_path.clear()
#         return ipc_decs

# def visit_leaf_node(leaf_node, tree):
#     # print(tree)
#     for node in tree.keys():
#         # print(node)
#         if tree[node]:
#             # print(tree[node])
#             visit_leaf_node(leaf_node, tree[node])
#         else:
#             leaf_node.append(node)
 
# class ipc_collection:
#     def __init__(self, field_name):
#         self.ipc_collection = create_collection(field_name)

#         index = {
#             "index_type": "IVF_FLAT",
#             "metric_type": "L2",
#             "params": {"nlist": 128},
#         }
#         self.ipc_collection.create_index("embeddings(gte)", index)

#         self.ipc_collection.load()

#     def insert(self, ipc_list):
#         self.ipc_collection.insert(ipc_list)
#         self.ipc_collection.flush()

#     def search(self, keyword):
#         search_params = {
#             "metric_type": "L2",
#             "params": {"nprobe": 10}
#         }
#         result = self.ipc_collection.search(keyword, "ipc_description", search_params, limit=5)
#         return result

# def connect_to_milvus():
#     connections.connect(host='localhost', port=19530)

# def create_collection(field_name):
#     fields = [
#         FieldSchema(name="ipc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=500),
#         FieldSchema(name="ipc_description", dtype=DataType.VARCHAR, max_length=500),
#         FieldSchema(name="embeddings(gte)", dtype=DataType.FLOAT_VECTOR, dim=8)
#     ]

#     schema = CollectionSchema(fields=fields)
#     ipc_collection = Collection(field_name, schema, consistency_level="Strong")
#     return ipc_collection
