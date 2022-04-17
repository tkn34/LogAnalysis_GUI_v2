
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import re
import streamlit as st
from collections import defaultdict, Counter


class DataPreprocessing:
    def __init__(self, x):
        self.x = x
        # Compiling for optimization
        self.re_sub_1 = re.compile(r"(:(?=\s))|((?<=\s):)")
        self.re_sub_2 = re.compile(r"(\d+\.)+\d+")
        self.re_sub_3 = re.compile(r"\d{2}:\d{2}:\d{2}")
        self.re_sub_4 = re.compile(r"Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep")
        self.re_sub_5 = re.compile(r":?(\w+:)+")
        self.re_sub_6 = re.compile(r"\.|\(|\)|\<|\>|\/|\-|\=|\[|\]")
        self.p = re.compile(r"[^(A-Za-z)]")

    def __call__(self):
        x_after = self._remove_parameters(self.x)
        return x_after

    def _remove_parameters(self, logs):
        log_after_list = []
        for log in logs:
            log_tmp = log
            # Removing parameters with Regex
            log = re.sub(self.re_sub_1, "", log)
            log = re.sub(self.re_sub_2, "", log)
            log = re.sub(self.re_sub_3, "", log)
            log = re.sub(self.re_sub_4, "", log)
            log = re.sub(self.re_sub_5, "", log)
            log = re.sub(self.re_sub_6, " ", log)
            L = log.split()
            # Filtering strings that have non-letter tokens
            new_log = [k for k in L if not self.p.search(k)]
            log = " ".join(new_log)
            log_after_list.append(log)
        return log_after_list




class FeatureExtraction:
    def __init__(self, x, mode="train", fe_type="TF-ILF"):
        self.outputdir = "./model/"
        self.x = x
        self.mode = mode
        self.fe_type = fe_type # "tfilf" or "tfidf"
    
    def __call__(self):
        # Build Vocablary
        self.vocabulary = self._create_vocab()
        self._save_vocab()
        #print("Build Vocablary : Done")
        # Crete Vector
        self.vector = self._create_vector()
        #print("Crete Vector    : Done")
        #print("Create Feature  : Done")
        # Create Feature
        self.feature, self.ixf_dict = self._create_feature()
        self._save_feature()
        # ラベル付与
        return self.feature, self.vocabulary
        
    def _create_vocab(self):
        if self.mode == "train":
            vocabulary = {}
            for line in self.x:
                token_list = line.strip().split()
                for token in token_list:
                    if token not in vocabulary:
                        vocabulary[token] = len(vocabulary)
        else:
            with open(self.outputdir + "vocablary.pickle", "rb") as f:
                vocabulary = pickle.load(f)
        return vocabulary
        
    def _save_vocab(self):
        if self.mode == "train":
            with open(self.outputdir + "vocablary.pickle", "wb") as f:
                pickle.dump(self.vocabulary, f)
        
    def _create_vector(self):
        result = []
        for line in self.x:
            temp = []
            token_list = line.strip().split()
            if token_list:
                for token in token_list:
                    if token not in self.vocabulary:
                        continue
                    else:
                        temp.append(self.vocabulary[token])
            result.append(temp)
        return np.array(result)        

    def _create_feature(self):
        if self.fe_type == "TF-ILF":
            feature, ilf_dict = self._create_feature_tf_ilf()
        else:
            feature, ilf_dict = self._create_feature_tf_idf()
        return feature, ilf_dict

    def _create_feature_tf_ilf(self):
        feature_vectors = []
        # Calculate lf
        token_index_ilf_dict = defaultdict(set)
        for line in self.vector:
            for location, token in enumerate(line):
                token_index_ilf_dict[token].add(location)
        # Calculate ilf
        ilf_dict = {}
        max_length = len(max(self.vector, key=len))
        for token in token_index_ilf_dict:
            ilf_dict[token] = np.log(float(max_length) / float(len(token_index_ilf_dict[token]) + 0.01))
        # Create feature
        tfinvf = []
        for line in self.vector:
            cur_tfinvf = np.zeros(len(self.vocabulary))
            count_dict = Counter(line)
            for token_index in line:
                cur_tfinvf[token_index] = (float(count_dict[token_index]) * ilf_dict[token_index])
            tfinvf.append(cur_tfinvf)
        tfinvf = np.array(tfinvf)
        feature_vectors.append(tfinvf)
        feature = np.hstack(feature_vectors)
        feature = pd.DataFrame(feature)
        return feature, ilf_dict

    def _create_feature_tf_idf(self):
        feature_vectors = []
        # Calculate lf
        token_index_idf_dict = defaultdict(set)
        for line in self.vector:
            for location, token in enumerate(line):
                token_index_idf_dict[token].add(location)
        # Calculate idf
        idf_dict = {}
        total_log_num = len(self.vector)
        for token in token_index_idf_dict:
            idf_dict[token] = np.log(float(total_log_num) / float(len(token_index_idf_dict[token]) + 0.01))
        # Create feature    
        tfinvf = []
        for line in self.vector:
            cur_tfinvf = np.zeros(len(self.vocabulary))
            count_dict = Counter(line)
            for token_index in line:
                cur_tfinvf[token_index] = (float(count_dict[token_index]) * idf_dict[token_index])
            tfinvf.append(cur_tfinvf)
        tfinvf = np.array(tfinvf)
        feature_vectors.append(tfinvf)
        feature = np.hstack(feature_vectors)
        feature = pd.DataFrame(feature)
        return feature, idf_dict

    def _save_feature(self):
        if self.mode == "train":
            if self.fe_type == "tfilf":
                with open(self.outputdir + "tf_ilf.pickle", "wb") as f:
                    pickle.dump(self.ixf_dict, f)
            else:
                with open(self.outputdir + "tf_idf.pickle", "wb") as f:
                    pickle.dump(self.ixf_dict, f)


class RelatedInfo:
    def __init__(self, data_feature, feedback, th):
        self.data_feature = data_feature
        self.analysis_feature = self.data_feature[self.data_feature["label"]!=-1]
        self.anomaly_feature = self.data_feature[self.data_feature["label"]==-1]
        self.anomaly_feature["title"] = feedback["title"].values
        self.th = th
    
    def __call__(self):
        all_proba_list = []
        for row_feature in self.analysis_feature.values:
            proba_dict = {}
            analysis_row = row_feature[:-1]
            for anomaly_row in self.anomaly_feature.values:
                category = anomaly_row[-1]
                anomaly_row = anomaly_row[:-2]
                anom_proba = np.dot(analysis_row, anomaly_row) / (np.linalg.norm(analysis_row) * np.linalg.norm(anomaly_row))
                proba_dict[category] = anom_proba * 100
            max_kv = list(max(proba_dict.items(), key=lambda x: x[1]))
            if max_kv[1] >= self.th:
                max_kv[1] = str('{:.2f}'.format(max_kv[1]))
                max_kv.append(1)
            else:
                max_kv[0] = "-"
                max_kv[1] = "-"
                max_kv.append(0)
            all_proba_list.append(max_kv)
        all_proba = pd.DataFrame(all_proba_list, columns=["関連する過去の障害","関連度(%)", "正常(0)/異常(1)"])
        return all_proba[["正常(0)/異常(1)", "関連する過去の障害","関連度(%)"]]
    