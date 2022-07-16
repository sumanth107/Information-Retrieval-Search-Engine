from util import *
import math
import numpy as np


# Add your import statements here


class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None

        # Fill in code here
        index = {}
        for i in range(len(docIDs)):
            index[docIDs[i]] = docs[i]

        self.index = index

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        # Fill in code here
        words = []
        doc_words = {}
        for i in self.index:
            doc_words[i] = []
            for s in self.index[i]:
                words.extend(s)
                doc_words[i].extend(s)
        words = list(set(words))
        # df of each word
        df = {}
        for w in words:
            df[w] = 0
            for i in self.index:
                if w in doc_words[i]:
                    df[w] += 1
        # idf of each word
        idf = {}
        for w in words:
            idf[w] = math.log10(len(self.index) / df[w])
        # tf-idf of each word in each doc
        tf_idf = {}
        for i in self.index:
            tf_idf[i] = {}
            for w in words:
                tf_idf[i][w] = 0
            for w in doc_words[i]:
                tf_idf[i][w] += idf[w]

        # tf-idf 2d list
        tf_idf_2d = []
        for i in self.index:
            tf_idf_2d.append([])
            for w in words:
                tf_idf_2d[-1].append(tf_idf[i][w])

        # LSA
        U, S, V = np.linalg.svd(np.array(tf_idf_2d).T)
        M = np.matmul(np.matmul(U[:, :301], np.diag(S[:301])), V[:301, :])
        tf_idf_temp = M.T

        # tf-idf new dictionary
        tf_idf_new = {}
        for i in self.index:
            tf_idf_new[i] = {}
            for j in range(len(words)):
                tf_idf_new[i][words[j]] = tf_idf_temp[i - 1][j]
        # -----------------------------------------------------------------------------------------------------------------------#
        # #VSM
        # # norms of each doc - VSM
        # norms = {}
        # for i in self.index:
        #     norms[i] = 0
        #     for w in words:
        #         norms[i] += tf_idf[i][w] ** 2
        #     norms[i] = math.sqrt(norms[i])
        #
        # # cosine similarity of each query with each doc - VSM
        # q_idx = 0
        # q_vec = {}
        # cos_sim = {}
        # for q in queries:
        #     q_vec[q_idx] = {}
        #     cos_sim[q_idx] = {}
        #     for w in words:
        #         q_vec[q_idx][w] = 0
        #     for s in q:
        #         for w in s:
        #             if w in words:
        #                 q_vec[q_idx][w] += idf[w]
        #     norm_q = 0
        #     for w in words:
        #         norm_q += q_vec[q_idx][w] ** 2
        #     norm_q = math.sqrt(norm_q)
        #     for i in self.index:
        #         cos_sim[q_idx][i] = 0
        #         for w in words:
        #             cos_sim[q_idx][i] += tf_idf[i][w] * q_vec[q_idx][w]
        #         if norms[i] == 0 or norm_q == 0:
        #             cos_sim[q_idx][i] = 0
        #         else:
        #             cos_sim[q_idx][i] /= norms[i] * norm_q
        #     q_idx += 1
        # sort each doc according to cosine similarity with each query
        # for q in range(len(queries)):
        #     doc_IDs_ordered.append([])
        #     for i in self.index:
        #         doc_IDs_ordered[-1].append(i)
        #     doc_IDs_ordered[-1].sort(key=lambda x: cos_sim[q][x], reverse=True)
        # -----------------------------------------------------------------------------------------------------------------------#
        # #LSA
        # # norms of each doc - LSA
        # norms = {}
        # for i in self.index:
        #     norms[i] = 0
        #     for w in words:
        #         norms[i] += tf_idf_new[i][w] ** 2
        #     norms[i] = math.sqrt(norms[i])
        #
        # # cosine similarity of each query with each doc - LSA
        # q_idx = 0
        # q_vec = {}
        # cos_sim = {}
        # for q in queries:
        #     q_vec[q_idx] = {}
        #     cos_sim[q_idx] = {}
        #     for w in words:
        #         q_vec[q_idx][w] = 0
        #     for s in q:
        #         for w in s:
        #             if w in words:
        #                 q_vec[q_idx][w] += idf[w]
        #     norm_q = 0
        #     for w in words:
        #         norm_q += q_vec[q_idx][w] ** 2
        #     norm_q = math.sqrt(norm_q)
        #     for i in self.index:
        #         cos_sim[q_idx][i] = 0
        #         for w in words:
        #             cos_sim[q_idx][i] += tf_idf_new[i][w] * q_vec[q_idx][w]
        #         if norms[i] == 0 or norm_q == 0:
        #             cos_sim[q_idx][i] = 0
        #         else:
        #             cos_sim[q_idx][i] /= norms[i] * norm_q
        #     q_idx += 1
        # sort each doc according to cosine similarity with each query
        # for q in range(len(queries)):
        #     doc_IDs_ordered.append([])
        #     for i in self.index:
        #         doc_IDs_ordered[-1].append(i)
        #     doc_IDs_ordered[-1].sort(key=lambda x: cos_sim[q][x], reverse=True)
        # -----------------------------------------------------------------------------------------------------------------------#
        # BM25
        avgdl = sum(len(doc_words[i]) for i in self.index) / len(self.index)
        k = 1.2
        b = 0.75
        bm_idf = {}
        N = len(self.index)
        q_idx = 0
        for q in queries:
            bm_idf[q_idx] = {}
            for i in self.index:
                bm_idf[q_idx][i] = 0
                for s in q:
                    for w in s:
                        if w in set(doc_words[i]):
                            fq = doc_words[i].count(w)
                            bm_idf[q_idx][i] += idf[w]*(fq * (k + 1)) / (fq + k * (1 - b + b * len(doc_words[i]) / avgdl))
            q_idx += 1
        for q in range(len(queries)):
            doc_IDs_ordered.append([])
            for i in self.index:
                doc_IDs_ordered[-1].append(i)
            doc_IDs_ordered[-1].sort(key=lambda x: bm_idf[q][x], reverse=True)

        # -----------------------------------------------------------------------------------------------------------------------#


        return doc_IDs_ordered
