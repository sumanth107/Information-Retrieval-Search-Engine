from util import *
import math
import numpy as np


# Add your import statements here


class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        precision = -1

        # Fill in code here
        cnt = 0
        for i in range(k):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                cnt += 1
        precision = cnt / k

        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        meanPrecision = -1

        # Fill in code here
        precision_sum = 0
        for i in range(len(query_ids)):
            true_doc_IDs = [int(d["id"]) for d in qrels if d["query_num"] == str(query_ids[i])]
            precision_sum += self.queryPrecision(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)
        meanPrecision = precision_sum / len(query_ids)

        return meanPrecision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        recall = -1

        # Fill in code here
        cnt = 0
        for i in range(k):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                cnt += 1
        recall = cnt / len(true_doc_IDs)

        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        meanRecall = -1

        # Fill in code here
        recall_sum = 0
        for i in range(len(query_ids)):
            true_doc_IDs = [int(d["id"]) for d in qrels if d["query_num"] == str(query_ids[i])]
            recall_sum += self.queryRecall(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)
        meanRecall = recall_sum / len(query_ids)

        return meanRecall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        fscore = -1

        # Fill in code here
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        if precision == 0 or recall == 0:
            fscore = 0
        else:
            fscore = (2 * precision * recall) / (precision + recall)

        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1

        # Fill in code here
        fscore_sum = 0
        for i in range(len(query_ids)):
            true_doc_IDs = [int(d["id"]) for d in qrels if d["query_num"] == str(query_ids[i])]
            fscore_sum += self.queryFscore(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)
        meanFscore = fscore_sum / len(query_ids)

        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        nDCG = -1

        # Fill in code here
        DCG = 0
        ideal = []
        for i in range(k):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                DCG += true_doc_IDs[query_doc_IDs_ordered[i]] / math.log2(i + 2)
        for i in range(len(query_doc_IDs_ordered)):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                ideal.append(true_doc_IDs[query_doc_IDs_ordered[i]])
        ideal.sort(reverse=True)
        ideal = ideal[:k]

        IDCG = 0
        for i in range(len(ideal)):
            IDCG += ideal[i] / math.log2(i + 2)
        if IDCG == 0:
            nDCG = 0
        else:
            nDCG = DCG / IDCG
        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = -1

        # Fill in code here
        nDCG_sum = 0
        cnt = 0
        for q in query_ids:
            true_doc_IDs = {}
            for d in qrels:
                if d["query_num"] == str(q):
                    true_doc_IDs[int(d["id"])] = 5-int(d["position"])
            nDCG_sum += self.queryNDCG(doc_IDs_ordered[cnt], q, true_doc_IDs, k)
            cnt += 1
        meanNDCG = nDCG_sum / len(query_ids)

        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        avgPrecision = -1

        # Fill in code here
        cnt = 0
        p = 0
        for i in range(k):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                cnt += 1
                p += cnt / (i + 1)
        if cnt == 0:
            avgPrecision = 0
        else:
            avgPrecision = p / cnt

        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        # Fill in code here
        AveragePrecision_sum = 0
        for i in range(len(query_ids)):
            true_doc_IDs = [int(d["id"]) for d in q_rels if d["query_num"] == str(query_ids[i])]
            AveragePrecision_sum += self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)
        meanAveragePrecision = AveragePrecision_sum / len(query_ids)

        return meanAveragePrecision
