from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
from util import sc
from util import q_exp

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:

    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """
        # # Spell Checker
        # queries = [sc().querys(query) for query in queries]

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            # # Query Expansion ------------------------------------------------------------------------------------------#
            # stopwordRemovedQuery = q_exp().expand(stopwordRemovedQuery, 2)
            # # -----------------------------------------------------------------------------------------------------------#
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
                             [item["query"] for item in queries_json]
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for each query
        doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " +
                  str(k) + " : " + str(precision) + ", " + str(recall) +
                  ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +
                  str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot_BM25.png")

        plt.clf()
        plt.cla()

        p_vsm = [0.6488888888888888, 0.5466666666666666, 0.48592592592592615, 0.43444444444444447, 0.38577777777777794,
                 0.35851851851851846, 0.33396825396825425, 0.31666666666666665, 0.29975308641975346, 0.2835555555555557,
                 0.26747474747474764, 0.2544444444444444, 0.24102564102564117, 0.23174603174603198, 0.22222222222222235,
                 0.215, 0.20705882352941185, 0.19827160493827167, 0.19017543859649122, 0.18355555555555572,
                 0.17756613756613746, 0.17151515151515148, 0.16772946859903373, 0.16203703703703698,
                 0.15715555555555522, 0.15299145299145311, 0.14847736625514413, 0.14476190476190479,
                 0.14191570881226054, 0.1392592592592594, 0.13562724014336938, 0.13180555555555556, 0.12888888888888902,
                 0.1265359477124184, 0.12431746031746005, 0.12259259259259257, 0.11975975975975973, 0.11777777777777777,
                 0.1156695156695154, 0.11344444444444449, 0.11176151761517594, 0.1098412698412699, 0.10842377260981899,
                 0.10656565656565645, 0.10469135802469111, 0.10289855072463758, 0.1012765957446808, 0.1001851851851852,
                 0.0983219954648527, 0.09688888888888882, 0.09542483660130696, 0.09461538461538466, 0.09308176100628944,
                 0.0917695473251029, 0.0905858585858586, 0.08928571428571432, 0.08810916179337225, 0.08712643678160915,
                 0.08617702448210926, 0.08481481481481484, 0.08393442622950814, 0.0827956989247313, 0.08176366843033492,
                 0.08090277777777778, 0.07993162393162376, 0.07898989898989905, 0.07807628524046441,
                 0.07725490196078438, 0.07652173913043474, 0.07568253968253956, 0.07499217527386545,
                 0.07425925925925925, 0.0736073059360731, 0.07297297297297299, 0.07217777777777767, 0.07134502923976604,
                 0.07070707070707079, 0.0699715099715098, 0.06925457102672311, 0.06838888888888889, 0.06792866941015084,
                 0.06742547425474246, 0.06677376171352076, 0.06608465608465615, 0.06546405228758155, 0.0648062015503875,
                 0.06431673052362716, 0.06373737373737366, 0.06332084893882642, 0.06266666666666652,
                 0.06217338217338203, 0.06159420289855065, 0.06107526881720446, 0.06052009456264778,
                 0.060116959064327506, 0.059629629629629595, 0.059152348224513034, 0.05859410430839001,
                 0.058136924803591505, 0.057644444444444366]
        r_vsm = [0.10919181807099311, 0.1808846901723276, 0.228655709398868, 0.2645503767399963, 0.2883589899754627,
                 0.3241227385725446, 0.348107742542606, 0.3713727081774375, 0.3931739889591105, 0.40999451439276363,
                 0.4239633436753183, 0.435410140173839, 0.445473632237331, 0.46110342220045425, 0.4707121455758443,
                 0.4814665288106199, 0.49097112553360733, 0.4982570001528152, 0.50331536154451, 0.5099359014983832,
                 0.5164073541686189, 0.5210756118368766, 0.532157702585634, 0.5358320344266325, 0.5407723903669884,
                 0.5465365229205946, 0.5495183250690635, 0.5542634528659154, 0.5601964334655626, 0.5666824623366491,
                 0.5693676475218343, 0.5709972771514639, 0.5745507977049844, 0.5808470940012808, 0.5875987374195909,
                 0.5956296016171219, 0.5977428913774703, 0.6031562002907791, 0.6075688987034775, 0.6100139623669322,
                 0.614441411461048, 0.6169205940919547, 0.6222563805277411, 0.6259036468416741, 0.6296990613037553,
                 0.6327967044013985, 0.6375485562532504, 0.6447957018951328, 0.6457552978547288, 0.6478560590168935,
                 0.6504931583010515, 0.6563681703748465, 0.6580718740785503, 0.6599388774455536, 0.6630746799146894,
                 0.6656392141458903, 0.667443335950012, 0.6712684979836365, 0.6742688679840063, 0.6751577568728953,
                 0.6787693674845058, 0.6793804785956169, 0.6822161860979912, 0.6847924388213027, 0.6865320572275877,
                 0.6879732629354602, 0.6897325221947194, 0.6921377021998993, 0.694942305004502, 0.6962780125068763,
                 0.6993478377433682, 0.7019700599655904, 0.7041135276090578, 0.7060280981510794, 0.7074108142004623,
                 0.7084901792798273, 0.7102981714825563, 0.7116611344455193, 0.7130400518244364, 0.7130400518244364,
                 0.7160083057926904, 0.7186564539408385, 0.7206194169038015, 0.7211002486127033, 0.7223559805351017,
                 0.7232871974663188, 0.7259532143604059, 0.7278387362459278, 0.7305190584779743, 0.7311539791128949,
                 0.7334892364481522, 0.7349707179296338, 0.7361053980643139, 0.7365583895173052, 0.7389404557941083,
                 0.7405009496212688, 0.7419694356511582, 0.7427101763918991, 0.7441352204836098, 0.7451405114889009]
        precisions1, recalls1 = [], []
        for k in range(1, 101):
            precision1 = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions1.append(precision1)
            recall1 = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls1.append(recall1)
        plt.plot(recalls1, precisions1, label="With BM25")
        plt.plot(r_vsm, p_vsm, label="Without BM25")
        plt.legend()
        plt.title("Precision vs Recall - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "PR_BM25.png")

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset',
                        default="/Users/Sumanth Nethi/Desktop/AcadSpace/SEM-8/NLP/Assignment 1/template_code_assignment_1/template_code_part1/cranfield/",
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder',
                        default="/Users/Sumanth Nethi/Desktop/AcadSpace/SEM-8/NLP/Assignment 1/template_code_assignment_1/template_code_part1/output/"
                        ,
                        help="Path to output folder")
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer', default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action="store_true",
                        help="Take custom query as input")

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    # Either handle query from user or evaluate on the complete dataset
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
