from util import *

# Add your import statements here
import nltk.data



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		updated_text = text.replace("?", ".")
		updated_text_1 = updated_text.replace("!", ".")
		segmentedText = updated_text_1.split(".")[:-1]

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText = nltk.data.load('tokenizers/punkt/english.pickle').tokenize(text.strip())
		
		return segmentedText