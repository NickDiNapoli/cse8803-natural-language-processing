import numpy as np

class BagofWords_Test():
	def __init__(self):

		# Sample data: 
		self.cleaned_text = ['year old man typewriter destroyed angry cop internet got new one',
							 'identify united states leaders',
							 'index economic activity declined march',
							 'best news bloopers control omg',
							 'pictures everyone loves spilling tea understand',
							 ' ']
		

		# Sample outputs for misc helper functions:
		self.data_split = [['year', 'old', 'man', 'typewriter', 'destroyed', 'angry', 'cop',
		 'internet', 'got', 'new', 'one'], ['identify', 'united', 'states', 'leaders'], 
		 ['index', 'economic', 'activity', 'declined', 'march'], ['best', 'news', 'bloopers', 
		 'control', 'omg'], ['pictures', 'everyone', 'loves', 'spilling', 'tea', 'understand'], []]

		self.flattened_list = np.array(['year', 'old', 'man', 'typewriter', 'destroyed', 'angry', 'cop',
									   'internet', 'got', 'new', 'one', 'identify', 'united', 'states',
									   'leaders', 'index', 'economic', 'activity', 'declined', 'march',
									   'best', 'news', 'bloopers', 'control', 'omg', 'pictures',
									   'everyone', 'loves', 'spilling', 'tea', 'understand'], dtype='<U10')

		self.fitted_categories = np.array(['activity', 'angry', 'best', 'bloopers', 'control', 'cop',
										'declined', 'destroyed', 'economic', 'everyone', 'got', 'identify',
										'index', 'internet', 'leaders', 'loves', 'man', 'march', 'new',
										'news', 'old', 'omg', 'one', 'pictures', 'spilling', 'states',
										'tea', 'typewriter', 'understand', 'united', 'year'], dtype='<U10')

		self.vocab_size = 31

		self.encode_0 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
							       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
							        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])

		self.cleaned_text_bow = np.array([[0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
									        1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.],
									       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
									        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
									       [1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
									        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
									       [0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
									        0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
									       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
									        0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0.],
									       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
									        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])




