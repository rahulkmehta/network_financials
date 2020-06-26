############################################################################
# File Name: downloader.py                                                 #
#                                                                          #
# Developer: Rahul Mehta                                                   #
#                                                                          #
# Designer: Debi Prasad Sahoo, Anshul Prakash Deshkar, Rahul Mehta         #
#                                                                          #
# (c)2016-2020 Copyright Protected,NetworkFinancials Inc.,San Jose(CA),USA #
#                                                                          #
############################################################################

#[IMPORTS]
import pickle

def loadpickle():
	with open ('pickles/randomlysampledstores.pickle', 'rb') as f:
		sampledstores = pickle.load(f)
		return sampledstores

#[DRIVER]
def test_main():
	sampledstores = loadpickle()
	print (sampledstores)

test_main()