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
import requests
import urllib

API_KEY = 'AIzaSyCu96-phtTlLZegwMRBaWMGkqMXcVXJtO4'

def loadpickle():
	with open ('pickles/randomlysampledstores.pickle', 'rb') as f:
		sampledstores = pickle.load(f)
		return sampledstores

def garnercoordinates(sampledstores):
	dict_ = {'2019': {}, '2018': {}, '2017': {}}
	for k,v in sampledstores.items():
		for tup in v:
			try: 
				constructed_string = tup[0]
				constructed_string = constructed_string.replace(" ", "+")
				constructed_string = constructed_string + ',+' + tup[1]
				response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=' + constructed_string + '&key='+ API_KEY)
				resp_json_payload = response.json()
				dict_[k][tup] = resp_json_payload['results'][0]['geometry']['location']
			except:
				print (tup)
				print (constructed_string)
	with open('pickles/storeswithcoordinates.pickle', 'wb') as handle:
		pickle.dump(dict_, handle, protocol = pickle.HIGHEST_PROTOCOL)

def verify_pickled():
    with open('pickles/storeswithcoordinates.pickle', 'rb') as f:
        sampledstores = pickle.load(f)
        return sampledstores

def reduce_download_load(dict_):
	bystate = {'2019' : {}, '2018': {}, '2017': {}}
	for k,v in dict_.items():
 		for item in v:
 			if item[1] in bystate[k]:
 				bystate[k][item[1]] += 1
 			else:
 				bystate[k][item[1]] = 1
	d = {'2019' : {}, '2018': {}, '2017': {}}
	for k,v in bystate.items():
		for key, value in v.items():
			if value > 10:
				d[k][key] = value
	return d

def compress(coords, reduced):
	final = {'2019' : {}, '2018': {}, '2017': {}}
	for k,v in coords.items():
		for key, value in v.items():
			if key[1] in reduced[k].keys():
				final[k][key] = value
	with open('pickles/finalwithreducedload.pickle', 'wb') as handle:
		pickle.dump(final, handle, protocol = pickle.HIGHEST_PROTOCOL)

def whatdownloadsneeded():
	with open('pickles/whatstatesneedtobedownloaded.pickle', 'rb') as f:
		n = pickle.load(f)
		print (n)

#[DRIVER]
def test_main():
	#sampledstores = loadpickle()
	#garnercoordinates(sampledstores)
    #dictcoords = verify_pickled()
    #reduced_load_on_machine = reduce_download_load(dictcoords)
    #final = compress(dictcoords, reduced_load_on_machine)
    #whatdownloadsneeded()
    with open('pickles/finalwithreducedload.pickle', 'rb') as f:
        n = pickle.load(f)
        print (n)
test_main()
