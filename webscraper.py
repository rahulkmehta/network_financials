############################################################################
# File Name: webscraper.py                                                 #
#                                                                          #
# Developer: Rahul Mehta                                                   #
#                                                                          #
# Designer: Debi Prasad Sahoo, Anshul Prakash Deshkar, Rahul Mehta         #
#                                                                          #
# (c)2016-2020 Copyright Protected,NetworkFinancials Inc.,San Jose(CA),USA #
#                                                                          #
############################################################################

# [PACKAGE IMPORTS]
import csv
import requests
import json
import argparse
import traceback
import json
from bs4 import BeautifulSoup
from boxsdk import OAuth2, Client
import ast
import pickle
import random

# [BOX API CREDENTIALS]
auth = OAuth2(
    client_id = 'l1je8xajdp77h9f8eu2bs27w6a2c2kcf',
    client_secret = 'T1IFvFU4Pv6Dz6gDB8Lf8LujnpIxJsJr',
    access_token = 'CYf8dCFSNQ91UjAj1ATZ1lgbAQaRDv70',
)
client = Client(auth)

# [MANUAL INPUT OF DICT]
year_dict = {'2019': ['al', 'ar', 'az', 'co', 'fl', 'ga', 'ia', 'id', 'il', 'ks', 'la', 'mn', 'mt', 'nd', 'nj', 'nv', 'ny', 'oh', 'ok', 'pa', 'sc', 'tx', 'wa', 'wy'],
 '2018': ['ca', 'ct', 'de', 'in', 'ky', 'ma', 'md', 'me', 'mi', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nm', 'or', 'ri', 'sd', 'tn', 'ut', 'va', 'vt', 'wi', 'wv'],
 '2017': ['al', 'ar', 'az', 'co', 'de', 'fl', 'ga', 'ia', 'id', 'il', 'ks', 'la', 'md', 'mn', 'mt', 'nd', 'nj', 'nv', 'ny', 'oh', 'ok', 'pa', 'sc', 'wa', 'wi', 'wy']}

#[SCRAPE FROM NAIP DATABASE]
def absorbdataintoarray():
	dict_ = {}
	file_object = open("extra_data/countynames.txt", "r")
	lines = file_object.readlines()
	for line in lines:
		temp_arr = line.split()
		dict_[(temp_arr[0], temp_arr[1])] = [" ".join(temp_arr[2:])]
	return dict_

#[ZIPCODE SCRAPING]
def scrape_csv():
	zipcode_array = []
	with open ('extra_data/uszips.csv', newline = '') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			zipcode_array.append(row[0])
	return zipcode_array

#[SCRAPE ALL OF WALMART'S STORES]
def locate_stores():
	storedict = {}
	for i in range(2750, 8000):
		print (i)
		page = requests.get("https://www.walmart.com/store/" + str(i))
		soup = BeautifulSoup(page.content, 'html.parser')
		try: 
			mydivs = soup.find("div", {"class": "store-address"}).findChildren()
			storedict[i] = (mydivs[0].text, mydivs[3].text, mydivs[4].text, mydivs[5].text)
		except:
			print ()
	with open('stores2.txt', 'w') as file:
	 	file.write(json.dumps(storedict))

#[CONTINGENCY FUNCTION]
def scrape_stores():
    scraped_data = locate_stores(zip_code)
    if scraped_data:
        print ("Writing scraped data to %s_stores.csv"%(zip_code))
        with open('%s_stores.csv'%(zip_code),'wb') as csvfile:
            fieldnames = ["name","store_id","distance","address","zip_code","city","phone"]
            writer = csv.DictWriter(csvfile,fieldnames = fieldnames,quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for data in scraped_data:
                writer.writerow(data)

#[CROSS-REFERENCE COUNTIES WITH STORES' ADDRESS AND SAVE TO FILE]
def filter_stores(dictionary):
	def_list = []
	with open('extra_data/stores.txt') as json_file:
		data = json.load(json_file)
		for k,v in data.items():
			for key, value in dictionary.items():
				temp_city_holder = v[1].upper()
				temp_city_holder = temp_city_holder.replace("SAINT", "ST")
				if (temp_city_holder == value[0].upper() and v[2].upper() == key[0].upper()):
					def_list.append((str(v[0].upper()), key[0], key[1]))
	with open('filteredstorelist.pickle', 'wb') as f:
		pickle.dump(def_list, f)

#[SCRAPE DICTIONARY OF AVAILABLE DATA FROM NAIP]
def scrape_naip(c19, c18, c17):
	fsl = []
	state_breakdown = {'2019': [], '2018': [], '2017': []}
	finalbreakdown_withcountycrossref = {'2019': [], '2018': [], '2017': []}
	with open ('filteredstorelist.pickle', 'rb') as f:
		data = pickle.load(f)
		for item in data:
			if (item[1].lower() in year_dict['2019']):
				state_breakdown['2019'].append(item)
			if (item[1].lower() in year_dict['2018']):
				state_breakdown['2018'].append(item)
			if (item[1].lower() in year_dict['2017']):
				state_breakdown['2017'].append(item)

	for k,v in state_breakdown.items():
		for item in v:
			if k == '2019':
				if int(item[2]) in c19[item[1].lower()]:
					finalbreakdown_withcountycrossref['2019'].append((item[0], item[1], item[2]))
			elif k == '2018':
				if int(item[2]) in c18[item[1].lower()]:
					finalbreakdown_withcountycrossref['2018'].append((item[0], item[1], item[2]))
			else:
				if int(item[2]) in c17[item[1].lower()]:
					finalbreakdown_withcountycrossref['2017'].append((item[0], item[1], item[2]))
	return finalbreakdown_withcountycrossref

# [POPULATES COUNTY DICTIONARY]
def populate_county_dictionary():
	county2019 = {}
	for item in year_dict['2019']:
		county2019[item] = []

	county2018 = {}
	for item in year_dict['2018']:
		county2018[item] = []

	county2017 = {}
	for item in year_dict['2017']:
		county2017[item] = []

	#[2019]
	county2019['al'] = list(range(1, 134, 2))
	county2019['ar'] = list(range(1, 150, 2))
	county2019['az'] = list(range(1, 28, 2))
	county2019['az'].append(12)
	county2019['co'] = list(range(1, 128, 2))
	county2019['fl'] = list(range(1, 135, 2))
	county2019['ga'] = list(range(1, 322, 2))
	county2019['ia'] = list(range(1, 198, 2))
	county2019['id'] = list(range(1, 89, 2))
	county2019['il'] = list(range(1, 205, 2))
	county2019['ks'] = list(range(1, 211, 2))
	county2019['la'] = list(range(1, 129, 2))
	county2019['mn'] = list(range(1, 175, 2))
	county2019['mt'] = list(range(1, 99, 2))
	county2019['nd'] = list(range(1, 107, 2))
	county2019['nj'] = list(range(1, 42, 2))
	county2019['nv'] = list(range(1, 33, 2))
	county2019['oh'] = list(range(1, 175, 2))
	county2019['ok'] = list(range(1, 154, 2))
	county2019['pa'] = list(range(1, 135, 2))
	county2019['sc'] = list(range(1, 93, 2))
	county2019['tx'] = list(range(1, 511, 2))
	county2019['wa'] = list(range(1, 79, 2))
	county2019['wy'] = list(range(1, 47, 2))

	#[2018]	
	county2018['ca'] = list(range(1, 117, 2))
	county2018['ct'] = list(range(1, 17, 2))
	county2018['de'] = list(range(1, 6, 2))
	county2018['in'] = list(range(1, 186, 2))
	county2018['ky'] = list(range(1, 241, 2))
	county2018['ma'] = list(range(1, 29, 2))
	county2018['md'] = list(range(1, 49, 2))
	county2018['me'] = list(range(1, 33, 2))
	county2018['mi'] = list(range(1, 167, 2))
	county2018['mo'] = list(range(1, 231, 2))
	county2018['ms'] = list(range(1, 165, 2))
	county2018['mt'] = list(range(1, 27, 2))
	county2018['nc'] = list(range(1, 201, 2))
	county2018['nd'] = list(range(1, 107, 2))
	county2018['ne'] = list(range(1, 187, 2))
	county2018['nh'] = list(range(1, 21, 2))
	county2018['nm'] = list(range(1, 67, 2))
	county2018['or'] = list(range(1, 97, 2))
	county2018['ri'] = list(range(1, 11, 2))
	county2018['sd'] = list(range(1, 132, 2))
	county2018['tn'] = list(range(1,191, 2))
	county2018['ut'] = list(range(1, 59, 2))
	county2018['va'] = list(range(1, 201, 2))
	county2018['vt'] = list(range(1, 29, 2))
	county2018['wi'] = list(range(1, 145, 2))
	county2018['wv'] = list(range(1, 111, 2))

	#[2017]	
	county2017['al'] = list(range(1, 134, 2))
	county2017['ar'] = list(range(1, 150, 2))
	county2017['az'] = list(range(1, 28, 2))
	county2017['co'] = list(range(1, 126, 2))
	county2017['de'] = list(range(1, 6, 2))
	county2017['fl'] = list(range(1, 134, 2))
	county2017['ga'] = list(range(1, 318, 2))
	county2017['ia'] = list(range(1, 198, 2))
	county2017['id'] = list(range(1, 88, 2))
	county2017['il'] = list(range(1, 204, 2))
	county2017['ks'] = list(range(1, 210, 2))
	county2017['la'] = list(range(1, 128, 2))
	county2017['md'] = list(range(1, 48, 2))
	county2017['mn'] = list(range(1, 174, 2))
	county2017['mt'] = list(range(1, 110, 2))
	county2017['nd'] = list(range(1, 106, 2))
	county2017['nj'] = list(range(1, 42, 2))
	county2017['nv'] = list(range(1, 34, 2))
	county2017['ny'] = list(range(1, 124, 2))
	county2017['oh'] = list(range(1, 176, 2))
	county2017['ok'] = list(range(1, 154, 2))
	county2017['pa'] = list(range(1, 134, 2))
	county2017['sc'] = list(range(1, 92, 2))
	county2017['wa'] = list(range(1, 78, 2))
	county2017['wi'] = list(range(1, 142, 2))
	county2017['wy'] = list(range(1, 46, 2))
	return county2019, county2018, county2017

# [RANDOM SAMPLING]
def rs(finalbeforers):
	finalafter = {'2019': [], '2018': [], '2017': []}
	for k,v in finalbeforers.items():
		if k == '2019':
			finalafter['2019'] = random.sample(finalbeforers[k], 100)
		elif k == '2018':
			finalafter['2018'] = random.sample(finalbeforers[k], 100)
		else:
			finalafter['2017'] = random.sample(finalbeforers[k], 100)
	return (finalafter)

# [DRIVER]
def testmain():
	c19, c18, c17 = populate_county_dictionary()
	finalbeforers = scrape_naip(c19, c18, c17)
	finalafterrs = rs(finalbeforers)
	with open('randomlysampledstores.pickle', 'wb') as f:
		pickle.dump(finalafterrs, f)

testmain()