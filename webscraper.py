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

#[PACKAGE IMPORTS]
import csv
import requests
import json
import argparse
import traceback
import json
from bs4 import BeautifulSoup

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
	file = open("reducedlist.txt", "w")
	with open('stores.txt') as json_file:
		data = json.load(json_file)
		for k,v in data.items():
			for key, value in dictionary.items():
				temp_city_holder = v[1].upper()
				temp_city_holder = temp_city_holder.replace("SAINT", "ST")
				if (temp_city_holder == value[0].upper() and v[2].upper() == key[0].upper()):
					file.write(str((v[0].upper(), key[0], key[1])))
					file.write('\n')
	file.close()

#[SCRAPE DICTIONARY OF AVAILABLE DATA FROM NAIP]
def scrape_naip():
	page = requests.get("https://nrcs.app.box.com/v/naip/folder/17936490251")
	soup = BeautifulSoup(page.content, 'html.parser')
	for item in soup.find("div", {"class": "ReactVirtualized__Table__headerRow table-header"}):
		print (item)

def testmain():
	scrape_naip()
	
testmain()