import requests
import config as C 
import json
from datetime import datetime 
import os

def get_weather():
	"""
	Query openweathermap.com's API and to get the weather for
	San Francisco, CA and then dump the json to the /weather data/ directory 
	with the file name "<today's date>.json" using get request.
	"""
	paramaters = {'q': 'San Francisco, USA', 'appid':C.API_KEY}

	result     = requests.get("http://api.openweathermap.org/data/2.5/weather?", paramaters)

	# If the API call was sucessful, get the json and dump it to a file with 
	# today's date as the title.
	if result.status_code == 200 :

		# Get the json data 
		json_data = result.json()
		file_name  = str(datetime.now().date()) + '.json'
		tot_name   = os.path.join(os.path.dirname(__file__), 'weather data', file_name)

		with open(tot_name, 'w') as outputfile:
			json.dump(json_data, outputfile)
        elif result.status_code == 301 :
                print "The server is redirecting you to a different endpoint (company switched domain names, or an endpoint name is changed)." 
        elif result.status_code == 401 :
                print "The server thinks you're not authenticated (you don't send the right credentials to access an API)."
        elif result.status_code == 400 :
                print "You made a bad request (send along the right data)."
        elif result.status_code == 403 :
                print "The access is forbidden  (you don't have the right permissions to see it)."
        elif result.status_code == 404: 
                print "The resource you tried to access wasn't found on the server. "
	else :
		print "Error In API call."


if __name__ == "__main__":
	get_weather()
        print ('done')
