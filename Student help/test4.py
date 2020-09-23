import pandas as pd 
import numpy as np  
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
df = pd.DataFrame(np.random.randint(1, 7, 6000), columns = ['one'])

import scrapy
from scrapy.crawler import CrawlerProcess

class OAKSpider(scrapy.Spider):
# Naming the spider is important if you are running more than one spider of
# this class simultaneously.
name = "OAKS"

# URL(s) to start with.
start_urls = [
'https://www.governmentjobs.com/jobs?keyword=&location=Oakland%2C+CA'
]

# Use XPath to parse the response we get.
def parse(self, response):

# Iterate over every <tr> element on the page.
for table_row in response.xpath('//tr'):

# Yield a dictionary with the values we want.
yield {
'Job Title': table_row.xpath('td[@class="job-table-title"]/h3/text()').extract_first(),
'Salary ': table_row.xpath('td[@class="job-table-salary"]/text()').extract_first(),
'Job Type': table_row.xpath('td[@class="job-table-type"]/text()').extract_first(),
'Closing': table_row.xpath('td[@class="job-table-closing"]/text()').extract_first()
}
# Get the URL of the next page.
next_page = response.xpath('//li[@class="PagedList-skipToNext"]/a/@href').extract_first()

if next_page is not None:
next_page = response.urljoin(next_page)
# Request the next page and recursively parse it the same way we did above
yield scrapy.Request(next_page, callback=self.parse)

# Tell the script how to run the crawler by passing in settings.
# The new settings have to do with scraping etiquette.
process = CrawlerProcess({
'FEED_FORMAT': 'json', # Store data in JSON format.
'FEED_URI': 'oak_data.json', # Name our storage file.
'LOG_ENABLED': False, # Turn off logging for now.
'ROBOTSTXT_OBEY': True,
'USER_AGENT': 'ThinkfulDataScienceBootcampCrawler (http://thinkful.com)',
'AUTOTHROTTLE_ENABLED': True,
'HTTPCACHE_ENABLED': True
})

# Start the crawler with our spider.
process.crawl(OAKSpider)
process.start()
print('Success!')```

'''
df['two'] = df['one'] + np.random.randint(1, 7, 6000)
print df
df.two.plot.hist( alpha=0.5)
plt.show()

import hdbscan
from sklearn.datasets import make_blobs

data, _ = make_blobs(1000)

clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
cluster_labels = clusterer.fit_predict(data)
hierarchy = clusterer.cluster_hierarchy_
alt_labels = hierarchy.get_clusters(0.100, 5)
hierarchy.plot()
plt.show()
'''
