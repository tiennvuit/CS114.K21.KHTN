# The program crawl video links

import argparse

import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import colorama

def get_links(url: str, max_numbers: int):
	main_page = BeautifulSoup(requests.get(url).content, "html.parser")
	href_Tags = main_page.find_all("div",attrs= {"class":"mozaique"})[0].find_all(href = True)
	video_links = []
	for tag in href_Tags:
		link = tag.attrs["href"]
		if link not in video_links and "video" in link and "http" not in link:
			video_links.append(link)

	link_videos = set()
	i = 0
	for video in video_links:
		video_url = url+video
		web_page = BeautifulSoup(requests.get(video_url).content, "html.parser")
		try:
			link = web_page.find_all("div" , attrs={"id":"html5video_base"})[0].find_all(href = True)
			for x in link:
				if "High" in str(x):
					mp4_link = x.attrs['href']
					link_videos.add(mp4_link)
					i += 1
		except:
			pass

		if i == max_numbers:
			break
	return link_videos

def main(args):
	url = args.url
	max_numbers = args.n

	links = get_links(url=url, max_numbers=max_numbers)
	#print("The crawled video links: ")

	with open('video_links.txt', 'a') as f:
		for link in links:
			f.write(str(link))

if __name__ == "__main__":
	parser  = argparse.ArgumentParser(description='Process crawl xvideo')
	parser.add_argument('-url', help='URL of page', default='https://xvideos.com')
	parser.add_argument('-n', help='Maximum number of video crawled', default=10)
	args = parser.parse_args()
	main(args)
