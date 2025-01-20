from selenium import webdriver
from bs4 import BeautifulSoup
import bs4
import requests
import pandas as pd
from selenium.webdriver.common.by import By
from datetime import datetime

driver = webdriver.Chrome()

hacking_trends = pd.DataFrame()
title_link = []

URL = "https://thehackernews.com/"
for i in range(120):
    driver.get(URL)
    story_link = driver.find_elements(By.CSS_SELECTOR, 'div a')

    for a in story_link:
        if a.get_attribute('class') == 'story-link':
            title_link.append(a.get_attribute('href'))
        if a.get_attribute('id') == 'Blog1_blog-pager-older-link':
            URL = a.get_attribute('href')

hacking_trends['title_link'] = title_link
hacking_trends['titles'] = ''
hacking_trends['posted_date'] = ''
hacking_trends['hacking_type'] = ''

for i in range(hacking_trends.shape[0]):
    url = hacking_trends.title_link[i]
    driver.get(url)
    x = driver.page_source
    x = x.replace(">","> ")
    soup = bs4.BeautifulSoup(x, 'html.parser')
    title_text = soup.find("h1", {"class": "story-title"})
    post_date = soup.find("span", {"class": "author"})
    hack_type = soup.find("span", {"class": "p-tags"})
    loc_para = soup.find("p")

    try:
        hacking_trends.titles[i] = title_text.text
        hacking_trends.posted_date[i] = post_date.text
        hacking_trends.hacking_type[i] = hack_type.text
        hacking_trends.text_loc[i] = loc_para.text
    except:
        pass

hacking_trends.to_csv("Hacking_Trends.csv", index = False)