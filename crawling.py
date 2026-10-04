import os
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs


news = []
for i in range(1,4):
  url = 'https://news.daum.net/breakingnews/sports/worldsoccer?page=' + str(i)
  res = requests.get(url)

  soup = bs(res.text, 'lxml')
  ul = soup.find("ul",{"class":"list_news2 list_allnews"}).findAll("li")

  for li in ul:
        data = li.find("a",{"class":"link_txt"})
        news.append({
          'title': data.text,
          'topic_idx': 5
              })

# 중복제거
news = list(map(dict, set(tuple(sorted(d.items())) for d in news)))

dataframe = pd.DataFrame(news)

PATH = "/content/drive/MyDrive/files/news/kor/craw_data/sport6.csv"
dataframe.to_csv(PATH, index=False )


# 수집한 데이터를 하나의 csv 파일로 만든다.
forders = os.listdir('/content/drive/MyDrive/files/news/eng/craw_data')
print(forders)

df_all = pd.DataFrame()
for i in range(0,len(forders)):
    if forders[i].split('.')[1] == 'csv':
        file = '/content/drive/MyDrive/files/news/eng/craw_data/'+forders[i]
        df= pd.read_csv(file,encoding='utf-8') 
        df_all = pd.concat([df_all, df])

# csv 파일로 저장
PATH = "/content/drive/MyDrive/files/news/eng/craw_data/eng_data.csv"
df_all.to_csv(PATH, index=False )