import re


def ko_en(text):
  text = re.sub('\\\\n', ' ', text)
  text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]', ' ', text)
  text = re.sub('[\s]+', '', text)
  text = text.lower()

  for txt in text.split():
    cnt_en, cnt_ko = 0, 0
    for i in txt:
      for j in i:
        if ord('ㄱ') <= ord(j) <= ord('힣'):
          cnt_ko += 1
        elif ord('a') <= ord(j) <= ord('z'):
          cnt_en += 1

  if len(text):
    if (cnt_ko / len(text)) >= 0.2:
      return 'Korean'
    else:
      return 'English'