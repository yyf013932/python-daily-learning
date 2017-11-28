import pandas as pd
import numpy as np

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import jieba
import matplotlib.pyplot as plt

data = pd.read_csv('d:/resources/qq_mess_analysis/qq_format.csv')

content = ""

data_tuzi = data[data.name == '三洋虚不虚']

for value in data_tuzi['content']:
    content += value

# 中文字体路径
font = 'C:\Windows\Fonts\simfang.ttf'
wc = WordCloud(font_path=font, width=800, height=600, min_font_size=6, max_words=100)

f = pd.read_csv('d:/resources/sms_analysis/stopwords.csv', encoding='gb2312')
stopwords = list(np.squeeze(f.values))
stopwords.append("图片")
stopwords.append("表情")

# 进行分词
wordList = jieba.cut(content)
textNew = ' '.join([w for w in wordList if w not in stopwords])

wc.generate(textNew)

plt.figure()
# 以下代码显示图片
plt.imshow(wc)
plt.axis("off")
plt.show()

wc.to_file("d:/out1.jpg")
