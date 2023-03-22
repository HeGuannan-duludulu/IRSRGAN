import jieba


cnm = "我爱老师"

fen_list = jieba.lcut(cnm)
print(fen_list)