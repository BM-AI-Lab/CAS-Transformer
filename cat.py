import pandas as pd

# data1 = pd.read_csv('./swin-T/result/spiral_datas.csv')
# data2 = pd.read_csv('./viT/result/spiral_datas.csv')
data3 = pd.read_csv('./data/CA_Mdatas.csv')
data4 = pd.read_csv('./data/meander_datas.csv')

#将swinT和viT提取的特征拼接起来

# file = [data1, data2]
# outfile = pd.concat(file, axis=1, join='outer')
# outfile.to_csv('./data_cat/result_spiral.csv', index=0, sep=',')
#
# print('拼接成功')
#
#


#
# file = [data3, data4]
# outfile = pd.concat(file, axis=1, join='outer')
# outfile.to_csv('./data_cat/CA-Mcat.csv', index=0, sep=',')
#
# print('拼接成功')




#首先将拼接成功的特征用txt打开，保存到txt文件中，然后消除空格，在重新保存到新的csv文件中
f = open('./data_cat/CA-Mcat.txt')
lines = f.readlines()
newtxt = ''
for line in lines:
    #line = " ".join(line.split(","))
    newtxt = newtxt + ' '.join(line.split(',')) + '\n'
print(newtxt)

fo = open('./data_cat/CA-Mcat1.csv', 'x')
fo.write(newtxt)
fo.close()








    #line = str.replace(',', ' ')
    #line = ' '.join(line.split(','))
    #line = line.strip(",")
    #line = line.split(" ")
    #line = [float(x) for x in line]


