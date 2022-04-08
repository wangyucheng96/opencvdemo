import numpy as np
# filename = 'cali_test_2.txt'
f = open("cali_test_2.txt", "r")
lines = f.readlines()  # 读取全部内容
cali_data = []
for line in lines:
    cali_data.append(line)
print(cali_data[0], cali_data[1], cali_data[2], cali_data[3], cali_data[4], cali_data[5])
# data = np.loadtxt(filename, dtype=np.float32, delimiter=',')
# print(data)
theo_angs = []
coordings = []
for i in range(0, len(cali_data)):
    data = cali_data[i].split()
    theo_angs.append(eval(data[0]))
    theo_angs.append(eval(data[1]))
    coordings.append(eval(data[2]))
    coordings.append(eval(data[3]))
    # np.savetxt('data.txt', (eval(data[0]), eval(data[1]), eval(data[2]), eval(data[3])))
    # with open('data.txt', 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
    #     for i in data:
    #         print(eval(i))
    #         f.write("%f\n" % i)  # 写入数据，文件保存在上面指定的目录，加\n为了换行更方便阅读
    # print(eval(data[0]), eval(data[1]), eval(data[2]), eval(data[3]))
print(theo_angs)
print(coordings)
a=np.array(theo_angs)
b=np.array(coordings)
np.save('theo_angs.npy',a)
np.save('coording.npy',b)
