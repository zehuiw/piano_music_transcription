a = open('dev.lst', 'r')
b = open('test.lst', 'r')

candy = []
for l in a:
	candy.append(l)
for l in b:
	candy.append(l)

from random import shuffle
shuffle(candy)
a.close()
b.close()

ll = len(candy) / 2
with open('test.txt', 'wb') as f:
	for ii in candy[:ll]:
		f.write(ii)
with open('dev.txt', 'wb') as f:
	for ii in candy[ll:]:
		f.write(ii)
