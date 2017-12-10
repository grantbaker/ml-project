import csv

directors = {}
unique_director = 1
with open('data/MetaData2.csv', newline='\n') as csvfile:
	meta = csv.reader(csvfile, delimiter=',')
	for row in meta:
		director = row[-1]
		if (director in directors.keys()):
			directors[director][0] += 1
			if (len(directors[director]) == 1):
				directors[director].append(unique_director)
				unique_director += 1
		else:
			directors[director] = [1]

outfile = open('data/MetaData3.csv', 'w')

with open('data/MetaData2.csv', newline='\n') as csvfile:
	prev_meta = csv.reader(csvfile, delimiter=',')
	for row in prev_meta:
		director = row[-1]
		if (directors[director][0] > 1):
			dir_num = directors[director][1]
		else:
			dir_num = 0
		#print(dir_num)
		outfile.write(','.join(row)+','+str(dir_num)+'\n')

outfile.close()


