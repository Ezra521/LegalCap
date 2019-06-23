import json


def init():
	f = open('../data_original/accu.txt', 'r', encoding = 'utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()
		accu[line.strip()] = len(accu)
		line = f.readline()
	f.close()
	return  (accu, accuname)

accu, accuname = init()

def getClassNum(kind):
	global accu
	if kind == 'accu':
		return len(accu)
def getName(index, kind):
	global accuname
	if kind == 'accu':
		return accuname[index]



def getlabel(d):
	global accu
	return accu[d['meta']['accusation'][0]]




