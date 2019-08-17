#usr/bin/env python
# -*- coding:utf-8 -*-
import sys

filename=sys.argv[1]
filename_out=sys.argv[2]

label_title = []
f1 = open(filename,"r")
for line in f1:
    words_list = line.strip("\n").split("\t")
    label = words_list[0]
    title = "".join(words_list[1:])
    title = title.replace(" ", "").replace("\r", "").decode("utf-8")
    label_title.append("__label__"+label+" , , " + ''.join(aa+ " " for aa in title)+"\n" )
f1.close()
f2 = open(filename_out,'w')
for line in label_title:
    f2.write(line.encode('utf-8'))
f2.close()
