# -*- coding:utf-8 -*-

import pickle,pprint
from PIL import Image
import numpy as np
import sys
import matplotlib.image as plimg
SIZE=32
def txt2array(filename):
    print("List read begin.")
    names = []
    labels = []
    with open(filename, 'r') as file_to_read:
      while True:
        lines = file_to_read.readline() 
        if not lines:
          break
          pass
        p_tmp, E_tmp = [str(i) for i in lines.split()]
        names.append(p_tmp)  
        labels.append(E_tmp)
        pass
      names_arr = np.array(names) 
      labels_arr = np.array(labels)
      pass
    print("List read over.")
    return names_arr,labels_arr


def image_input(filenames,all_list):
    output = sys.stdout
    j="$"
    for index,filename in enumerate(filenames):
        arr = read_file(filename)
        if all_list==[]:
            all_list = arr
        else:
            all_list= np.concatenate((all_list,arr))
            j+="$"
            output.write('\rhave done -->:%.0f%%' % int((index/len(filenames))*100))
    all_re=all_list.reshape((len(filenames),pow(SIZE,2)*3))
    output.flush()
    return all_re
def read_file(filename):
    im = Image.open(filename)
    r, g, b = im.split()
    r_arr = plimg.pil_to_array(r)
    g_arr = plimg.pil_to_array(g)
    b_arr = plimg.pil_to_array(b)
    r_arr1 = r_arr.reshape(pow(SIZE,2))
    g_arr1 = g_arr.reshape(pow(SIZE,2))
    b_arr1 = b_arr.reshape(pow(SIZE,2))
    arr = np.concatenate((r_arr1, g_arr1, b_arr1))
    arr_list=arr.tolist()
    return arr_list

def pickle_save(picarr,labarr):
    print("\r Saving")
    contact = {'data': picarr,'labels':labarr}
    f = open(txtdir+"_bin", 'w+b')
    pickle.dump(contact, f)
    f.close()
    print("Save done!")

if __name__ == "__main__":
    txtdir="800"+".txt"
    filenames,labels =txt2array(txtdir)
    all_arr=[]
    all_ar=image_input(filenames,all_arr)
    pickle_save(all_ar,labels)
    print("Size of the last array:"+str(all_ar.shape))
