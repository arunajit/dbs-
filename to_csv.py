import os
import csv

dirpath = './document-analytics-master/employment contracts'
output = './output_file.csv'
with open(output, 'w') as outfile:
    csvout = csv.writer(outfile)
    csvout.writerow(['FileName', 'Content'])

    files = os.listdir(dirpath)
    
    for filename in files:
        with open(dirpath + '/' + filename, mode='r' , encoding="utf8") as afile:
            csvout.writerow([filename, afile.read().encode("utf-8")])
            afile.close()       
            
    outfile.close()
    
    

