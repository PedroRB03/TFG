import csv
import os
from math import floor

# comms: ['score', 'created', 'link', 'body', 'id', 'subreddit', 'parent_id', 'controversiality']
comm_file = r"C:\\Users\\pedro\\Downloads\\reddit\\comments\\RC_2024-12.csv"
comm_out_file = r"C:\\Users\\pedro\\Downloads\\reddit\\comments\\RC_2024-12-onlytop.csv"

file_lines=0
with open(file=comm_file,encoding='utf_8') as f:    
    with open(file=comm_out_file,mode="w", newline='',encoding='utf_8') as f2:
        reader = csv.reader(f)
        writer = csv.writer(f2)
        first = True
        for r in reader:
            if not first:
                if r[0][0:3] == "t3_":
                    #print(r)
                    writer.writerow(r)
                file_lines += 1
                if file_lines % 1000000 == 0:
                    print(f"{file_lines} lines done")
            else:
                writer.writerow(r)
                first = False
