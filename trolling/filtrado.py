import openpyxl
import csv
import math
import re
import pandas as pd

def normalize_text(text):
        text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text)
        text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text)
        text = re.sub(r'(\<b\>)|(\<b\\\/\>)|\[removed\]','', text)
        return text


data = {
    "text" : [],
    "ragescore" : [],
}

with open('ragebait.csv','w',encoding='utf-8') as f:

    writer = csv.writer(f,quoting=csv.QUOTE_ALL)
    writer.writerow(['text','ragescore'])

    wb = openpyxl.load_workbook('trolling.xlsx')
    sheet = wb.active

    n = 0
    gen = sheet.rows
    gen.__next__()

    N = math.inf
    for row in gen:
        #print(row[4].value)
        n+=1
        txt = normalize_text(str(row[0].value))
        rb = [row[1].value,row[2].value,row[3].value].count("Trolling")
        if len(txt) > 0:
            writer.writerow([txt, rb])
        if rb == "Ragebait":
            data['text'].append(txt)
            data['ragescore'].append(rb)
        if n >= N:
            break

df = pd.DataFrame(data)
df.to_pickle("rb_db.pkl")
print(df.head())