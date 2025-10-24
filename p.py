import time

wins1=0
wins2=0
for j in range(0,1000):
    t1 = 0
    for i in range(0,10000):
        a = [1,2,3,4]
        b = 5
        c = time.time()
        [1,2,3,4].append(b)
        t1+=time.time()-c


    #print(f"append ns: {t1}")
    t2 = 0

    for i in range(0,10000):
        a = [1,2,3,4]
        b = 5
        c = time.time()
        a.append(b)
        t2+=time.time()-c

    #print(f"+ ns: {t2}")
    if t1 > t2:
        wins2+=1
    elif t2 > t1:
        wins1+=1
        
print(f"w1: {wins1}, w2: {wins2}")
if wins1 > wins2:
    print("wins1 mejor")
else:
    print("wins2 mejor")