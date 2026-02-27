import json
import math
from nltk.sentiment import SentimentIntensityAnalyzer    
sia = SentimentIntensityAnalyzer()


stats = {
    'rb':{
        'count' : 0,
        'mean' : 0,
        'min' : math.inf,
        'max' : -math.inf,
        'weighted':{
            'mean' : 0,
            'min' : math.inf,
            'max' : -math.inf,
        }
    },
    'no-rb':{
        'count' : 0,
        'mean' : 0,
        'min' : math.inf,
        'max' : -math.inf,
        'weighted':{
            'mean' : 0,
            'min' : math.inf,
            'max' : -math.inf,
        }
    },
    'maybe-rb':{
        'count' : 0,
        'mean' : 0,
        'min' : math.inf,
        'max' : -math.inf,
        'weighted':{
            'mean' : 0,
            'min' : math.inf,
            'max' : -math.inf,
        }
    }
}

inputf = "reddit_db.json"

#MAXC = math.inf

stat_keys = ['no-rb','maybe-rb','rb']

with open(inputf,encoding='utf-8') as f:
    dic = json.load(f)
    for pk in dic['posts']:
        pst = dic['posts'][pk]
        comments = list(pst['comments'].values())
        comments.sort(reverse=True,key=lambda c : c['score'])

        ragers = 0
        ragers_w = 0
        total = 0
        total_w = 0
        for c in comments:
            scores = sia.polarity_scores(c['body'])
            if scores['neg'] > 0:
                ragers+=1
                ragers_w+= scores['neg']*c['score']
            #else:
                #ragers_w+= -max(0,c['score'])
            total+=1
            total_w+= max(1,c['score'])
            #if total >= MAXC:
            #    break
        #print(f"Ragers: {ragers} / Total: {total}, Ratio: {ragers/total}")
        #print(f"Weighted: Ragers: {ragers_w} / Total: {total_w}, Ratio: {ragers_w/total_w}")
        if total > 0:
            k = stat_keys[pst['ragescore']]
            stats[k]['count']+=1
            stats[k]['mean']+=ragers/total
            stats[k]['weighted']['mean']+=ragers_w/total
            stats[k]['min'] = min(stats[k]['min'],ragers/total)
            stats[k]['max'] = max(stats[k]['max'],ragers/total)
            stats[k]['weighted']['min'] = min(stats[k]['weighted']['min'],ragers_w/total)
            stats[k]['weighted']['max'] = max(stats[k]['weighted']['max'],ragers_w/total)
    
    for k in stat_keys:
        stats[k]['mean'] /= stats[k]['count']
        stats[k]['weighted']['mean'] /= stats[k]['count']

    print("Non-Weighted:")
    print("\tMax:")
    print(f"\t\tRB {stats['rb']['max']}, Maybe-RB {stats['maybe-rb']['max']}, No-RB {stats['no-rb']['max']}")
    print("\tMin:")
    print(f"\t\tRB {stats['rb']['min']}, Maybe-RB {stats['maybe-rb']['min']}, No-RB {stats['no-rb']['min']}")

    print(f"\tMean RB: {stats['rb']['mean']}, Mean Maybe-RB: {stats['maybe-rb']['mean']}, Mean Non-RB: {stats['no-rb']['mean']}")
    print("Weighted:")
    print("\tMax:")
    print(f"\t\tRB {stats['rb']['weighted']['max']}, Maybe-RB {stats['maybe-rb']['weighted']['max']}, No-RB {stats['no-rb']['weighted']['max']}")
    print("\tMin:")
    print(f"\t\tRB {stats['rb']['weighted']['min']}, Maybe-RB {stats['maybe-rb']['weighted']['min']}, No-RB {stats['no-rb']['weighted']['min']}")

    print(f"\tMean RB: {stats['rb']['weighted']['mean']}, Mean Maybe-RB: {stats['maybe-rb']['weighted']['mean']}, Mean Non-RB: {stats['no-rb']['weighted']['mean']}")
    print("If joined:")
    print(f"\tMean RB: {(stats['rb']['mean']+stats['maybe-rb']['mean'])/2}, Mean Non-RB: {stats['no-rb']['mean']}\n\tMean Weighted RB: {(stats['rb']['weighted']['mean']+stats['maybe-rb']['weighted']['mean'])/2}, Mean Non-RB: {stats['no-rb']['weighted']['mean']}")

    # Weighted RB and Maybe-RB Mean: 2.766072189540906
    # Weighted Non-RB Mean: 1.7984402470303966
    