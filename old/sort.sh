(head -n 1 comments/RC_2024-12.csv && tail -n +2 comments/RC_2024-12.csv | sort --parallel=8 -t, -k1,1 -S 500M) > comments/sorted.csv
> finished