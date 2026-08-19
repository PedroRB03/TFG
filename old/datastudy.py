def _stat_tokencount(df,tok):
    cats = df['label'].unique()
    t = df['txt'].str.count(tok).sum()
    print("\\multicolumn{3}{|c|}{"+tok+"} \\\\ \\hline")
    print("Clase & Apariciones & Frecuencia \\\\ \\hline")
    for c in cats:
        n = df[df['label'] == c]['txt'].str.count(tok).sum()
        print(f'{c} & {n} & {n/t*100:.2f}\\% \\\\ \\hline')
    print(f'Total & {t} & \\\\ \\hline')

# Muestra el nº de tokens por clase
def stat_tokencount(df):
    print("="*20 + "Frecuencia de [URL] y [USER]" + "="*20)
    print("\\begin{tabular}{|c|c|c|} \\hline")
    _stat_tokencount(df,"[URL]")
    _stat_tokencount(df,"[USER]")
    print("\\end{tabular}")

        # Contar [URL] y [USER]
        #stat_tokencount(df)
        