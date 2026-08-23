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
        
# Muestra número tokens medio, mínimo y máximo por clase
def stats_tokens(df,model_name):
    t = len(df['label'])
    print("="*10 + "Tokens por clase" + "="*10)


    tokenizer, tokenize_function = get_tokenizer(model_name)


    df = Dataset.from_pandas(df)
    df = df.map(tokenize_function, batched=True)
    df = df.to_pandas()
    df["num_tokens"] = df["input_ids"].apply(len)

    for c, grp in df.groupby("label"):
        print(f"Valor mínimo de {c}: {grp['num_tokens'].min()}")
        print(f"Valor máximo de {c}: {grp['num_tokens'].max()}")
        print(f"Valor medio de {c}: {grp['num_tokens'].mean()}")

        # Número de tokens
        stats_tokens(df,MODEL_NAME)