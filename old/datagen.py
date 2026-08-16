
    data = { # datos crisp
        "txt" : [],
        "label" : [],
    }

    data_f = { # datos fuzzy
        "txt" : [],
        "label" : [],
    }
    

wb = openpyxl.load_workbook(FILE)
    sheet = wb.active

    n = 0
    gen = sheet.rows
    gen.__next__() # Saltar primera fila

    for row in gen: # Recordamos, saltando la primera fila
        txt = normalize_text(str(row[0].value))
        vote = [row[1].value,row[2].value,row[3].value]
        rb = vote.count("Trolling")

        filtrar = False
        # Comprobamos que no hayan valores no deseados
        for v in vote:
            if v not in ["Trolling","Normal"]:
                filtrar = True
                break
            
        if len(txt) > 0 and not filtrar: # No añadir filas en blanco o con valores no admitidos
            n+=1
            #if rb in [0,3]:
            data_f['txt'].append(txt)
            data_f['label'].append(round(rb/3 *100)/100)

            data['txt'].append(txt)
            data['label'].append(round(rb/3))
            
            if n >= N: # Cortamos al leer N filas.
                break


                        if VERBOSE:
                            print(f"TRAIN \n{df_tr.tail()}")
                            print(f"EVAL \n{df_ev.tail()}")
                            print(f"TEST \n{df_tst.tail()}")
                            
                        if CHECK_COL and (_distnel(df_tr["txt"],df_ev["txt"]) or _distnel(df_tst["txt"],df_ev["txt"])):
                            print("Hay elementos comunes en los datasets...")
                            print("Dataset: "+title+" en semilla: "+str(seed))
                            break