import xmltodict, json
import pandas as pd
classes = []    
with open('EN_LAPT_SB2_TEST.gold.xml', 'r') as myfile:
    data=myfile.read().replace('\n', '')
    d = xmltodict.parse(data)
    reviews = json.loads(json.dumps(dict(d)))["Reviews"]["Review"]
    for r in reviews:
        for opinion in r[u'Opinions'][u'Opinion']:
            try:
                if opinion[u'@category'] not in classes:
                    classes.append(opinion[u'@category'])
            except:
                if r[u'Opinions'][u'Opinion'][u'@category'] not in classes:
                    classes.append(r[u'Opinions'][u'Opinion'][u'@category'])

            # try:
            #     categories = opinion[u'@category'].split("#")
            #     for c in categories:
            #         if c not in classes:
            #             classes.append(c)
                
            # except Exception as e:
            #     print(e)
 
    df = pd.DataFrame( columns=["sentence"]+classes)
    for r in reviews:
        row={}
        s = ""
        for sentence in r[u'sentences'][u'sentence']:
            s+= sentence[u'text']
        # print(s)
        row["sentence"] = s
        for opinion in r[u'Opinions'][u'Opinion']:
            try:
                row[opinion[u'@category']]="100" if opinion[u'@polarity']=='positive' else "010"
            except:
                row[r[u'Opinions'][u'Opinion'][u'@category']]="100" if r[u'Opinions'][u'Opinion'][u'@polarity']=='positive' else "010"
            # try:
            #     categories = opinion[u'@category'].split("#")
            #     for c in categories:
            #         row[c]=1
                
            # except Exception as e:
            #     print(e)
        df=df.append(row, ignore_index=True)

df = df.fillna("001")
df.to_csv("dataset_embedding_test.csv", encoding='utf-8', index=False)