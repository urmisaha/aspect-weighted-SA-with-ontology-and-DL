import xmltodict, json
import numpy as np

food_count=0
service_count=0
price_count=0
ambience_count=0
misc_count=0
food_pos=0
food_neg=0
service_pos=0
service_neg=0
price_pos=0
price_neg=0
ambience_pos=0
ambience_neg=0
misc_pos=0
misc_neg=0
overall_pos=0
overall_neg=0
 
# pn: aspect positive overall negative, np: aspect negative, overall positive
food={'pp': 0, 'pn': 0, 'np': 0, 'nn': 0}
service={'pp': 0, 'pn': 0, 'np': 0, 'nn': 0}
price={'pp': 0, 'pn': 0, 'np': 0, 'nn': 0}
ambience={'pp': 0, 'pn': 0, 'np': 0, 'nn': 0}
misc={'pp': 0, 'pn': 0, 'np': 0, 'nn': 0}

dataset = []
 
with open('test_data.xml', 'r') as myfile:
    data=myfile.read().replace('\n', '')
    d = xmltodict.parse(data)
    sentences = json.loads(json.dumps(dict(d)))["sentences"]["sentence"]
    for s in sentences:
        # print (s[u'@id'])
        dataset.append([s[u'text'], 1 if s[u'overallSentiment'][u'@polarity']=='positive' else 0])
        if s[u'overallSentiment'][u'@polarity'] == 'positive':
            overall_pos = overall_pos + 1
            overall = 1
        else:
            overall_neg = overall_neg + 1
            overall = -1
        for aspectCategory in s[u'aspectCategories'][u'aspectCategory']:
            try:
                category = aspectCategory[u'@category']
                polarity = aspectCategory[u'@polarity']
            except:
                category = s[u'aspectCategories'][u'aspectCategory'][u'@category']
                polarity = s[u'aspectCategories'][u'aspectCategory'][u'@polarity']
            if category == 'food':
                food_count = food_count + 1
                if polarity == "positive" or polarity == "neutral":
                    food_pos = food_pos + 1
                    if overall == 1:
                        food['pp'] = food['pp'] + 1
                    else:
                        food['pn'] = food['pn'] + 1
                else:
                    food_neg = food_neg + 1
                    if overall == 1:
                        food['np'] = food['np'] + 1
                    else:
                        food['nn'] = food['nn'] + 1
            elif category == 'service':
                service_count = service_count + 1
                if polarity == "positive" or polarity == "neutral":
                    service_pos = service_pos + 1
                    if overall == 1:
                        service['pp'] = service['pp'] + 1
                    else:
                        service['pn'] = service['pn'] + 1
                else:
                    service_neg = service_neg + 1
                    if overall == 1:
                        service['np'] = service['np'] + 1
                    else:
                        service['nn'] = service['nn'] + 1
            elif category == 'price':
                price_count = price_count + 1
                if polarity == "positive" or polarity == "neutral":
                    price_pos = price_pos + 1
                    if overall == 1:
                        price['pp'] = price['pp'] + 1
                    else:
                        price['pn'] = price['pn'] + 1
                else:
                    price_neg = price_neg + 1
                    if overall == 1:
                        price['np'] = price['np'] + 1
                    else:
                        price['nn'] = price['nn'] + 1
            elif category == 'ambience':
                ambience_count = ambience_count + 1
                if polarity == "positive" or polarity == "neutral":
                    ambience_pos = ambience_pos + 1
                    if overall == 1:
                        ambience['pp'] = ambience['pp'] + 1
                    else:
                        ambience['pn'] = ambience['pn'] + 1
                else:
                    ambience_neg = ambience_neg + 1
                    if overall == 1:
                        ambience['np'] = ambience['np'] + 1
                    else:
                        ambience['nn'] = ambience['nn'] + 1
            elif category == 'anecdotes/miscellaneous':
                misc_count = misc_count + 1
                if polarity == "positive" or polarity == "neutral":
                    misc_pos = misc_pos + 1
                    if overall == 1:
                        misc['pp'] = misc['pp'] + 1
                    else:
                        misc['pn'] = misc['pn'] + 1
                else:
                    misc_neg = misc_neg + 1
                    if overall == 1:
                        misc['np'] = misc['np'] + 1
                    else:
                        misc['nn'] = misc['nn'] + 1
            else:
                print ("No aspect found!")

food['pp'] = food['pp']/food_count
food['pn'] = food['pn']/food_count
food['np'] = food['np']/food_count
food['nn'] = food['nn']/food_count
service['pp'] = service['pp']/service_count
service['pn'] = service['pn']/service_count
service['np'] = service['np']/service_count
service['nn'] = service['nn']/service_count
price['pp'] = price['pp']/price_count
price['pn'] = price['pn']/price_count
price['np'] = price['np']/price_count
price['nn'] = price['nn']/price_count
ambience['pp'] = ambience['pp']/ambience_count
ambience['pn'] = ambience['pn']/ambience_count
ambience['np'] = ambience['np']/ambience_count
ambience['nn'] = ambience['nn']/ambience_count
misc['pp'] = misc['pp']/misc_count
misc['pn'] = misc['pn']/misc_count
misc['np'] = misc['np']/misc_count
misc['nn'] = misc['nn']/misc_count

aspect_weights = {}
aspect_weights['food'] = food['pp'] + food['nn']
aspect_weights['service'] = service['pp'] + service['nn']
aspect_weights['price'] = price['pp'] + price['nn']
aspect_weights['ambience'] = ambience['pp'] + ambience['nn']
aspect_weights['misc'] = misc['pp'] + misc['nn']

# print ("counts:")
# print ("overall positive sentences: ", overall_pos)
# print ("overall negative sentences: ", overall_neg)
# print ("food_count = ", food_count, "   pos: ", food_pos, "  neg: ", food_neg)
# print ("service_count = ", service_count, " pos: ", service_pos, "   neg: ", service_neg)
# print ("price_count = ", price_count, " pos: ", price_pos, "     neg: ", price_neg)
# print ("ambience_count = ", ambience_count, "   pos: ", ambience_pos, "  neg: ", ambience_neg)
# print ("misc_count = ", misc_count, "   pos: ", misc_pos, "  neg: ", misc_neg)
# print ("\nmatrix:")
# print ("food: ", food)
# print ("service: ", service)
# print ("price: ", price)
# print ("ambience: ", ambience)
# print ("misc: ", misc)
print ("\naspect_weights:")
print (aspect_weights)

weights = []
weights.append(aspect_weights['food'])
weights.append(aspect_weights['service'])
weights.append(aspect_weights['price'])
weights.append(aspect_weights['ambience'])
weights.append(aspect_weights['misc'])

def softmax(l):
    return np.exp(l)/np.sum(np.exp(l))  

weights = softmax(weights)
aspects = ['food', 'service', 'price', 'ambience', 'misc']
aspect_weights = {}

for i, aspect in enumerate(aspects):
    aspect_weights[aspect] = weights[i]


f1 = open("train.csv", "w")
f2 = open("test.csv", "w")
for i, row in enumerate(dataset): 
    if i < int(len(dataset)/4):
        f2.write(str(row[0]).replace(","," ")+','+str(row[1])+'\n')
    else:
        f1.write(str(row[0]).replace(","," ")+','+str(row[1])+'\n')
f2.close()
f1.close()