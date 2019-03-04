from xml.etree.ElementTree import ElementTree

tree = ElementTree()
root = tree.parse("test_data.xml")
sentences = root.findall("sentence")
aspect_term_list = []

for sentence in sentences:
    for aspect in sentence.findall("aspectTerms"):
        for term in aspect.findall("aspectTerm"):
            aspect_term = term.attrib["term"]
            if aspect_term not in aspect_term_list:
                aspect_term_list.append(aspect_term)

# print (aspect_term_list)
