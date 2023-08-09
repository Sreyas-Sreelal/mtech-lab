import xmlschema
from lxml import etree

def print_xml(xml_data,query):
    xml = xml_data.xpath(query)
    print("####################",query,"###########################")
    try:
        for x in xml:
            try:
                print(''.join(x.itertext()))
            except AttributeError:
                print(x)
    except TypeError:
        print(xml)
schema = xmlschema.XMLSchema("schema.xsd")
data = "data.xml"
if not schema.is_valid("data.xml"):
    print("Data is not valid and not following the schema!!")
    pass

xml_data = etree.parse(data)

print_xml(xml_data,"//books")
print_xml(xml_data,"//book/title")
print_xml(xml_data,"//book[price>100]/title")
print_xml(xml_data,'//book[author="George RR Martin"]/title')
print_xml(xml_data,'//book[year>1990]/title')
print_xml(xml_data,'//book[rating>5]/title')
print_xml(xml_data,'//book[starts-with(@publisher,"H")]/@publisher')
print_xml(xml_data,'//book[starts-with(@publisher,"H")]/title')
print_xml(xml_data,'sum(//book/price)')
