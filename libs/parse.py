import re

def parse_picks(picks):
    return re.split(r',\s*|\s+', picks) if picks else None


def get_channels_from_xml_desc(xml_desc):
    sensor_names = []
    ch = xml_desc.child("channels").child("channel")
    while not ch.empty():
        label = ch.child_value("label")
        if label:
            sensor_names.append(label)
        ch = ch.next_sibling("channel")

    return sensor_names


def print_xml_element(element, indent=""):
    print(f"{indent}Element: {element.name()}")
    print(f"{indent}  Value: {element.value()}")
    
    child = element.first_child()
    while child.e:
        print_xml_element(child, indent + "  ")
        child = child.next_sibling()