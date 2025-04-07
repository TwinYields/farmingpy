# convert dictionary to NGSI-LD format with Property/Relationship notation
# input parameter is pydantic dump
def pydump_to_ngsild(pydump):
    ngsidump = {}
    for key, value in pydump.items():
        if key == 'agriparcel':
            ngsidump[key] = {'object':value, 'type':'Relationship'}
        elif key == 'id' or key == 'type':
            ngsidump[key] = value
        else:
            ngsidump[key] = {'value':value, 'type':'Property'}
    return ngsidump

# convert dictionary to NGSI-LD format with Property/Relationship notation
# input parameter is pydantic object
def pyobj_to_ngsild(pyobj):
    pydump = pyobj.model_dump(exclude_none=True)
    ngsidump = {}
    for key, value in pydump.items():
        if key == 'agriparcel':
            ngsidump[key] = {'object':value, 'type':'Relationship'}
        elif key == 'id' or key == 'type':
            ngsidump[key] = value
        elif key == 'location':
            ngsidump[key] = {'value':value, 'type':'GeoProperty'}
        else:
            ngsidump[key] = {'value':value, 'type':'Property'}
    return ngsidump

