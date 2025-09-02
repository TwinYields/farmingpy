# convert dictionary to NGSI-LD format with Property/Relationship notation
# input parameter is pydantic dump
import shapely
import json
import datetime

# convert dictionary to NGSI-LD format with Property/Relationship notation
# input parameter is pydantic object
def pyobj_to_ngsild(pyobj):
    pydump = pyobj.model_dump(exclude_none=True)
    ngsidump = {}
    ngsidump["@context"] = "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    for key, value in pydump.items():
        if key == 'agriparcel':
            ngsidump[key] = {'object':value, 'type':'Relationship'}
        elif key == 'id' or key == 'type':
            ngsidump[key] = value
        elif key == 'location':
            value_geojson = shapely.to_geojson(value)
            value_geojson_dict = json.loads(value_geojson)
            ngsidump[key] = {'value':value_geojson_dict, 'type':'GeoProperty'}
        elif isinstance(value, datetime.date): 
            datetime_str = value.isoformat() # NGSI-LD requires date as a string in ISO 8601 format
            ngsidump[key] = {'value':datetime_str, 'type':'Property'}
        else:
            ngsidump[key] = {'value':value, 'type':'Property'}
    return ngsidump

