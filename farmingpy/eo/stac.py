
import pandas as pd

def items_to_df(items, source="cdse"):
    item_data = []
    if source == "cdse":
        for item in items:
            props = dict(time = item.properties["datetime"],
                gridcode = item.properties["grid:code"],
                id = item.id
                )
            props.update(item.properties["statistics"])
            item_data.append(props)
    elif source == "es":
        for item in items:
            props = dict(time = item.properties["datetime"],
                gridcode = item.properties["grid:code"],
                id = item.id,
                vegetation =  item.properties["s2:vegetation_percentage"],
                not_vegetated = item.properties['s2:not_vegetated_percentage'], 
                water =  item.properties['s2:water_percentage'],
                nodata = item.properties['s2:nodata_pixel_percentage']
                )
            item_data.append(props)
    
    data = pd.DataFrame(item_data)
    data["good"] = data[["vegetation", "not_vegetated", "water"]].sum(axis=1)
    data.insert(0, "date", pd.to_datetime(data["time"]).dt.date)
    return data