
# TAAR – Evaluating existing recommenders

Not every recommender can always make a recommendation. To evaluate the individual recommenders for the ensemble, we want to find out how often this is the case and how well the recommenders complement each other.

This notebook either needs to be executed in the [TAAR](http://github.com/mozilla/taar) repository or somewhere where TAAR is in the Python path, because some TAAR recommenders are loaded in.

## Retrieving the relevant variables from the longitudinal dataset


```python
%%time
frame = sqlContext.sql("""
WITH addons AS (
    SELECT client_id, feature_row.*
    FROM longitudinal
    LATERAL VIEW explode(active_addons[1]) feature_row
),
    
non_system_addons AS(
    SELECT client_id, collect_set(key) AS installed_addons
    FROM addons
    WHERE NOT value.is_system
    GROUP BY client_id
)

SELECT
    l.client_id,
    non_system_addons.installed_addons,
    l.settings[1].locale AS locale,
    l.geo_city[1] AS geoCity,
    subsession_length[1] AS subsessionLength,
    system_os[1].name AS os,
    scalar_parent_browser_engagement_total_uri_count[1].value AS total_uri,
    scalar_parent_browser_engagement_tab_open_event_count[1].value as tab_open_count,
    places_bookmarks_count[1].sum as bookmark_count,
    scalar_parent_browser_engagement_unique_domains_count[1].value as unique_tlds
FROM longitudinal l LEFT OUTER JOIN non_system_addons
ON l.client_id = non_system_addons.client_id
""")

rdd = frame.rdd
```

    CPU times: user 128 ms, sys: 28 ms, total: 156 ms
    Wall time: 18min 16s


## Loading addon data (AMO)

We need to load the addon database to find out which addons are legacy addons.


```python
from taar.recommenders.utils import get_s3_json_content
```


```python
AMO_DUMP_BUCKET = 'telemetry-parquet'
AMO_DUMP_KEY = 'telemetry-ml/addon_recommender/addons_database.json'
```


```python
amo_dump = get_s3_json_content(AMO_DUMP_BUCKET, AMO_DUMP_KEY)
```

## Filtering out legacy addons

This is a helper function that takes a list of addon IDs and only returns the IDs that are from legacy addons.


```python
def get_legacy_addons(installed_addons):
    legacy_addons = []
    
    for addon_id in installed_addons:
        if addon_id in amo_dump:
            addon = amo_dump[addon_id]
            addon_files = addon.get('current_version', {}).get('files', {})

            is_webextension = any([f.get("is_webextension", False) for f in addon_files])
            is_legacy = not is_webextension

            if is_legacy:
                legacy_addons.append(addon_id)
            
    return legacy_addons
```

## Completing client data


```python
def complete_client_data(client_data):
    client = client_data.asDict()
    
    client['installed_addons'] = client['installed_addons'] or []
    client['disabled_addon_ids'] = get_legacy_addons(client['installed_addons'])
    client['locale'] = str(client['locale'])
    
    return client
```

## Evaluating the existing recommenders

To check if a recommender is able to make a recommendation, it's sometimes easier and cleaner to directly query it instead of checking the important attributes ourselves. For example, this is the case for the locale recommender.


```python
from taar.recommenders import CollaborativeRecommender, LegacyRecommender, LocaleRecommender
```


```python
class DummySimilarityRecommender:
    def can_recommend(self, client_data):
        REQUIRED_FIELDS = ["geoCity", "subsessionLength", "locale", "os", "bookmark_count", "tab_open_count",
                           "total_uri", "unique_tlds"]

        has_fields = all([client_data.get(f, None) is not None for f in REQUIRED_FIELDS])
        return has_fields
```


```python
recommenders = {
    "collaborative": CollaborativeRecommender(),
    "legacy": LegacyRecommender(),
    "locale": LocaleRecommender(),
    "similarity": DummySimilarityRecommender()
}
```


```python
def test_recommenders(client):
    return tuple([recommender.can_recommend(client) for recommender in recommenders.values()])
```

## Computing combined counts

We iterate over all clients in the longitudinal dataset, change the attributes to the expected format and then query the individual recommenders.


```python
from operator import add
from collections import defaultdict
```


```python
%%time
results = rdd\
    .map(complete_client_data)\
    .map(test_recommenders)\
    .map(lambda x: (x, 1))\
    .reduceByKey(add)\
    .collect()
```

    CPU times: user 9.54 s, sys: 468 ms, total: 10 s
    Wall time: 11min 5s



```python
results = defaultdict(int, results)
```


```python
num_clients = sum(results.values())
```

## Computing individual counts


```python
individual_counts = []

for i in range(len(recommenders)):
    count = 0
    
    for key, key_count in results.items():
        if key[i]:
            count += key_count
            
    individual_counts.append(count)
```

## Displaying the results


```python
from pandas import DataFrame
```


```python
def format_int(num):
    return "{:,}".format(num)
```


```python
def format_frequency(frequency):
    return "%.5f" % frequency
```


```python
def get_relative_counts(counts, total=num_clients):
    return [format_frequency(count / float(total)) for count in counts]
```

This is a bit hacky. Sorting a data frame by formatted counts does not work; so we have to add the unformatted ones, sort the data frame, and then remove that column again.


```python
def sorted_dataframe(df, order, key="unformatted_counts"):
    df[key] = order
    return df.sort_values(by=key, ascending=False).drop(key, axis=1)
```

### Individual counts


```python
df = DataFrame(index=recommenders.keys(),
          columns=["Relative count"],
          data=get_relative_counts(individual_counts)
)

sorted_dataframe(df, individual_counts)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Relative count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>locale</th>
      <td>0.89227</td>
    </tr>
    <tr>
      <th>collaborative</th>
      <td>0.37663</td>
    </tr>
    <tr>
      <th>similarity</th>
      <td>0.27520</td>
    </tr>
    <tr>
      <th>legacy</th>
      <td>0.01470</td>
    </tr>
  </tbody>
</table>
</div>



$\implies$ The locale and collaborative recommenders are able to generate recommendations most of the time. The legacy recommender can only make recommendations very seldomly as not many users seem to have (legacy) addons installed.

### Combined counts

It's interesting to see how well the individual recommenders complement each other. In the following, we count how often different combinations of the recommenders can make recommendations.

The table is easier to read if cells are empty if a recommender is not available. If this is not desired, these variables can be changed:


```python
recommender_available_label = "Available"
recommender_unavailable_label = ""
```


```python
def format_labels(keys):
    return tuple([recommender_available_label if key else recommender_unavailable_label for key in keys])
```


```python
def format_data(keys, counts):
    formatted_keys = map(format_labels, keys)
    return [elems + (count,) for elems, count in zip(formatted_keys, *counts)]
```


```python
columns = recommenders.keys() + ["Relative counts"]

counts = get_relative_counts(results.values())
data = format_data(results.keys(), [counts])
```


```python
df = DataFrame(columns=columns, data=data)
sorted_dataframe(df, results.values())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Relative counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.38608</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.22323</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.13861</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.12965</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.10762</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00781</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00689</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00005</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



$\implies$ If any recommender is available, then the locale recommenders is generally also available. Other than that, there is a good chance the the collaborative recommender is available.
There is only a very small portion of cases where the similarity recommender was able to make a recommendation, when locale/collaborative were not; and not a single such case for the legacy recommender.

### Grouped by number of available recommenders


```python
from itertools import groupby
from operator import itemgetter
```


```python
from IPython.display import display, Markdown
```


```python
for num, group in groupby(sorted(results.keys(), key=sum), sum):
    display(Markdown("#### %d available recommender%s" % (num, "s" if num != 1 else "")))
    
    sub_keys = list(group)
    formatted_keys = map(format_labels, sub_keys)
    
    sub_counts = [results[key] for key in sub_keys]
    sub_counts_to_total = get_relative_counts(sub_counts)
    sub_counts_to_table = get_relative_counts(sub_counts, sum(sub_counts))
    
    zipped_data = zip(formatted_keys, sub_counts_to_total, sub_counts_to_table)
    data = [elems + (counts, table_counts) for elems, counts, table_counts in zipped_data]
    
    columns = recommenders.keys() + ["Relative to all", "Relative to this table"]
    
    df = DataFrame(columns=columns, data=data)
    df = sorted_dataframe(df, sub_counts)
    display(df)
```


#### 0 available recommenders



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Relative to all</th>
      <th>Relative to this table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.10762</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>



#### 1 available recommender



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Relative to all</th>
      <th>Relative to this table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.38608</td>
      <td>0.99979</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00005</td>
      <td>0.00014</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00003</td>
      <td>0.00007</td>
    </tr>
  </tbody>
</table>
</div>



#### 2 available recommenders



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Relative to all</th>
      <th>Relative to this table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.22323</td>
      <td>0.63255</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.12965</td>
      <td>0.36736</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00003</td>
      <td>0.00008</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



#### 3 available recommenders



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Relative to all</th>
      <th>Relative to this table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.13861</td>
      <td>0.94667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00781</td>
      <td>0.05332</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



#### 4 available recommenders



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Relative to all</th>
      <th>Relative to this table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00689</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>

