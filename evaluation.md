
# TAAR – Evaluating existing recommenders

Not every recommender can always make a recommendation. To evaluate the individual recommenders for the ensemble, we want to find out how often this is the case and how well the recommenders complement each other.

This notebook either needs to be executed in the [TAAR](http://github.com/mozilla/taar) repository or somewhere where TAAR is in the Python path, because some TAAR recommenders are loaded in.

## Retrieving the relevant variables from the longitudinal dataset


```python
%%time
frame = sqlContext.sql("""
WITH valid_clients AS (
    SELECT *
    FROM longitudinal
    WHERE normalized_channel='release' AND build IS NOT NULL AND build[0].application_name='Firefox'
),

addons AS (
    SELECT client_id, feature_row.*
    FROM valid_clients
    LATERAL VIEW explode(active_addons[0]) feature_row
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
    settings[0].locale AS locale,
    geo_city[0] AS geoCity,
    subsession_length[0] AS subsessionLength,
    system_os[0].name AS os,
    scalar_parent_browser_engagement_total_uri_count[0].value AS total_uri,
    scalar_parent_browser_engagement_tab_open_event_count[0].value as tab_open_count,
    places_bookmarks_count[0].sum as bookmark_count,
    scalar_parent_browser_engagement_unique_domains_count[0].value as unique_tlds,
    profile_creation_date[0] as profile_date,
    submission_date[0] as submission_date
FROM valid_clients l LEFT OUTER JOIN non_system_addons
ON l.client_id = non_system_addons.client_id
""")

rdd = frame.rdd
```

    CPU times: user 140 ms, sys: 12 ms, total: 152 ms
    Wall time: 18min 43s


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
from dateutil.parser import parse as parse_date
from datetime import datetime
```


```python
def compute_weeks_ago(formatted_date):
    try:
        date = parse_date(formatted_date).replace(tzinfo=None)
        days_ago = (datetime.today() - date).days
        return days_ago / 7
    except:
        return float("inf")
```


```python
def complete_client_data(client_data):
    client = client_data.asDict()
    
    client['installed_addons'] = client['installed_addons'] or []
    client['disabled_addon_ids'] = get_legacy_addons(client['installed_addons'])
    client['locale'] = str(client['locale'])
    client['profile_age_in_weeks'] = compute_weeks_ago(client['profile_date'])
    client['submission_age_in_weeks'] = compute_weeks_ago(client['submission_date'])
    
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
rdd_completed = rdd.map(complete_client_data)
```


```python
def analyse(rdd):
    results = rdd\
        .map(test_recommenders)\
        .map(lambda x: (x, 1))\
        .reduceByKey(add)\
        .collect()
        
    return defaultdict(int, results)
```


```python
%time results = analyse(rdd_completed)
```

    CPU times: user 9.71 s, sys: 404 ms, total: 10.1 s
    Wall time: 13min 10s



```python
num_clients = sum(results.values())
total_results = results
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
      <td>0.99977</td>
    </tr>
    <tr>
      <th>collaborative</th>
      <td>0.41949</td>
    </tr>
    <tr>
      <th>similarity</th>
      <td>0.28339</td>
    </tr>
    <tr>
      <th>legacy</th>
      <td>0.01575</td>
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
    return [elems + count for elems, count in zip(formatted_keys, zip(*counts))]
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
      <td>0.44747</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.26032</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.14333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.13290</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00865</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00710</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00011</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00006</td>
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
      <td>0.00011</td>
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
      <td>0.44747</td>
      <td>0.99980</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00006</td>
      <td>0.00014</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00003</td>
      <td>0.00006</td>
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
      <td>0.26032</td>
      <td>0.66197</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.13290</td>
      <td>0.33795</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00003</td>
      <td>0.00007</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
      <td>0.00000</td>
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
      <td>0.14333</td>
      <td>0.94310</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00865</td>
      <td>0.05690</td>
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
      <td>0.00710</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>


## By dates

In this section, we perform a similar analysis as before but on subsets of the data. These subsets are specified by when the client profile was generated. `conditions` is a list that contains ranges for the profile age in weeks. The end of the range is exclusive, similar to ranges in Python's standard library.


```python
conditions = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4)
]
```


```python
import numpy as np
from numpy import argsort
from itertools import product
```


```python
def attribute_between(attr, min_weeks, max_weeks):
    return lambda client: min_weeks <= client[attr] < max_weeks
```


```python
def get_conditioned_results(attr, conditions):
    conditioned_results = {}

    for (min_weeks, max_weeks) in conditions:
        sub_rdd = rdd_completed.filter(attribute_between(attr, min_weeks, max_weeks))
        conditioned_results[(min_weeks, max_weeks)] = analyse(sub_rdd)
        
    return conditioned_results
```

### By profile age in weeks


```python
%time conditioned_results = get_conditioned_results("profile_age_in_weeks", conditions)
```

    CPU times: user 39.1 s, sys: 1.49 s, total: 40.6 s
    Wall time: 52min 19s


To make things a little bit easier to read, only recommender combinations that actually appear are displayed in the table.


```python
def nonzero_combinations(conditioned_results):
    combinations = []

    for sub_result in conditioned_results.values():
        combinations += [key for key, value in sub_result.items() if value > 0]

    return set(combinations)
```


```python
combinations = nonzero_combinations(conditioned_results)
```


```python
def display_individual_filtered_results(conditioned_results, combinations, label):
    display(Markdown("### Filtering on the %s, Python-like exclusive ranges" % label))

    counts = []
    titles = []

    columns = recommenders.keys() + ["Relative counts"]

    for key in conditions:
        sub_results = conditioned_results[key]
        values = [sub_results[sub_key] for sub_key in combinations]
        summed = sum(values)

        sub_counts = get_relative_counts(values, summed)
        data = format_data(combinations, [sub_counts])
        counts.append(sub_counts)

        title = "Between %d and %d weeks" % key
        titles.append(title)
        display(Markdown("#### %s" % title))

        df = DataFrame(columns=columns, data=data)
        df = sorted_dataframe(df, values)
        display(df)

    return counts, titles
```


```python
counts, titles = display_individual_filtered_results(conditioned_results, combinations, label="profile age")
```


### Filtering on the profile age, Python-like exclusive ranges



#### Between 0 and 1 weeks



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
      <td>0.56426</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.24668</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.11291</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.06858</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00447</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00304</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>2</th>
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
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



#### Between 1 and 2 weeks



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
      <td>0.55178</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.21705</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.14102</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.08284</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00407</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00307</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00008</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00008</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
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
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



#### Between 2 and 3 weeks



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
      <td>0.52717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.22324</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.14923</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.09279</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00344</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00003</td>
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
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
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



#### Between 3 and 4 weeks



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
      <td>0.52334</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.22640</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.14447</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.09773</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00421</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00375</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00002</td>
    </tr>
    <tr>
      <th>2</th>
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
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>


To make things a little bit easier to read, we can display all results in a single table.


```python
def display_merged_filtered_results(counts, titles, total_results, combinations, label):
    values = [total_results[sub_key] for sub_key in combinations]
    sub_counts = get_relative_counts(values)
    counts.append(sub_counts)
    titles.append("Total, without any condition")  

    columns = recommenders.keys() + titles
    data = format_data(combinations, counts)

    df = DataFrame(columns=columns, data=data)
    df = sorted_dataframe(df, counts[0])

    display(Markdown("### Filtering on the %s, Python-like exclusive ranges – All in one table" % label))
    display(df)
```


```python
display_merged_filtered_results(counts, titles, total_results, combinations, label="profile age")
```


### Filtering on the profile age, Python-like exclusive ranges – All in one table



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Between 0 and 1 weeks</th>
      <th>Between 1 and 2 weeks</th>
      <th>Between 2 and 3 weeks</th>
      <th>Between 3 and 4 weeks</th>
      <th>Total, without any condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.56426</td>
      <td>0.55178</td>
      <td>0.52717</td>
      <td>0.52334</td>
      <td>0.44747</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.24668</td>
      <td>0.21705</td>
      <td>0.22324</td>
      <td>0.22640</td>
      <td>0.26032</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.11291</td>
      <td>0.14102</td>
      <td>0.14923</td>
      <td>0.14447</td>
      <td>0.13290</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.06858</td>
      <td>0.08284</td>
      <td>0.09279</td>
      <td>0.09773</td>
      <td>0.14333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00447</td>
      <td>0.00407</td>
      <td>0.00400</td>
      <td>0.00421</td>
      <td>0.00865</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00304</td>
      <td>0.00307</td>
      <td>0.00344</td>
      <td>0.00375</td>
      <td>0.00710</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00006</td>
      <td>0.00008</td>
      <td>0.00006</td>
      <td>0.00006</td>
      <td>0.00011</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00001</td>
      <td>0.00000</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
      <td>0.00008</td>
      <td>0.00003</td>
      <td>0.00003</td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
      <td>0.00001</td>
      <td>0.00003</td>
      <td>0.00002</td>
      <td>0.00003</td>
    </tr>
  </tbody>
</table>
</div>


### By submission date in weeks


```python
%time conditioned_results_submission_date = get_conditioned_results("submission_age_in_weeks", conditions)
```


```python
label = "submission date"
combinations = nonzero_combinations(conditioned_results_submission_date)
counts, titles = display_individual_filtered_results(conditioned_results_submission_date, combinations, label)
display_merged_filtered_results(counts, titles, total_results, combinations, label)
```


### Filtering on the submission date, Python-like exclusive ranges



#### Between 0 and 1 weeks



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
      <td>0.28043</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.25749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.23062</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.20527</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01489</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.01114</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00005</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00004</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00003</td>
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
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



#### Between 1 and 2 weeks



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
      <td>0.41392</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.25433</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.15828</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.15660</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00968</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00701</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00009</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00004</td>
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
      <th>2</th>
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



#### Between 2 and 3 weeks



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
      <td>0.46368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.26353</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.13141</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.12639</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00911</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00568</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00011</td>
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
      <td>0.00002</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00002</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



#### Between 3 and 4 weeks



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
      <td>0.48052</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.26312</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.12283</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.11945</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00870</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00514</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00015</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
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
      <th>9</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00002</td>
    </tr>
    <tr>
      <th>2</th>
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



### Filtering on the submission date, Python-like exclusive ranges – All in one table



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locale</th>
      <th>legacy</th>
      <th>collaborative</th>
      <th>similarity</th>
      <th>Between 0 and 1 weeks</th>
      <th>Between 1 and 2 weeks</th>
      <th>Between 2 and 3 weeks</th>
      <th>Between 3 and 4 weeks</th>
      <th>Total, without any condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.28043</td>
      <td>0.41392</td>
      <td>0.46368</td>
      <td>0.48052</td>
      <td>0.44747</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.25749</td>
      <td>0.15660</td>
      <td>0.12639</td>
      <td>0.11945</td>
      <td>0.14333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.23062</td>
      <td>0.25433</td>
      <td>0.26353</td>
      <td>0.26312</td>
      <td>0.26032</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.20527</td>
      <td>0.15828</td>
      <td>0.13141</td>
      <td>0.12283</td>
      <td>0.13290</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01489</td>
      <td>0.00701</td>
      <td>0.00568</td>
      <td>0.00514</td>
      <td>0.00710</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.01114</td>
      <td>0.00968</td>
      <td>0.00911</td>
      <td>0.00870</td>
      <td>0.00865</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.00005</td>
      <td>0.00009</td>
      <td>0.00011</td>
      <td>0.00015</td>
      <td>0.00011</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00004</td>
      <td>0.00003</td>
      <td>0.00002</td>
      <td>0.00003</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00003</td>
      <td>0.00004</td>
      <td>0.00005</td>
      <td>0.00003</td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00003</td>
      <td>0.00003</td>
      <td>0.00002</td>
      <td>0.00002</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>

