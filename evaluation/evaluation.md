
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

rdd = frame.rdd.cache()
```

    CPU times: user 12 ms, sys: 0 ns, total: 12 ms
    Wall time: 1min 14s


## Loading addon data (AMO)

We need to load the addon database to find out which addons are considered useful by TAAR.


```python
import boto3
import json
import logging

from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AMO_DUMP_BUCKET = 'telemetry-parquet'
AMO_DUMP_KEY = 'telemetry-ml/addon_recommender/addons_database.json'
```


```python
def load_amo_external_whitelist():
    """ Download and parse the AMO add-on whitelist.
    :raises RuntimeError: the AMO whitelist file cannot be downloaded or contains
                          no valid add-ons.
    """
    final_whitelist = []
    amo_dump = {}
    try:
        # Load the most current AMO dump JSON resource.
        s3 = boto3.client('s3')
        s3_contents = s3.get_object(Bucket=AMO_DUMP_BUCKET, Key=AMO_DUMP_KEY)
        amo_dump = json.loads(s3_contents['Body'].read())
    except ClientError:
        logger.exception("Failed to download from S3", extra={
            "bucket": AMO_DUMP_BUCKET,
            "key": AMO_DUMP_KEY})

    # If the load fails, we will have an empty whitelist, this may be problematic.
    for key, value in amo_dump.items():
        addon_files = value.get('current_version', {}).get('files', {})
        # If any of the addon files are web_extensions compatible, it can be recommended.
        if any([f.get("is_webextension", False) for f in addon_files]):
            final_whitelist.append(value['guid'])

    if len(final_whitelist) == 0:
        raise RuntimeError("Empty AMO whitelist detected")

    return final_whitelist
```


```python
whitelist = set(load_amo_external_whitelist())
```

    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com


## Filtering out legacy addons 

This is a helper function that takes a list of addon IDs and only returns the IDs of addons that are useful for TAAR.


```python
def get_whitelisted_addons(installed_addons):
    return whitelist.intersection(installed_addons)
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
    except ValueError: # raised when the date is in an unknown format
        return float("inf")
    
    days_ago = (datetime.today() - date).days
    return days_ago / 7
```


```python
def complete_client_data(client_data):
    client = client_data.asDict()
    
    addons = client['installed_addons'] or []
    client['installed_addons'] = get_whitelisted_addons(addons)
    client['disabled_addons_ids'] = addons
    
    client['locale'] = str(client['locale'])
    client['profile_age_in_weeks'] = compute_weeks_ago(client['profile_date'])
    client['submission_age_in_weeks'] = compute_weeks_ago(client['submission_date'])
    
    return client
```

## Evaluating the existing recommenders

To check if a recommender is able to make a recommendation, it's sometimes easier and cleaner to directly query it instead of checking the important attributes ourselves. For example, this is the case for the locale recommender.


```python
sc.addPyFile("./taar/dist/mozilla_taar-0.0.16.dev15+g824aa58.d20171018-py2.7.egg")
```


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
    "locale": LocaleRecommender("./top_addons_by_locale.json"),
    "similarity": DummySimilarityRecommender()
}
```

    INFO:requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:boto3.resources.action:Calling s3:get_object with {u'Bucket': 'telemetry-parquet', u'Key': 'taar/legacy/legacy_dict.json'}
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:boto3.resources.action:Calling s3:get_object with {u'Bucket': 'telemetry-parquet', u'Key': 'taar/locale/top10_dict.json'}
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com



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

    CPU times: user 1.3 s, sys: 116 ms, total: 1.41 s
    Wall time: 51.1 s



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
      <td>0.95706</td>
    </tr>
    <tr>
      <th>similarity</th>
      <td>0.28744</td>
    </tr>
    <tr>
      <th>collaborative</th>
      <td>0.08311</td>
    </tr>
    <tr>
      <th>legacy</th>
      <td>0.02739</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.62725</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.23407</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.03869</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.03081</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.03016</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00972</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00894</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00603</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00575</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00553</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00109</td>
    </tr>
    <tr>
      <th>14</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00082</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00045</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00027</td>
    </tr>
    <tr>
      <th>10</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00023</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00019</td>
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
      <td>0.03016</td>
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
      <td>0.62725</td>
      <td>0.98237</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00972</td>
      <td>0.01522</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00109</td>
      <td>0.00171</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00045</td>
      <td>0.00070</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.23407</td>
      <td>0.82705</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.03869</td>
      <td>0.13671</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00894</td>
      <td>0.03157</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00082</td>
      <td>0.00290</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00027</td>
      <td>0.00097</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00023</td>
      <td>0.00081</td>
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
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.03081</td>
      <td>0.72010</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00603</td>
      <td>0.14099</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00575</td>
      <td>0.13443</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00019</td>
      <td>0.00448</td>
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
      <td>0.00553</td>
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

    CPU times: user 5.86 s, sys: 388 ms, total: 6.25 s
    Wall time: 1min 3s


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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.77052</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.15120</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.02326</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01290</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00524</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00337</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00307</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00181</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00133</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00114</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00030</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00024</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00024</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00006</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.73305</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.18876</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.01970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00292</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00254</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00241</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00173</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00105</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00035</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00015</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00011</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00008</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00007</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.72011</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.19623</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.02061</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01699</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00717</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00295</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00291</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00283</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00212</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00058</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00043</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00024</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00016</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00014</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00011</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.70803</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.21223</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.02100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00685</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00310</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00272</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00260</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00212</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00061</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00046</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00027</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00024</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00006</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00006</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.77052</td>
      <td>0.73305</td>
      <td>0.72011</td>
      <td>0.70803</td>
      <td>0.62725</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.15120</td>
      <td>0.18876</td>
      <td>0.19623</td>
      <td>0.21223</td>
      <td>0.23407</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02525</td>
      <td>0.02495</td>
      <td>0.02642</td>
      <td>0.02468</td>
      <td>0.03016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.02326</td>
      <td>0.01970</td>
      <td>0.02061</td>
      <td>0.02100</td>
      <td>0.03869</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01290</td>
      <td>0.01497</td>
      <td>0.01699</td>
      <td>0.01497</td>
      <td>0.03081</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00524</td>
      <td>0.00715</td>
      <td>0.00717</td>
      <td>0.00685</td>
      <td>0.00972</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00337</td>
      <td>0.00292</td>
      <td>0.00295</td>
      <td>0.00310</td>
      <td>0.00894</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00307</td>
      <td>0.00241</td>
      <td>0.00283</td>
      <td>0.00272</td>
      <td>0.00575</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00181</td>
      <td>0.00254</td>
      <td>0.00291</td>
      <td>0.00260</td>
      <td>0.00553</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00133</td>
      <td>0.00173</td>
      <td>0.00212</td>
      <td>0.00212</td>
      <td>0.00603</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00114</td>
      <td>0.00105</td>
      <td>0.00058</td>
      <td>0.00061</td>
      <td>0.00109</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00030</td>
      <td>0.00035</td>
      <td>0.00043</td>
      <td>0.00046</td>
      <td>0.00082</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00024</td>
      <td>0.00011</td>
      <td>0.00011</td>
      <td>0.00006</td>
      <td>0.00019</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00024</td>
      <td>0.00015</td>
      <td>0.00024</td>
      <td>0.00027</td>
      <td>0.00045</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00006</td>
      <td>0.00007</td>
      <td>0.00016</td>
      <td>0.00006</td>
      <td>0.00023</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00006</td>
      <td>0.00008</td>
      <td>0.00014</td>
      <td>0.00024</td>
      <td>0.00027</td>
    </tr>
  </tbody>
</table>
</div>


### By submission date in weeks


```python
%time conditioned_results_submission_date = get_conditioned_results("submission_age_in_weeks", conditions)
```

    CPU times: user 5.11 s, sys: 220 ms, total: 5.33 s
    Wall time: 1min 10s



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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.42921</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.37171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.06815</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.05122</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.01944</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.01649</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01133</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.01046</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00926</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00783</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00180</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00154</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00048</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00043</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00038</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00026</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.58081</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.27560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.04266</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.03264</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02649</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.01156</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00916</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00623</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00618</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00537</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00123</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00085</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00042</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00031</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00028</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00021</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.65590</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.21726</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.03932</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.02249</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00951</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00862</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00584</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00520</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00414</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00113</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00059</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00045</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00024</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00020</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00018</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.68416</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.19886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.03622</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.02901</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01904</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00935</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.00722</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00516</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00472</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00376</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00104</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00050</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00047</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00020</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00018</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00011</td>
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
      <th>2</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.42921</td>
      <td>0.58081</td>
      <td>0.65590</td>
      <td>0.68416</td>
      <td>0.62725</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.37171</td>
      <td>0.27560</td>
      <td>0.21726</td>
      <td>0.19886</td>
      <td>0.23407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.06815</td>
      <td>0.03264</td>
      <td>0.02249</td>
      <td>0.01904</td>
      <td>0.03081</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.05122</td>
      <td>0.04266</td>
      <td>0.03932</td>
      <td>0.03622</td>
      <td>0.03869</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.01944</td>
      <td>0.02649</td>
      <td>0.02892</td>
      <td>0.02901</td>
      <td>0.03016</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>0.01649</td>
      <td>0.01156</td>
      <td>0.00862</td>
      <td>0.00722</td>
      <td>0.00972</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.01133</td>
      <td>0.00537</td>
      <td>0.00414</td>
      <td>0.00376</td>
      <td>0.00553</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.01046</td>
      <td>0.00623</td>
      <td>0.00520</td>
      <td>0.00472</td>
      <td>0.00603</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00926</td>
      <td>0.00916</td>
      <td>0.00951</td>
      <td>0.00935</td>
      <td>0.00894</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00783</td>
      <td>0.00618</td>
      <td>0.00584</td>
      <td>0.00516</td>
      <td>0.00575</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00180</td>
      <td>0.00085</td>
      <td>0.00059</td>
      <td>0.00050</td>
      <td>0.00082</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>0.00154</td>
      <td>0.00123</td>
      <td>0.00113</td>
      <td>0.00104</td>
      <td>0.00109</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td>Available</td>
      <td>0.00048</td>
      <td>0.00031</td>
      <td>0.00024</td>
      <td>0.00020</td>
      <td>0.00027</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>Available</td>
      <td></td>
      <td></td>
      <td>0.00043</td>
      <td>0.00042</td>
      <td>0.00045</td>
      <td>0.00047</td>
      <td>0.00045</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td>Available</td>
      <td>0.00038</td>
      <td>0.00021</td>
      <td>0.00018</td>
      <td>0.00011</td>
      <td>0.00019</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Available</td>
      <td>Available</td>
      <td></td>
      <td>0.00026</td>
      <td>0.00028</td>
      <td>0.00020</td>
      <td>0.00018</td>
      <td>0.00023</td>
    </tr>
  </tbody>
</table>
</div>


## Addon counts

We want to train an ensemble model using the individual recommenders that we already have. To optimize this ensemble model, we need some training data. We have information about what addons different users have installed, so it would make sense to check if our ensemble model would also recommend these addons to the respective users.

However, there is one fundamental conflict here: To be able to make recommendations, the collaborative recommender already needs some information about which addons a user has installed. Thus, we can only use a subset of a user's installed addons for evaluation. These addons are masked and the collaborative recommender then only uses the unmasked addons. The open question is how large this subset of masked addons should be.

Choosing this size is a trade-off between three factors:
1. By masking more addons, we give the evaluation function more data to work with
2. If we mask fewer addons, the collaborative filter can make better recommendations
3. There are not that many users who have many addons installed. This means we need to be careful not to make our evaluation set too biased. For example, if we always mask at least five addons, then only users with more than six addons could be part of the evaluation set, which is a small subset of the entire population

To be able to make this decision, it's helpful to look at the distribution of the number of addons that users have installed.


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="darkgrid")
```

    /mnt/anaconda2/lib/python2.7/site-packages/matplotlib/__init__.py:878: UserWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      warnings.warn(self.msg_depr % (key, alt_key))



```python
addon_counts = rdd_completed\
    .map(lambda client: len(client['installed_addons']))\
    .map(lambda x: (x, 1))\
    .reduceByKey(add)\
    .collect()
    
addon_counts = sorted(addon_counts, key=itemgetter(0))
```


```python
def addon_counts_to_proportions(addon_counts):
    num_addons, num_addons_count = zip(*addon_counts)
    
    num_addons_total = float(sum(num_addons_count))
    num_addons_count = np.array(num_addons_count) / num_addons_total
    
    return num_addons, num_addons_count
```


```python
def plot_addon_distribution(addon_counts):
    num_addons, num_addons_count = addon_counts_to_proportions(addon_counts)
    
    plt.bar(num_addons, num_addons_count, width=1.)
    plt.title("Number of addons per user")
    plt.xlabel("Number of addons")
    plt.ylabel("Proportion of users")
    plt.show()
    
    plt.plot(np.cumsum(num_addons_count))
    plt.title("Number of addons per user (cumulative)")
    plt.xlabel("Number of addons")
    plt.ylabel("Proportion of users")
    plt.ylim(0, 1)
    plt.show()
```


```python
plot_addon_distribution(addon_counts)
```


![png](output_72_0.png)



![png](output_72_1.png)


As we can see, there is a substantial number of users that have no addons installed. Nearly all other users have fewer than 10 addons installed. To make the plot a bit easier to read, it's helpful to hide the long tail.


```python
reasonable_addon_counts = filter(lambda (num_addons, count): num_addons < 15, addon_counts)
plot_addon_distribution(reasonable_addon_counts)
```


![png](output_74_0.png)



![png](output_74_1.png)


$\implies$ To get an evaluation set of a decent size that's at least partly representative, it's important to include users that only have a few addons installed. One idea could be to mask half of the addons that users in the evaluation set have installed. This way, we would be able to include users that only have a few addons installed. If the number of addons is odd, we could uniformly randomly round up or down.

The alternative would be to make the portion of masked addons a little bit higher or lower. This could make sense if we notice that the collaborative recommender is generally able to make much better recommendations using two addons than using one addon; or, the other way around, if we notice that it adds little value to evaluate based on a single addon.


```python
num_addons, num_addons_count = addon_counts_to_proportions(addon_counts)
num_addons_count = map(format_frequency, num_addons_count)

DataFrame(
    columns=["Number of addons", "Proportion"],
    data=zip(num_addons, num_addons_count)
).head(40)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number of addons</th>
      <th>Proportion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.91689</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.06219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.01440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.00386</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.00142</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.00062</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.00028</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.00014</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.00008</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.00004</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>27</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>28</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>31</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>36</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>45</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>


