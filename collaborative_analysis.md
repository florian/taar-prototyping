
# TAAR – Analysing the collaborative filter

## Retrieving the data

We're only checking how many users have addons installed, so this the only attribute we need to retrieve.


```python
frame = sqlContext.sql("""
SELECT map_keys(active_addons[0]) as installed_addons
FROM longitudinal
WHERE normalized_channel='release' AND build IS NOT NULL AND build[0].application_name='Firefox'
""")

rdd = frame.rdd
```


```python
total_count = float(rdd.count())
```

## Handling the case where no information is available

If `installed_addons` is `None`, then we want to default to an empty list to clean the data up a little bit.


```python
def complete_client(client):
    client = client.asDict()
    client['installed_addons'] = client['installed_addons'] or []
    return client
```


```python
rdd_completed = rdd.map(complete_client)
```

## Counting what portion of users have any addons installed


```python
rdd_completed.filter(lambda client: len(client['installed_addons']) > 0).count() / total_count
```




    0.9566861553945836



## Counting what portion of users have whitelisted addons installed

### Loading the whitelist

This function is copied over from the [TAAR utils](https://github.com/mozilla/python_mozetl/blob/master/mozetl/taar/taar_utils.py#L56) in python_mozetl.


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

To allow efficient access, we're converting the list into a set.


```python
whitelist = set(load_amo_external_whitelist())
```

    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com


To filter for addons that are in the whitelist, we then simply take the intersection with the whitelist.


```python
def whitelist_filter(installed_addons):
    return whitelist.intersection(installed_addons)
```

### Counting


```python
rdd_completed.filter(lambda client: len(whitelist_filter(client['installed_addons'])) > 0).count() / total_count
```




    0.08347842005727894


