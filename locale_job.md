
# TAAR – Install numbers by locale

This notebook mostly contains code from the [generator job](https://github.com/mozilla/python_mozetl/blob/master/mozetl/taar/taar_locale.py) for finding the most installed addons by locale. It is adapted to include the actual install numbers, normalized by each locale. The result is a dictionary where the keys are locales. The values are dictionaries that contain (addon_id, relative_install_number) pairs.

Because this generation process takes quite some time, the results are saved to a JSON file.


```python
OUTPUT_PATH = 'top_addons_by_locale.json'
```

## Loading the whitelist


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

## Loading the addon data (locale generator job)


```python
import click
import json
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCALE_FILE_NAME = 'top10_dict'


def get_addons(spark):
    """ Longitudinal sample is selected and freshest ping chosen per client.
    Only Firefox release clients are considered.
    Columns are exploded (over addon keys)  to include locale of each addon
    installation instance system addons, disabled addons, unsigned addons
    are filtered out.
    Sorting by addon-installations and grouped by locale.
    """
    return spark.sql("""
        WITH sample AS (
        SELECT client_id,
        settings[0].locale AS locality,
        EXPLODE(active_addons[0])
        FROM longitudinal
        WHERE normalized_channel='release'
          AND build IS NOT NULL
          AND build[0].application_name='Firefox'
        ),
        filtered_sample AS (
        SELECT locality, key AS addon_key FROM sample
        WHERE value['blocklisted'] = FALSE -- not blocklisted
          AND value['type'] = 'extension' -- nice webextensions only
          AND value['signed_state'] = 2 -- fully reviewed addons only
          AND value['user_disabled'] = FALSE -- active addons only get counted
          AND value['app_disabled'] = FALSE -- exclude compatibility disabled addons
          AND value['is_system'] = FALSE -- exclude system addons
          AND locality <> 'null'
          AND key is not null
        ),
        country_addon_pairs AS (
        SELECT
        COUNT(*) AS pair_cnts, addon_key, locality
        from filtered_sample
        GROUP BY locality, addon_key
        )
        SELECT * FROM country_addon_pairs
        ORDER BY locality, pair_cnts DESC
    """)


def compute_threshold(addon_df):
    """ Get a threshold to remove locales with a small
    number of addons installations.
    """
    addon_install_counts = (
        addon_df
        .groupBy('locality')
        .agg({'pair_cnts': 'sum'})
    )

    # Compute a threshold at the 25th percentile to remove locales with a
    # small number of addons installations.
    locale_pop_threshold =\
        addon_install_counts.approxQuantile('sum(pair_cnts)', [0.25], 0.2)[0]

    # Safety net in case the distribution gets really skewed, we should
    # require 2000 addon installation instances to make recommendations.
    return 2000 if locale_pop_threshold < 2000 else locale_pop_threshold


def transform(addon_df, threshold):
    """ Converts the locale-specific addon data in to a dictionary.
    :param addon_df: the locale-specific addon dataframe;
    :param threshold: the minimum number of addon-installs per locale;
    :param num_addons: requested number of recommendations.
    :return: a dictionary {<locale>: ['GUID1', 'GUID2', ...]}
    """
    top10_per = {}

    # Decide that we can not make reasonable recommendations without
    # a minimum number of addon installations.
    grouped_addons = (
        addon_df
        .groupBy('locality')
        .agg({'pair_cnts': 'sum'})
        .collect()
    )
    list_of_locales =\
        [i['locality'] for i in grouped_addons if i['sum(pair_cnts)'] > threshold]

    for specific_locale in list_of_locales:
        # Most popular addons per locale sorted by number of installs
        # are added to the list.
        sorted_addon_guids = (
            addon_df
            .filter(addon_df.locality == specific_locale)
            .sort(addon_df.pair_cnts.desc())
            .collect()
        )FI

        # Creates a dictionary of locales (keys) and list of
        # recommendation GUIDS (values).
        top10_per[specific_locale] = sorted_addon_guids

    return top10_per


def generate_dictionary(spark):
    """ Wrap the dictionary generation functions in an
    easily testable way.
    """
    # Execute spark.SQL query to get fresh addons from longitudinal telemetry data.
    addon_df = get_addons(spark)

    # Load external whitelist based on AMO data.
    amo_whitelist = load_amo_external_whitelist()

    # Filter to include only addons present in AMO whitelist.
    addon_df_filtered = addon_df.where(col("addon_key").isin(amo_whitelist))

    # Make sure not to include addons from very small locales.
    locale_pop_threshold = compute_threshold(addon_df_filtered)
    return transform(addon_df_filtered, locale_pop_threshold)
```


```python
locale_dict = generate_dictionary(sqlContext)
```

    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com


## Computing relative install numbers


```python
result = {}
```


```python
for locale, addons in locale_dict.items():
    max_count = float(max([addon['pair_cnts'] for addon in addons]))
    result[locale] = { addon['addon_key']: addon['pair_cnts'] / max_count for addon in addons }
```

## Storing the result in JSON


```python
import json
```


```python
with open(OUTPUT_PATH, 'w') as outfile:
    json.dump(result, outfile)
```
