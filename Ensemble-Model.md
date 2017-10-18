
# TAAR – Ensemble Learning

Currently we have four different addon recommenders in TAAR. The goal of this project is to combine them into one model. In Machine Learning, this is known as an ensemble model. Generally, ensembles work well if the individual models are quite diverse. There is a good chance that this is the case here as our individual models are all based on different features.

The method that we're trying to implement here is known as *linear stacking* or *linear blending*. Our ensemble model uses the output of the previous models as features and learns to weight them, i.e. it learns how useful the individual recommenders are in general. To do this, we have to extend the existing recommenders so that they're able to return weighted recommendations (meaning pairs of recommendations and confidence scores, instead of just an ordered list).


```python
sc.addPyFile("./taar/dist/mozilla_taar-0.0.16.dev15+g824aa58.d20171018-py2.7.egg")
```

## Collecting and preprocessing the data

### Retrieving the relevant variables from the longitudinal dataset


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
    geo_city[0] AS geo_city,
    subsession_length[0] AS subsession_length,
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

    CPU times: user 20 ms, sys: 4 ms, total: 24 ms
    Wall time: 2min 29s



```python
rdd = rdd.cache()
```

## RDD partitions

For some reason, Spark doesn't correctly adjust the number of partitions in our RDD here. This makes it extremely slow to process the data because we're only using one core instead of `number of cores`  (typically 16 here) times `number of nodes`. Because of this, we manually repartition the RDD here. This will dramatically speed things up when having a cluster with multiple nodes.


```python
rdd.getNumPartitions()
```




    2166




```python
rdd = frame.rdd.repartition(sc.defaultParallelism)
```


```python
rdd.getNumPartitions()
```




    320



### Loading addon data (AMO)


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

### Filtering out legacy addons

The collaborative recommender should make recommendations based on WebExtensions. In general, our ensemble should also only recommend WebExtensions. Because of this, we need a way to filter for WebExtensions. We store this collection in two different data structures (list, set) as both will be useful in the remainder of the notebook.


```python
whitelist = load_amo_external_whitelist()
whiteset = set(whitelist)
```

    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTP connection (1): 169.254.169.254
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com



```python
def get_whitelisted_addons(installed_addons):
    return whiteset.intersection(installed_addons)
```

### Completing client data


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


```python
completed_rdd = rdd.map(complete_client_data).cache()
```

## Computing confidence scores

We add a method `get_weighted_recommendations` to each recommender. This method returns a (default) dictionary where the keys are addons and the values indicate how confident the recommender is that the respective addon would be a good recommendation.

A default dictionary is used to return `0 ` if a recommender is not in a position to judge how good a potential recommendation would be. The scores returned by this method do not need to be normalized. This is done one step afterwards.

**Important note**: The code below is not directly used in the rest of the notebook, it's just here for explanatory and documentation reasons. To use the adapted classes on the worker nodes, they need to be in the TAAR egg, because pickle seems to have problems with subclassing here. Thus, the code below is copied into the TAAR folder from where the TAAR egg is produced.


```python
from collections import defaultdict
from operator import itemgetter
```

### Collaborative Recommender

Use the scores computed internally


```python
from taar.recommenders import CollaborativeRecommender
```


```python
import numpy as np
import operator as op
```


```python
def java_string_hashcode(s):
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def positive_hash(s):
    return java_string_hashcode(s) & 0x7FFFFF
```


```python
class NewCollaborativeRecommender(CollaborativeRecommender):
    def recommend(self, client_data, limit):
        recommendations = self.get_weighted_recommendations(client_data)
        
        # Sort the suggested addons by their score and return the sorted list of addon
        # ids.
        sorted_dists = sorted(recommendations.items(), key=op.itemgetter(1), reverse=True)
        return [s[0] for s in sorted_dists[:limit]]
    
    def get_weighted_recommendations(self, client_data):
        # Addons identifiers are stored as positive hash values within the model.
        installed_addons =\
            [positive_hash(addon_id) for addon_id in client_data.get('installed_addons', [])]

        # Build the query vector by setting the position of the queried addons to 1.0
        # and the other to 0.0.
        query_vector = np.array([1.0 if (entry.get("id") in installed_addons) else 0.0
                                 for entry in self.raw_item_matrix])

        # Build the user factors matrix.
        user_factors = np.matmul(query_vector, self.model)
        user_factors_transposed = np.transpose(user_factors)

        # Compute the distance between the user and all the addons in the latent
        # space.
        distances = {}
        for addon in self.raw_item_matrix:
            # We don't really need to show the items we requested. They will always
            # end up with the greatest score. Also filter out legacy addons from the
            # suggestions.
            hashed_id = str(addon.get("id"))
            if (hashed_id in installed_addons or
                    hashed_id not in self.addon_mapping or
                    self.addon_mapping[hashed_id].get("isWebextension", False) is False):
                continue

            dist = np.dot(user_factors_transposed, addon.get('features'))
            # Read the addon ids from the "addon_mapping" looking it
            # up by 'id' (which is an hashed value).
            addon_id = self.addon_mapping[hashed_id].get("id")
            distances[addon_id] = dist

        return defaultdict(int, distances)
```

### Similarity Recommender

Use similarity scores computed internally

TODO: Compute logarithm


```python
from taar.recommenders.similarity_recommender import SimilarityRecommender
```


```python
from scipy.spatial.distance import hamming, canberra
```


```python
def cdist(dist, A, b):
    return np.array([dist(a, b) for a in A])
```


```python
CATEGORICAL_FEATURES = ["geo_city", "locale", "os"]
CONTINUOUS_FEATURES = ["subsession_length", "bookmark_count", "tab_open_count", "total_uri", "unique_tlds"]

class NewSimilarityRecommender(SimilarityRecommender):
    def get_similar_donors(self, client_data):
        """Computes a set of :float: similarity scores between a client and a set of candidate
        donors for which comparable variables have been measured.
        A custom similarity metric is defined in this function that combines the Hamming distance
        for categorical variables with the Canberra distance for continuous variables into a
        univariate similarity metric between the client and a set of candidate donors loaded during
        init.
        :param client_data: a client data payload including a subset fo telemetry fields.
        :return: the sorted approximate likelihood ratio (np.array) corresponding to the
                 internally computed similarity score and a list of indices that link
                 each LR score with the related donor in the |self.donors_pool|.
        """
        client_categorical_feats = [client_data.get(specified_key) for specified_key in CATEGORICAL_FEATURES]
        client_continuous_feats = [client_data.get(specified_key) for specified_key in CONTINUOUS_FEATURES]

        # Compute the distances between the user and the cached continuous
        # and categorical features.
        cont_features = cdist(canberra, self.continuous_features, client_continuous_feats)
        
        # The lambda trick is needed to prevent |cdist| from force-casting the
        # string features to double.
        cat_features = cdist(hamming, self.categorical_features, client_categorical_feats)

        # Take the product of similarities to attain a univariate similarity score.
        # Addition of 0.001 to the continuous features avoids a zero value from the
        # categorical variables, allowing categorical features precedence.
        distances = (cont_features + 0.001) * cat_features

        # Compute the LR based on precomputed distributions that relate the score
        # to a probability of providing good addon recommendations.
        lrs_from_scores =\
            np.array([self.get_lr(distances[i]) for i in range(self.num_donors)])

        # Sort the LR values (descending) and return the sorted values together with
        # the original indices.
        indices = (-lrs_from_scores).argsort()
        return lrs_from_scores[indices], indices

    def get_weighted_recommendations(self, client_data):
        recommendations = defaultdict(int)

        for donor_score, donor in zip(*self.get_similar_donors(client_data)):
            for addon in self.donors_pool[donor]['active_addons']:
                recommendations[addon] += donor_score
        
        return recommendations
```

### Locale Recommender

Depends on number of installs in that locale


```python
from taar.recommenders import LocaleRecommender
```


```python
TOP_ADDONS_BY_LOCALE_FILE_PATH = "top_addons_by_locale.json"
```


```python
class NewLocaleRecommender(LocaleRecommender):
    def __init__(self, TOP_ADDONS_BY_LOCALE_FILE_PATH):
        OriginalLocaleRecommender.__init__(self)
        
        with open(TOP_ADDONS_BY_LOCALE_FILE_PATH) as data_file:
            top_addons_by_locale = json.load(data_file)
            
        self.top_addons_by_locale = defaultdict(lambda: defaultdict(int), top_addons_by_locale)
        
    def get_weighted_recommendations(self, client_data):
        client_locale = client_data.get('locale', None)
        return defaultdict(int, self.top_addons_by_locale[client_locale])
```

### Legacy Recommender

1 for all replacement addons, 0 otherwise


```python
from taar.recommenders import LegacyRecommender
```


```python
class NewLegacyRecommender(LegacyRecommender):
    def get_weighted_recommendations(self, client_data):
        recommendations = defaultdict(int)
        addons = client_data.get('disabled_addons_ids', [])
        
        for addon in addons:
            for replacement in self.legacy_replacements.get(addon, []):
                recommendations[replacement] += 1
                
        return recommendations
```

## Choosing training, validation and test sets

For training and validation purposes, only clients that have WebExtension addons installed are useful.


```python
useful_clients = completed_rdd.filter(lambda client: len(client['installed_addons']) >= 1).cache()
```


```python
useful_clients.count()
```




    435313



These users are useful for training and evaluating our model:


```python
training, validation, test = useful_clients.randomSplit([0.8, 0.1, 0.1])
```

## Masking addons

First, we'll introduce a small helper function `random_partition`. It takes in an iterable `A` that should be partitioned into to new lists where the first list has a length of `k`. This partitioning is done randomly.


```python
from random import sample
```


```python
def random_partition(A, k):
    n = len(A)
    A = list(A)
    indices = set(sample(range(n), k))
    
    first = []
    second = []
    
    for i in range(n):
        element = A[i]
        
        if i in indices:
            first.append(element)
        else:
            second.append(element)
            
    return first, second
```

Next, we can use this function to randomly decide on a subset of addons that we want to mask for a user.


```python
def get_num_masked(addons):
    return max(1, len(addons) / 2)
```


```python
def mask_addons(client):
    addons = client['installed_addons']
    num_mask = get_num_masked(addons)
    
    masked, unmasked = random_partition(addons, num_mask)
    
    client['installed_addons'] = unmasked
    client['masked_addons'] = masked
    
    return client
```


```python
training_masked = training.map(mask_addons).cache()
```

## Creating the feature matrices

For each user, we want a matrix that contains a row for each whitelisted addon and a column for each recommender. A cell then contains the confidence score that the respective recommender gave for the respective user and addon.


```python
recommenders = {
    "collaborative": CollaborativeRecommender(),
    "similarity": SimilarityRecommender(),
    "locale": LocaleRecommender("./top_addons_by_locale.json"),
    "legacy": LegacyRecommender()
}
```

    INFO:requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:boto3.resources.action:Calling s3:get_object with {u'Bucket': 'telemetry-parquet', u'Key': 'taar/similarity/donors.json'}
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:boto3.resources.action:Calling s3:get_object with {u'Bucket': 'telemetry-parquet', u'Key': 'taar/similarity/lr_curves.json'}
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:boto3.resources.action:Calling s3:get_object with {u'Bucket': 'telemetry-parquet', u'Key': 'taar/locale/top10_dict.json'}
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com
    INFO:boto3.resources.action:Calling s3:get_object with {u'Bucket': 'telemetry-parquet', u'Key': 'taar/legacy/legacy_dict.json'}
    INFO:botocore.vendored.requests.packages.urllib3.connectionpool:Starting new HTTPS connection (1): s3-us-west-2.amazonaws.com



```python
def compute_features(client_data):
    recommendations = []
    matrix = []
    
    for _, recommender in recommenders.items():
        recommendations.append(recommender.get_weighted_recommendations(client_data))

    for addon in whitelist:
        matrix.append([features[addon] for features in recommendations])

    return client_data, np.array(matrix)
```


```python
X_unnormalized = training_masked.map(compute_features).cache()
```

    CPU times: user 1.3 s, sys: 20 ms, total: 1.32 s
    Wall time: 1.34 s


## Normalization

The optimization algorithms that we use here are much more minimal than what you typically from highly optimized ML libs. Because of this, we need to take special care of properly preprocessing the data.

In the following, we perform these operations:
- [Min-Max scaling](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- Changing the locale scores to a double square root


```python
max_feature_values = X_unnormalized.map(lambda (_, features): np.max(features, axis=0)).reduce(np.maximum)
```


```python
def preprocess_locale_scores(scores):
    return np.sqrt(np.sqrt(scores))
```


```python
def scale_features((client, features)):
    features = features / max_feature_values
    features[:, 0] = preprocess_locale_scores(features[:, 0])
    return client, features
```


```python
X = X_unnormalized.map(scale_features).cache()
```

## Making recommendations

Computing recommendations then reduces down to a dot product. These results are then sorted.


```python
def get_weighted_recommendations(client_data, features, weights):
    scores = features.dot(weights)
    return client_data, np.argsort(-scores)
```

## Measuring the performance (MAP)

We use the [MAP](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) measure as an error metric for this optimization problem. The reason for this is mostly that we only have positive data, i.e. we know addons which users like, but we don't really have a lot of data about addons that users hate.


```python
def average_precision(client_data, recommendations):
    tp = fp = 0.
    masked = set(client_data['masked_addons'])
    precisions = []
    
    for recommendation in recommendations:
        if whitelist[recommendation] in masked:
            tp += 1
            precisions.append(tp / (tp + fp))
            if tp == len(masked):
                break
        else:
            fp += 1
    
    if len(precisions) > 0:
        return np.mean(precisions)
    else:
        return 0.
```

## Training an ensemble model


```python
from scipy.optimize import minimize
```

### Defining a cost function

We find recommendations, compute the average precision (AP) and then calculate the mean of that (MAP). This produces a value between 0 and 1. We then subtract this value from 1 because SciPy looks for a function to minimize, not to maximize.


```python
def cost(weights):
    return 1 - X.map(lambda (client_data, features): get_weighted_recommendations(client_data, features, weights))\
            .map(lambda (client_data, recommendations): average_precision(client_data, recommendations))\
            .mean()
```

### Choosing an initial guess

There are many ways of choosing initial guesses. A constant vector of 1s seems to be a sensible prior (with properly normalized features it means that all recommenders are equally useful). However, randomly choosing initial values can also be useful.


```python
def get_initial_guess(n):
    return np.ones(n)
```


```python
def get_initial_guess(n):
    return np.random.random(n)
```

### Logging the optimization process

SciPy is logging the optimization process to a stdout stream that Jupyter seems to ignore. Because it's extremely useful to see how the optimization process is actually going, we define a helper function that queries `cost` and then also prints the results.


```python
def verbose_cost(weights):
    new_cost = cost(weights)
    print "New guess:", weights, "leads to a cost of", new_cost
    return new_cost
```

### Optimizing

The 4-element vectors in the following correspond to the recommenders in this order:


```python
recommenders.keys()
```




    ['locale', 'legacy', 'collaborative', 'similarity']




```python
num_features = len(recommenders)
x0 = get_initial_guess(num_features)
print "Initial guess:", x0
minimize(verbose_cost, x0, method="COBYLA", bounds=[(0, None)] * num_features, tol=1e-10)
```

    Initial guess: [ 0.38848417  0.88084589  0.75092367  0.55297014]


    /mnt/anaconda2/lib/python2.7/site-packages/scipy/optimize/_minimize.py:400: RuntimeWarning: Method COBYLA cannot handle bounds.
      RuntimeWarning)


    New guess: [ 0.38848417  0.88084589  0.75092367  0.55297014] leads to a cost of 0.707622781302
    New guess: [ 1.38848417  0.88084589  0.75092367  0.55297014] leads to a cost of 0.656494973351
    New guess: [ 1.38848417  1.88084589  0.75092367  0.55297014] leads to a cost of 0.656422468122
    New guess: [ 1.38848417  1.88084589  1.75092367  0.55297014] leads to a cost of 0.654930943044
    New guess: [ 1.38848417  1.88084589  1.75092367  1.55297014] leads to a cost of 0.705653233305
    New guess: [ 2.09824812  1.88185242  1.77162925 -0.15116436] leads to a cost of 0.650331978164
    New guess: [ 1.77118452  1.88239279  1.78274545 -0.52919285] leads to a cost of 0.676865782411
    New guess: [ 2.09824812  2.13185216  1.77162925 -0.150807  ] leads to a cost of 0.650317222086
    New guess: [ 2.47914626  2.13193849  1.78550007  0.17280419] leads to a cost of 0.645031745
    New guess: [ 2.94162872  2.13430499  1.85173384  0.35089817] leads to a cost of 0.64766524252
    New guess: [ 2.47088222  2.13193989  2.03536151  0.17182145] leads to a cost of 0.644838888911
    New guess: [ 2.93514485  2.13431271  2.08578754  0.35046019] leads to a cost of 0.647555900808
    New guess: [ 2.2391335   2.13192224  2.03749002  0.61486546] leads to a cost of 0.648236958856
    New guess: [ 2.45361213  2.37412669  2.03962347  0.11241492] leads to a cost of 0.64485202398
    New guess: [ 2.40695018  2.06861335  2.05236846 -0.06080101] leads to a cost of 0.650097018706
    New guess: [ 2.40623863  2.15281843  2.03680255  0.27674149] leads to a cost of 0.647205125991
    New guess: [ 2.59372858  2.14491346  2.04137402  0.18996835] leads to a cost of 0.645074520339
    New guess: [ 2.47562601  2.11723493  2.03755087  0.11130106] leads to a cost of 0.644850497035
    New guess: [ 2.46923605  2.13154248  2.06654635  0.1729171 ] leads to a cost of 0.644832189335
    New guess: [ 2.40714605  2.12598887  2.07005314  0.17573524] leads to a cost of 0.64482586746
    New guess: [ 2.41015955  2.0958185   2.06956249  0.18328436] leads to a cost of 0.644830128659
    New guess: [ 2.38800431  2.1583545   2.11032381  0.20524088] leads to a cost of 0.644802670959
    New guess: [ 2.37211408  2.18940864  2.14394918  0.24472158] leads to a cost of 0.646658348049
    New guess: [ 2.36972605  2.17175585  2.11175431  0.14701277] leads to a cost of 0.644747091203
    New guess: [ 2.35667064  2.18512938  2.11912016  0.12307511] leads to a cost of 0.644759171807
    New guess: [ 2.36559072  2.17785816  2.09805386  0.14846163] leads to a cost of 0.644752380191
    New guess: [ 2.3651368   2.18635858  2.12730558  0.16938283] leads to a cost of 0.644767646575
    New guess: [ 2.35678667  2.16337623  2.11219305  0.14952328] leads to a cost of 0.644739497067
    New guess: [ 2.35849599  2.13264723  2.10888069  0.14523409] leads to a cost of 0.644745663067
    New guess: [ 2.34904771  2.16557804  2.11683346  0.13695873] leads to a cost of 0.644755850599
    New guess: [ 2.35505048  2.16249916  2.11736807  0.16413752] leads to a cost of 0.644762459314
    New guess: [ 2.35947533  2.15697678  2.10902591  0.14784325] leads to a cost of 0.644747134976
    New guess: [ 2.35421196  2.16644045  2.10553503  0.15035233] leads to a cost of 0.644743468454
    New guess: [ 2.35629285  2.16358134  2.11400735  0.15294105] leads to a cost of 0.644745882658
    New guess: [ 2.3550488   2.16260289  2.11246568  0.14917386] leads to a cost of 0.644737559785
    New guess: [ 2.35396551  2.16384721  2.11379679  0.14589284] leads to a cost of 0.644741053821
    New guess: [ 2.35416505  2.16328499  2.11087374  0.14935869] leads to a cost of 0.644738593765
    New guess: [ 2.35534529  2.16200211  2.11198312  0.14865235] leads to a cost of 0.644741003315
    New guess: [ 2.35423792  2.16296124  2.11354626  0.15053809] leads to a cost of 0.644737942187
    New guess: [ 2.35484684  2.16323991  2.1128379   0.14856679] leads to a cost of 0.644741385146
    New guess: [ 2.35494313  2.16249202  2.11257683  0.14962398] leads to a cost of 0.644737282883
    New guess: [ 2.35474013  2.16241058  2.11264482  0.14953948] leads to a cost of 0.644739098796
    New guess: [ 2.35526431  2.16280936  2.11259375  0.14980911] leads to a cost of 0.644737279802
    New guess: [ 2.35538456  2.16270274  2.11277262  0.14976691] leads to a cost of 0.644737776965
    New guess: [ 2.35547517  2.16256051  2.11224258  0.14990243] leads to a cost of 0.644738193279
    New guess: [ 2.35490227  2.1631241   2.11259745  0.14990005] leads to a cost of 0.644737963467



    

    KeyboardInterruptTraceback (most recent call last)

    <ipython-input-145-2aaca29e9936> in <module>()
          2 x0 = get_initial_guess(num_features)
          3 print "Initial guess:", x0
    ----> 4 minimize(verbose_cost, x0, method="COBYLA", bounds=[(0, None)] * num_features, tol=1e-10)
    

    /mnt/anaconda2/lib/python2.7/site-packages/scipy/optimize/_minimize.pyc in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        453                              **options)
        454     elif meth == 'cobyla':
    --> 455         return _minimize_cobyla(fun, x0, args, constraints, **options)
        456     elif meth == 'slsqp':
        457         return _minimize_slsqp(fun, x0, args, jac, bounds,


    /mnt/anaconda2/lib/python2.7/site-packages/scipy/optimize/cobyla.pyc in _minimize_cobyla(fun, x0, args, constraints, rhobeg, tol, iprint, maxiter, disp, catol, **unknown_options)
        256     xopt, info = _cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg,
        257                                   rhoend=rhoend, iprint=iprint, maxfun=maxfun,
    --> 258                                   dinfo=info)
        259 
        260     if info[3] > catol:


    /mnt/anaconda2/lib/python2.7/site-packages/scipy/optimize/cobyla.pyc in calcfc(x, con)
        246 
        247     def calcfc(x, con):
    --> 248         f = fun(x, *args)
        249         i = 0
        250         for size, c in izip(cons_lengths, constraints):


    <ipython-input-143-063bda5c54e1> in verbose_cost(weights)
          1 def verbose_cost(weights):
    ----> 2     new_cost = cost(weights)
          3     print "New guess:", weights, "leads to a cost of", new_cost
          4     return new_cost


    <ipython-input-137-24f764a1f265> in cost(weights)
          1 def cost(weights):
    ----> 2     return 1 - X.map(lambda (client_data, features): get_weighted_recommendations(client_data, features, weights))            .map(lambda (client_data, recommendations): average_precision(client_data, recommendations))            .mean()
    

    /usr/lib/spark/python/pyspark/rdd.py in mean(self)
       1153         2.0
       1154         """
    -> 1155         return self.stats().mean()
       1156 
       1157     def variance(self):


    /usr/lib/spark/python/pyspark/rdd.py in stats(self)
       1016             return left_counter.mergeStats(right_counter)
       1017 
    -> 1018         return self.mapPartitions(lambda i: [StatCounter(i)]).reduce(redFunc)
       1019 
       1020     def histogram(self, buckets):


    /usr/lib/spark/python/pyspark/rdd.py in reduce(self, f)
        800             yield reduce(f, iterator, initial)
        801 
    --> 802         vals = self.mapPartitions(func).collect()
        803         if vals:
        804             return reduce(f, vals)


    /usr/lib/spark/python/pyspark/rdd.py in collect(self)
        774         """
        775         with SCCallSiteSync(self.context) as css:
    --> 776             port = self.ctx._jvm.PythonRDD.collectAndServe(self._jrdd.rdd())
        777         return list(_load_from_socket(port, self._jrdd_deserializer))
        778 


    /usr/lib/spark/python/lib/py4j-0.10.3-src.zip/py4j/java_gateway.py in __call__(self, *args)
       1129             proto.END_COMMAND_PART
       1130 
    -> 1131         answer = self.gateway_client.send_command(command)
       1132         return_value = get_return_value(
       1133             answer, self.gateway_client, self.target_id, self.name)


    /usr/lib/spark/python/lib/py4j-0.10.3-src.zip/py4j/java_gateway.py in send_command(self, command, retry, binary)
        881         connection = self._get_connection()
        882         try:
    --> 883             response = connection.send_command(command)
        884             if binary:
        885                 return response, self._create_connection_guard(connection)


    /usr/lib/spark/python/lib/py4j-0.10.3-src.zip/py4j/java_gateway.py in send_command(self, command)
       1026 
       1027         try:
    -> 1028             answer = smart_decode(self.stream.readline()[:-1])
       1029             logger.debug("Answer received: {0}".format(answer))
       1030             if answer.startswith(proto.RETURN_MESSAGE):


    /mnt/anaconda2/lib/python2.7/socket.pyc in readline(self, size)
        449             while True:
        450                 try:
    --> 451                     data = self._sock.recv(self._rbufsize)
        452                 except error, e:
        453                     if e.args[0] == EINTR:


    KeyboardInterrupt: 


### Experimental: Grid search


```python
from itertools import product
```


```python
def grid_search(cost, parameter_space):
    for parameters in parameter_space:
        print parameters, cost(parameters)
```


```python
space = product(range(3), repeat=4)
grid_search(cost, space)
```

    (0, 0, 0, 0) 0.999317795469
    (0, 0, 0, 1) 0.781044284439
    (0, 0, 0, 2) 0.781044284439
    (0, 0, 1, 0) 0.943562693329
    (0, 0, 1, 1) 0.775298736943
    (0, 0, 1, 2) 0.777014400398
    (0, 0, 2, 0) 0.943562693329
    (0, 0, 2, 1) 0.767080292565
    (0, 0, 2, 2) 0.775298736943
    (0, 1, 0, 0) 0.995510515481
    (0, 1, 0, 1) 0.781035766119
    (0, 1, 0, 2) 0.780880352461
    (0, 1, 1, 0) 0.943214127322
    (0, 1, 1, 1) 0.775202154909
    (0, 1, 1, 2) 0.776862986849
    (0, 1, 2, 0) 0.943034455897
    (0, 1, 2, 1) 0.766951549013
    (0, 1, 2, 2) 0.775129598268
    (0, 2, 0, 0) 0.995510515481
    (0, 2, 0, 1) 0.781786764264
    (0, 2, 0, 2) 0.781035766119
    (0, 2, 1, 0) 0.943193566877
    (0, 2, 1, 1) 0.776209558464
    (0, 2, 1, 2) 0.776919897482
    (0, 2, 2, 0) 0.943214127322
    (0, 2, 2, 1) 0.76798319702
    (0, 2, 2, 2) 0.775202154909
    (1, 0, 0, 0) 0.653586710269
    (1, 0, 0, 1) 0.706249773598
    (1, 0, 0, 2) 0.713821595475
    (1, 0, 1, 0) 0.650420413684
    (1, 0, 1, 1) 0.704830022379
    (1, 0, 1, 2) 0.713027854432
    (1, 0, 2, 0) 0.650320396078
    (1, 0, 2, 1) 0.703943832142
    (1, 0, 2, 2) 0.712374380858
    (1, 1, 0, 0) 0.652823591273
    (1, 1, 0, 1) 0.705832584758
    (1, 1, 0, 2) 0.713615932023
    (1, 1, 1, 0) 0.649844424588
    (1, 1, 1, 1) 0.704397272961
    (1, 1, 1, 2) 0.712828033415
    (1, 1, 2, 0) 0.649716652369
    (1, 1, 2, 1) 0.703557325313
    (1, 1, 2, 2) 0.712172539026
    (1, 2, 0, 0) 0.653352678701
    (1, 2, 0, 1) 0.705896438837
    (1, 2, 0, 2) 0.713424691705
    (1, 2, 1, 0) 0.650397013469
    (1, 2, 1, 1) 0.704522417619
    (1, 2, 1, 2) 0.712589168725
    (1, 2, 2, 0) 0.650406273884
    (1, 2, 2, 1) 0.703674904419
    (1, 2, 2, 2) 0.711961697999
    (2, 0, 0, 0) 0.653586710269
    (2, 0, 0, 1) 0.690287976483
    (2, 0, 0, 2) 0.706249773598
    (2, 0, 1, 0) 0.651445337683
    (2, 0, 1, 1) 0.684947350383
    (2, 0, 1, 2) 0.705463845082
    (2, 0, 2, 0) 0.650420413684
    (2, 0, 2, 1) 0.680590889828
    (2, 0, 2, 2) 0.704830022379
    (2, 1, 0, 0) 0.653184699287
    (2, 1, 0, 1) 0.690123263589
    (2, 1, 0, 2) 0.706156340238
    (2, 1, 1, 0) 0.651128308542
    (2, 1, 1, 1) 0.68474123067
    (2, 1, 1, 2) 0.705355728315
    (2, 1, 2, 0) 0.650086878921
    (2, 1, 2, 1) 0.68035786005
    (2, 1, 2, 2) 0.704696794839
    (2, 2, 0, 0) 0.652823591273
    (2, 2, 0, 1) 0.689644935857
    (2, 2, 0, 2) 0.705832584758
    (2, 2, 1, 0) 0.650846311386
    (2, 2, 1, 1) 0.684340948629
    (2, 2, 1, 2) 0.705035266025
    (2, 2, 2, 0) 0.649844424588
    (2, 2, 2, 1) 0.680023236015
    (2, 2, 2, 2) 0.704397272961


## Comparison to old the method

To validate if the MAP numbers that we get are any good, it's useful to compare them to the results of the previous recommendation process. The following is a minimal reimplementation of this `RecommendationManager`. It's used here because we want to use our masked data instead of the data fetched from HBase.


```python
class RecommendationManager:
    def __init__(self, recommenders):
        self.recommenders = recommenders
        
    def recommend(self, client_data, limit):
        recommendations = []
        
        for r in self.recommenders:
            recommendations += r.recommend(client_data, limit)
            
            if len(recommendations) >= limit:
                break
            
        return recommendations[:limit]
```

This helper function is similar to `map(superlist.index, sublist`) but ignores elements from the sublist that don't appear in the superlist.


```python
def list_elements_to_indices(superlist, sublist):
    result = []
    
    for a in sublist:
        for i, b in enumerate(superlist):
            if a == b:
                result.append(i)
                break
        
    return result
```


```python
def evaluate_recommendation_manager(mngr):
    return 1 - training_masked\
        .map(lambda user: (user, mngr.recommend(user, 10)))\
        .map(lambda (user, recommendations): average_precision(user, list_elements_to_indices(whitelist, recommendations)))\
        .mean()
```

As we can see, the previous recommendation manager performs much worse than our new model:


```python
mngr = RecommendationManager([recommenders["legacy"], recommenders["collaborative"], recommenders["similarity"], recommenders["locale"]])
evaluate_recommendation_manager(mngr)
```




    0.94540781980452748



However, this comparison is a little bit unfair. The locale recommender is generally extremely useful and can be used as a better baseline. With this ordering (where nearly only the locale recommender is queried), we get a much more comparable result. The results are now in the same ballpark and the ensemble is better by around 2%.


```python
mngr = RecommendationManager([recommenders["locale"], recommenders["legacy"], recommenders["collaborative"], recommenders["similarity"]])
evaluate_recommendation_manager(mngr)
```




    0.66339567712705083




```python
mngr = RecommendationManager([recommenders["locale"]])
evaluate_recommendation_manager(mngr)
```




    0.66444559195452979


