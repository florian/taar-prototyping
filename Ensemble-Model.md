
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

    CPU times: user 24 ms, sys: 4 ms, total: 28 ms
    Wall time: 2min 50s



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

For the collaborative recommender, we use the confidence scores that were already used internally before. These are based on singular value decomposition: A generator job finds good feature representations for addons and computes the feature values for each addon. By looking at the addons that a user already has installed, we can then find feature values that indicate what kind of addons the user likes. After that, we can compute the confidence scores for addons by calculating the distance between their feature values and the feature values of the user.


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

Again, we already have some kind of confidence scores internally that we can reuse for the ensemble. These scores are based on a similarity measure: We find similar users to the current users (e.g. by comparing their locale, OS, number of bookmarks, etc.) and recommend their addons. The confidence score for an addon is then computed by summing up the similarity scores of all users that have the respective addon installed.

**TODO**: Compute logarithm


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

The confidence scores for the locale recommender are based on the number of addon installations in the respective locale. The more often an addon was installed in the locale, the higher its confidence score. We normalize the results for each locale separately, i.e. the most popular addon in each locale will have a confidence score of 1.


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

For the legacy recommender, we count how often an addon is listed as a replacement for an installed legacy addon. This count is a natural number that's directly used as the confidence score.


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

The [evaluation notebook](https://github.com/florian/taar-prototyping/blob/master/evaluation.ipynb) lists the portion of clients with a certain number of whitelisted addons. If the cut-off is set to `>= 3` or `>= 4` very few clients are left.


```python
useful_clients = completed_rdd.filter(lambda client: len(client['installed_addons']) >= 1).cache()
```


```python
useful_clients.count()
```




    435313



These users are useful for training and evaluating our model:


```python
training, test = useful_clients.randomSplit([0.8, 0.2])
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
def cost(weights, X=X):
    weighted_recommendations = X.map(lambda (client_data, features):
                get_weighted_recommendations(client_data, features, weights)
               )
    
    AP = weighted_recommendations.map(lambda (client_data, recommendations):
             average_precision(client_data, recommendations)
            )
        
    MAP = AP.mean()
            
    return 1 - MAP
```

### Choosing an initial guess

There are many ways of choosing initial guesses. A constant vector of 1s seems to be a sensible prior (with properly normalized features it means that all recommenders are equally useful). However, randomly choosing initial values can also be useful.


```python
def get_initial_guess_alternative(n):
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
best_weights = minimize(verbose_cost, x0, method="COBYLA", tol=1e-5)
```

    Initial guess: [ 0.48533936  0.55636737  0.13108097  0.47268632]
    New guess: [ 0.48533936  0.55636737  0.13108097  0.47268632] leads to a cost of 0.705378749967
    New guess: [ 1.48533936  0.55636737  0.13108097  0.47268632] leads to a cost of 0.651060536695
    New guess: [ 1.48533936  1.55636737  0.13108097  0.47268632] leads to a cost of 0.650813378865
    New guess: [ 1.48533936  1.55636737  1.13108097  0.47268632] leads to a cost of 0.649310991464
    New guess: [ 1.48533936  1.55636737  1.13108097  1.47268632] leads to a cost of 0.705340053019
    New guess: [ 2.1812669   1.55953397  1.15032963 -0.24516069] leads to a cost of 0.649269068547
    New guess: [ 1.82239216  1.56106918  1.15966164 -0.59318293] leads to a cost of 0.68011649334
    New guess: [ 2.1812669   1.80953154  1.15032963 -0.24405789] leads to a cost of 0.649252783755
    New guess: [ 2.54022724  1.80852452  1.16249432  0.10379047] leads to a cost of 0.644256645414
    New guess: [ 2.89564561  1.81024236  1.23708028  0.44746439] leads to a cost of 0.647646561847
    New guess: [ 2.53251318  1.80852796  1.41237407  0.10301236] leads to a cost of 0.64414121662
    New guess: [ 2.89159066  1.81025698  1.44762664  0.44915919] leads to a cost of 0.647528399838
    New guess: [ 2.70572828  1.81110052  1.41716028 -0.07717366] leads to a cost of 0.649620313481
    New guess: [ 2.28389193  2.22015535  1.41074175  0.2399346 ] leads to a cost of 0.646261875537
    New guess: [ 2.3711054   1.63964133  1.41141717  0.19202877] leads to a cost of 0.644120853605
    New guess: [ 2.43505878  1.63485173  1.4144207   0.29928083] leads to a cost of 0.647186124565
    New guess: [ 2.33574054  1.685659    1.41039694  0.2152001 ] leads to a cost of 0.644750594415
    New guess: [ 2.33845164  1.60879409  1.41222374  0.07538174] leads to a cost of 0.6440917324
    New guess: [ 2.33772888  1.60938307  1.47471495  0.07586042] leads to a cost of 0.644009589544
    New guess: [ 2.42535171  1.52196521  1.49168655  0.07169822] leads to a cost of 0.644082512942
    New guess: [ 2.29709238  1.56753478  1.47446749  0.0983011 ] leads to a cost of 0.64400555768
    New guess: [ 2.24370841  1.603316    1.58074647  0.08417185] leads to a cost of 0.643783705506
    New guess: [ 2.1748745   1.65692838  1.66953226  0.07278378] leads to a cost of 0.643679245126
    New guess: [ 2.12358501  1.58085683  1.75219668  0.09212549] leads to a cost of 0.643510185439
    New guess: [ 2.07508988  1.54715919  1.86234429  0.09440456] leads to a cost of 0.643346269316
    New guess: [ 1.97422953  1.50420451  1.92233998  0.09164693] leads to a cost of 0.643221598451
    New guess: [ 1.92453869  1.54478908  2.0288606   0.07891627] leads to a cost of 0.643100799414
    New guess: [  1.91327092e+00   1.49329991e+00   2.11136456e+00   1.19884174e-03] leads to a cost of 0.642887265516
    New guess: [ 1.89970632  1.44082328  2.19936718 -0.06910581] leads to a cost of 0.648365353754
    New guess: [ 1.9394283   1.46718024  2.13858713  0.04361035] leads to a cost of 0.642944585739
    New guess: [ 1.87403908  1.4741998   2.16067306 -0.10505426] leads to a cost of 0.648523392245
    New guess: [ 1.8630001   1.46215876  2.10191668  0.01908909] leads to a cost of 0.642836652223
    New guess: [ 1.79043854  1.42201217  2.14060563 -0.0660651 ] leads to a cost of 0.648279810268
    New guess: [ 1.84250083  1.50948094  2.11467384  0.05201167] leads to a cost of 0.642835425061
    New guess: [ 1.85109713  1.51133039  2.08771818  0.06515085] leads to a cost of 0.642949187669
    New guess: [ 1.80931759  1.50574013  2.15730912  0.02081292] leads to a cost of 0.642694870633
    New guess: [ 1.76667561  1.502212    2.1889827  -0.01193228] leads to a cost of 0.647967626059
    New guess: [ 1.79880642  1.51868732  2.1374796   0.00334186] leads to a cost of 0.642653097722
    New guess: [ 1.75406654  1.52468942  2.15130157 -0.0376157 ] leads to a cost of 0.648104492346
    New guess: [ 1.79964796  1.5344976   2.12561853  0.02753286] leads to a cost of 0.642590926776
    New guess: [ 1.80705948  1.54562394  2.13258086  0.02341699] leads to a cost of 0.642582412926
    New guess: [ 1.8324307   1.5378759   2.11636947  0.02025117] leads to a cost of 0.642677453204
    New guess: [ 1.78610746  1.56740136  2.13436564  0.03117161] leads to a cost of 0.642475887687
    New guess: [ 1.7683422   1.58515616  2.14444783  0.0467945 ] leads to a cost of 0.642573293813
    New guess: [ 1.78251233  1.58042962  2.11065488  0.01594986] leads to a cost of 0.642605513884
    New guess: [ 1.78054623  1.57035675  2.14462481  0.04113296] leads to a cost of 0.642521020339
    New guess: [ 1.78979051  1.56933176  2.13055628  0.03657832] leads to a cost of 0.64257366894
    New guess: [ 1.77703572  1.56832422  2.13649636  0.01866351] leads to a cost of 0.642569818419
    New guess: [ 1.7906249   1.57231243  2.13768078  0.02882232] leads to a cost of 0.642538645867
    New guess: [ 1.78376045  1.56875883  2.12087857  0.03858027] leads to a cost of 0.642550225269
    New guess: [ 1.78849908  1.56008802  2.13555681  0.03181311] leads to a cost of 0.642469661922
    New guess: [ 1.78565927  1.55881203  2.13576351  0.02946291] leads to a cost of 0.64247162563
    New guess: [ 1.78922731  1.55999212  2.13402434  0.03085045] leads to a cost of 0.642479782485
    New guess: [ 1.78643644  1.55929605  2.13500109  0.03498616] leads to a cost of 0.642503024201
    New guess: [ 1.78981947  1.5607      2.1389024   0.03041745] leads to a cost of 0.642470606799
    New guess: [ 1.78960884  1.5600329   2.13455655  0.0305563 ] leads to a cost of 0.642479242455
    New guess: [ 1.78884385  1.55924235  2.13589509  0.0318854 ] leads to a cost of 0.6424696855
    New guess: [ 1.78872172  1.56083818  2.13729181  0.03137475] leads to a cost of 0.642469648976
    New guess: [ 1.78822472  1.56081853  2.1375662   0.0321691 ] leads to a cost of 0.642469917759
    New guess: [ 1.78910145  1.56099475  2.13724084  0.03163382] leads to a cost of 0.642470818033
    New guess: [ 1.78802074  1.56053936  2.13736066  0.0307679 ] leads to a cost of 0.642470238001
    New guess: [ 1.78855447  1.56126247  2.13712095  0.03133963] leads to a cost of 0.64246964511
    New guess: [ 1.78944912  1.56156296  2.13692692  0.03149883] leads to a cost of 0.642468703147
    New guess: [ 1.79031974  1.5618698   2.13675835  0.03176924] leads to a cost of 0.642471907823
    New guess: [ 1.78939493  1.5613792   2.13649193  0.03161064] leads to a cost of 0.64246956174
    New guess: [ 1.78888366  1.56227688  2.13682278  0.03116207] leads to a cost of 0.642470407271
    New guess: [ 1.78949446  1.56130329  2.13703048  0.03110107] leads to a cost of 0.642468636616
    New guess: [ 1.78930409  1.56119788  2.1371166   0.03117061] leads to a cost of 0.642468161724
    New guess: [ 1.78930827  1.56101723  2.13752343  0.03137126] leads to a cost of 0.642468951299
    New guess: [ 1.78892824  1.56143402  2.13716901  0.03097404] leads to a cost of 0.642471136252
    New guess: [ 1.78930413  1.5610224   2.13696234  0.03124144] leads to a cost of 0.642467927256
    New guess: [ 1.78931129  1.56096071  2.13698544  0.03113891] leads to a cost of 0.642468350679
    New guess: [ 1.78913381  1.56110079  2.13685241  0.03135263] leads to a cost of 0.642467895216
    New guess: [ 1.78908565  1.56104591  2.13693722  0.0314014 ] leads to a cost of 0.642468252896
    New guess: [ 1.78930455  1.56117993  2.13675174  0.03147118] leads to a cost of 0.642468703147
    New guess: [ 1.78914061  1.56119741  2.13692572  0.03134059] leads to a cost of 0.64246789135
    New guess: [ 1.78899322  1.5612648   2.13682717  0.03118688] leads to a cost of 0.642467977338
    New guess: [ 1.78920341  1.56115866  2.1369562   0.03124825] leads to a cost of 0.642467901602
    New guess: [ 1.78922145  1.56125043  2.13685221  0.03135291] leads to a cost of 0.642468309046
    New guess: [ 1.78909777  1.5612084   2.13690914  0.03130194] leads to a cost of 0.64246783806
    New guess: [ 1.78908268  1.56119342  2.1369308   0.03130511] leads to a cost of 0.642467830073
    New guess: [ 1.78903065  1.56119853  2.13692727  0.03127382] leads to a cost of 0.642469730839
    New guess: [ 1.78906923  1.56120367  2.13692489  0.03132982] leads to a cost of 0.642467398037
    New guess: [ 1.78909932  1.56122045  2.13695178  0.03137242] leads to a cost of 0.642468252798
    New guess: [ 1.78907354  1.5611577   2.13689284  0.03135361] leads to a cost of 0.642467887229
    New guess: [ 1.78904312  1.56121681  2.13691663  0.03132686] leads to a cost of 0.642467398037
    New guess: [ 1.78907067  1.56121267  2.13693081  0.03134053] leads to a cost of 0.642467364919
    New guess: [ 1.78906693  1.56120876  2.13693613  0.03134138] leads to a cost of 0.642467364919
    New guess: [ 1.78906875  1.56120668  2.13692319  0.03135216] leads to a cost of 0.642467887229
    New guess: [ 1.7890772   1.56120938  2.13693288  0.03134127] leads to a cost of 0.642467364919
    New guess: [ 1.78907283  1.56122012  2.1369394   0.03133059] leads to a cost of 0.642467398037
    New guess: [ 1.78907033  1.56121341  2.13692953  0.03135041] leads to a cost of 0.642467887229



```python
cost([1.278125, 1., 1.8171874999999997, 0.10707070707070707])
```




    0.64280133639614734



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
def evaluate_recommendation_manager(mngr, data=training_masked):
    return 1 - data\
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




    0.66240704406886164



## Test set

The results using the test set are quite similar:


```python
test_masked = test.map(mask_addons).cache()
```


```python
X_test_unnormalized = test_masked.map(compute_features).cache()
```


```python
X_test = X_test_unnormalized.map(scale_features).cache()
```


```python
evaluate_recommendation_manager(mngr, test_masked)
```




    0.66328519154987853




```python
cost(best_weights, X_test)
```




    0.64363248228167091


