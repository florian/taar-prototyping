
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



We're using the [COBYLA](https://en.wikipedia.org/wiki/COBYLA) algorithm for optimization. There is no theoretical reason for this, it just seems to work pretty well here and finds good results fairly quickly. Of course other algorithms could be used instead of COBYLA here.


```python
num_features = len(recommenders)
x0 = get_initial_guess(num_features)
print "Initial guess:", x0
best_weights = minimize(verbose_cost, x0, method="COBYLA", tol=1e-5).x
```

    Initial guess: [ 0.92298266  0.80952834  0.47482559  0.93028254]
    New guess: [ 0.92298266  0.80952834  0.47482559  0.93028254] leads to a cost of 0.70481583848
    New guess: [ 1.92298266  0.80952834  0.47482559  0.93028254] leads to a cost of 0.683259673786
    New guess: [ 1.92298266  1.80952834  0.47482559  0.93028254] leads to a cost of 0.682828731505
    New guess: [ 1.92298266  1.80952834  1.47482559  0.93028254] leads to a cost of 0.677545615984
    New guess: [ 1.92298266  1.80952834  1.47482559  1.93028254] leads to a cost of 0.704320141592
    New guess: [ 2.54277091  1.8219189   1.62672706  0.16045467] leads to a cost of 0.644513981539
    New guess: [ 3.10869844  1.83493605  1.7863102  -0.64830352] leads to a cost of 0.653830802141
    New guess: [ 2.82573467  1.82842748  1.70651863 -0.24392442] leads to a cost of 0.650168526742
    New guess: [ 2.34803908  1.8219189   1.62672706  0.00367612] leads to a cost of 0.644276588671
    New guess: [ 2.64621403  1.82843916  1.70666192 -0.38959174] leads to a cost of 0.650625584604
    New guess: [ 2.34594988  2.07189664  1.62689649  0.00627108] leads to a cost of 0.644170271356
    New guess: [ 2.21974184  2.07285995  2.08903624  0.14944043] leads to a cost of 0.643864945793
    New guess: [ 1.89709424  2.08308918  1.92022466  0.49192625] leads to a cost of 0.646968490192
    New guess: [ 2.32789588  2.0921582   2.2038969  -0.04352929] leads to a cost of 0.649392835458
    New guess: [ 2.11333109  2.07265118  2.08010597  0.08446382] leads to a cost of 0.64377707068
    New guess: [ 1.99002108  2.07436896  1.99219151  0.28336766] leads to a cost of 0.646458215236
    New guess: [ 2.11360958  2.19764782  2.0801908   0.08359446] leads to a cost of 0.64379046169
    New guess: [ 2.21049135  2.06846178  2.18538188 -0.12037592] leads to a cost of 0.649653816166
    New guess: [ 2.16191122  2.07055648  2.13274393 -0.01795605] leads to a cost of 0.649351114205
    New guess: [ 2.09595331  2.07279573  2.13645453  0.10517811] leads to a cost of 0.643720436561
    New guess: [ 2.03797441  2.07334341  2.08527885  0.20338299] leads to a cost of 0.645379102778
    New guess: [ 2.11967882  2.0718779   2.16661094  0.05585166] leads to a cost of 0.643785403169
    New guess: [ 2.09577172  2.04154928  2.13632595  0.10559357] leads to a cost of 0.643710495351
    New guess: [ 2.04422612  2.03017339  2.14686283  0.13735619] leads to a cost of 0.643844416004
    New guess: [ 2.14362396  2.03913569  2.14496242  0.1447854 ] leads to a cost of 0.643881223236
    New guess: [ 2.09633595  2.03945135  2.15174423  0.07849893] leads to a cost of 0.643696912763
    New guess: [ 2.10316765  2.03945547  2.16389608  0.08555592] leads to a cost of 0.64368751853
    New guess: [ 2.13316467  2.03504251  2.16124672  0.07846737] leads to a cost of 0.643726021865
    New guess: [ 2.0827792   2.03394638  2.18656235  0.08965028] leads to a cost of 0.64370474659
    New guess: [ 2.10290283  2.054895    2.16609241  0.08462445] leads to a cost of 0.643682454215
    New guess: [ 2.10087471  2.0556882   2.16320964  0.09155138] leads to a cost of 0.643663234232
    New guess: [ 2.10250859  2.05918165  2.15847246  0.10593283] leads to a cost of 0.643720032552
    New guess: [ 2.09401137  2.05407414  2.1658778   0.09360208] leads to a cost of 0.643670862285
    New guess: [ 2.10180827  2.06418224  2.17094645  0.0810033 ] leads to a cost of 0.643688973126
    New guess: [ 2.09762119  2.05666936  2.15697182  0.08829919] leads to a cost of 0.643673629311
    New guess: [ 2.10320981  2.04867106  2.16387853  0.08912346] leads to a cost of 0.643670738137
    New guess: [ 2.10138011  2.05761482  2.16559307  0.08918268] leads to a cost of 0.643669899197
    New guess: [ 2.10270248  2.05628293  2.16249433  0.09487594] leads to a cost of 0.643668336993
    New guess: [ 2.1003321   2.05594746  2.16147325  0.09088954] leads to a cost of 0.643665414376
    New guess: [ 2.10011276  2.05614347  2.16337364  0.09192413] leads to a cost of 0.643662699314
    New guess: [ 2.09911257  2.05457339  2.16389872  0.09219518] leads to a cost of 0.64366146201
    New guess: [ 2.09848905  2.05465737  2.16558374  0.09143388] leads to a cost of 0.643659863805
    New guess: [ 2.09806703  2.05443576  2.16727347  0.09228962] leads to a cost of 0.643659828128
    New guess: [ 2.09726348  2.05490157  2.16703823  0.09247847] leads to a cost of 0.643660074734
    New guess: [ 2.0974859   2.05354516  2.16782444  0.09074681] leads to a cost of 0.643662872148
    New guess: [ 2.09893898  2.0560786   2.16782747  0.09206931] leads to a cost of 0.643660102617
    New guess: [ 2.09846868  2.05435243  2.16695516  0.09311671] leads to a cost of 0.643659868625
    New guess: [ 2.09829882  2.05484491  2.16740192  0.09226152] leads to a cost of 0.643659541738
    New guess: [ 2.09862907  2.05513668  2.16758835  0.09216423] leads to a cost of 0.643659491796
    New guess: [ 2.09839081  2.05554234  2.16753013  0.0922813 ] leads to a cost of 0.643659927876
    New guess: [ 2.0985701   2.05510278  2.16779929  0.09226662] leads to a cost of 0.643659585035
    New guess: [ 2.09882433  2.05487584  2.16758039  0.09180066] leads to a cost of 0.643659614623
    New guess: [ 2.09894348  2.05501919  2.16742686  0.09247996] leads to a cost of 0.643661448305
    New guess: [ 2.09855716  2.05511271  2.16765027  0.09194057] leads to a cost of 0.643659992214
    New guess: [ 2.09856409  2.05523417  2.16756798  0.09219179] leads to a cost of 0.643659854984
    New guess: [ 2.09867124  2.05517271  2.1676113   0.09215317] leads to a cost of 0.643659446062
    New guess: [ 2.09877068  2.05511418  2.16759821  0.09219078] leads to a cost of 0.643659741806
    New guess: [ 2.09866484  2.05516     2.16763123  0.09209726] leads to a cost of 0.643659835669
    New guess: [ 2.09867646  2.05517999  2.1675846   0.0921414 ] leads to a cost of 0.643659620469
    New guess: [ 2.09865439  2.05519963  2.16763462  0.09219978] leads to a cost of 0.643659846133
    New guess: [ 2.09869178  2.0551524   2.16761222  0.09216297] leads to a cost of 0.643659456912
    New guess: [ 2.09866067  2.05516321  2.16760646  0.09215591] leads to a cost of 0.643659446059
    New guess: [ 2.09866061  2.05516627  2.16760435  0.09216257] leads to a cost of 0.643659441699
    New guess: [ 2.09865596  2.05516635  2.16761766  0.09216839] leads to a cost of 0.643659490954
    New guess: [ 2.09865563  2.05517139  2.16760371  0.09215997] leads to a cost of 0.643659446318
    New guess: [ 2.09866689  2.05516597  2.1675905   0.0921613 ] leads to a cost of 0.643659446039
    New guess: [ 2.09866473  2.05516607  2.16759527  0.09216174] leads to a cost of 0.643659445519


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




    0.64123918634900057



## Optimizing on a subset with some manual decisions

When looking at the results so far, it seems like the locale and collaborative recommenders are the most important ones. The legacy recommender is also useful, but it can rarely be used and by using it we don't really optimize for MAP. Changes to the similarity recommender's weight also only lead to a small change.

Because of this, this section first tries to optimize for locale/collaborative weights using a grid search. Afterwards, we fix these weights and find the best weight for the similarity recommender. This grid search is quite expensive, so we'll work on a smaller subset of the data.

Generally, the results using the subset are in the same ballpark, but of course there are still some differences.

**Disclaimer:** This section is very much experimental and the code is not perfectly refactored


```python
from itertools import product
```


```python
X_full = X
```


```python
X = X_full.sample(False, 0.1).cache()
```


```python
n = 65
locale = np.linspace(0.2, 2.5, num=n)
collaborative = np.linspace(0.2, 2.5, num=n)
```


```python
%time z = [cost([x, 1., y, .11], X) for x, y in product(locale, collaborative)]
```

    CPU times: user 4min 57s, sys: 24.4 s, total: 5min 22s
    Wall time: 27min 26s



```python
xx, yy = np.meshgrid(locale, collaborative)
x = xx.ravel()
y = yy.ravel()
```

This plot isn't really informative, because there are some areas with very bad parameters which lead to the plot being mostly blue.


```python
hb = plt.hexbin(x, y, C=z, gridsize=30, cmap='jet')
plt.colorbar(hb)
plt.xlabel("Collaborative")
plt.ylabel("Locale")
plt.show()
```


![png](output_110_0.png)


To fix this, we can cut off all the results with cost > 0.65, which creates this much more interpretable result:


```python
z2 = np.minimum(z, 0.65)
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
hb = plt.hexbin(x, y, C=z2, gridsize=30, cmap='jet')
plt.colorbar(hb)
plt.xlabel("Collaborative")
plt.ylabel("Locale")
plt.show()
```


![png](output_114_0.png)


The optimum seems to be around (1.3, 1.8).


```python
list(product(locale, collaborative))[1995]
```




    (1.278125, 1.8171874999999997)



Next, we can try to find the best argument for the similarity recommender when fixing the other weights.


```python
x = np.linspace(0, 0.2, num=100)
```


```python
y = [cost([1.278125, 1., 1.8171874999999997, xi], X) for xi in x]
```


```python
plt.plot(x, y)
```




    [<matplotlib.lines.Line2D at 0x7fb3c33445d0>]




![png](output_120_1.png)



```python
x[np.argmin(y)]
```




    0.10707070707070707



When evaluating on all of the data, this doesn't really improve the recommender though.


```python
cost([1.278125, 1., 1.8171874999999997, 0.10707070707070707], X_test)
```




    0.64153074985822722


