import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import os


def get_art():
    art = pd.read_csv('MetObjects.csv')
    art.rename(columns=lambda c: c.lower().replace(' ','_'), inplace=True)

    return art

def filter_artist_cols(art):
    # separator value for filtering functions
    separator = '|'

    
    art.drop(columns=[  'artist_prefix',
                        'object_id', 
                        'artist_display_name', 
                        'artist_display_bio',
                        'artist_suffix',
                        'object_date',
                        'object_end_date',
                        'dimensions',
                        'geography_type',
                        'city',
                        'state',
                        'county',
                        'region',
                        'subregion',
                        'locale',
                        'locus',
                        'excavation',
                        'river',
                        'rights_and_reproduction',
                        'link_resource',
                        'metadata_date',
                        'repository',
                        'tags_aat_url',
                        'tags_wikidata_url',
                        "reign",
                        "dynasty",
                        "period",
                        'artist_wikidata_url'], inplace=True)

    # Artist Role
    # Role of the artist related to the type of artwork or object that was created
    #      if null, change to ''
    #      if multiple roles keep first role
    #      keep only the top 5 roles plus N/A, all else are 'other'
    art.artist_role = art.artist_role.fillna('')
    role_list = ['Artist', 'Publisher', 'Designer', 'Maker', 'Author', 'N/A']
    art['artist_role'] = art['artist_role'].map(lambda x: x.split(separator, 1)[0]).map(lambda x: 'N/A' if x == '' else x)
    art['artist_role'] = art['artist_role'].map(lambda x:  'Other' if x not in (role_list) else x)

    # artist_alpha_sort	
    # Used to sort alphabetically by author last name. 
    #      if null, change to ''
    #      if multiple artists keep only primary artist
    art.artist_alpha_sort = art.artist_alpha_sort.fillna('N/A')
    art['artist_alpha_sort'] = art['artist_alpha_sort'].map(lambda x: x.split(separator, 1)[0])

    # artist nationality
    # nationality of artist
    ##      if null, change to ''
    #       if multiple artists keep primary artist
    #       keep only the top 6 nationalities, all else 'Other'
    art.artist_nationality = art.artist_nationality.fillna('')
    nationality_list = ['American', 'French', 'Italian', 'British', 'German', 'Japanese']
    art['artist_nationality'] = art['artist_nationality'].map(lambda x: x.split(separator, 1)[0]).str.strip()
    art['artist_nationality'] = art['artist_nationality'].map(lambda x:  'Other' if x not in (nationality_list) else x)


    # artist_begin_date
    # birth date of artist, filter to year only
    ##      if null, change to '' to allow string filters
    #       if multiple artists keep primary artist info
    #       keep only the '-' sign if it exists, then the first four digits
    #       change blanks to N/A
    art.artist_begin_date = art.artist_begin_date.fillna('')
    art['artist_begin_date'] = art['artist_begin_date'].map(lambda x: x.split(separator, 1)[0])
    art['artist_begin_date'] = art['artist_begin_date'].str.strip()
    # filter down to year only, remove invalid values, keep negative sign. 
    regexp = r'(-?\d+)'
    art['artist_begin_date'] = art['artist_begin_date'].map(lambda x: re.findall(regexp, x)[0] if x != '' else 'N/A')


    # artist_end_date
    # death date of artist (9999 for still living) 
    ##      if null, change to '' to allow string filters
    #      if multiple artists keep primary artist
    art.artist_end_date = art.artist_end_date.fillna('')
    art['artist_end_date'] = art['artist_end_date'].map(lambda x: x.split(separator, 1)[0])
    art['artist_end_date'] = art['artist_end_date'].str.strip()
    # filter down to year only, remove invalid values, keep negative sign. 
    regexp = r'(-?\d+)'
    art['artist_end_date'] = art['artist_end_date'].map(lambda x: re.findall(regexp, x)[0] if x != '' else 'N/A')

    # artist_ulan_url
    # ULAN URL for the artist     
    ##      if null, change to '' and keep only primary artist url
    ##      change to bool if artist url exists
    ##      change column name to 'has_artist_url'
    art.artist_ulan_url = art.artist_ulan_url.fillna('').map(lambda x: x.split(separator, 1)[0])
    art.artist_ulan_url = np.where(art['artist_ulan_url'].str.contains('http'), True, False)
    art.rename(columns={'artist_ulan_url':'has_artist_url'}, inplace=True)

    # artist gender
    # Gender of the artist.  Original data only contains designations for females
    ##      if null, change to 'N/A'
    ##      keep gender only for primary artist
    ##      if not N/A or Female, then Male
    ##      if id'd as N/A but there is an URL for the artist, change to male (in testing 99.99% of blank genders with url present were actually male)
    art.artist_gender = art.artist_gender.fillna('N/A')
    art['artist_gender'] = art['artist_gender'].map(lambda x: x.split(separator, 1)[0]).map(lambda x: 'Male' if x == '' else x)
    art['artist_gender'] = np.where((art['artist_gender'] == 'N/A') & (art['has_artist_url'] == True), 'Male', art['artist_gender'])

    return art

# 
def first_part_clean(   df,
                        consolidate_threshold=.01,
                        consolidate_list=["culture","gallery_number","object_name","credit_line","medium"],
                        group_to_decade=True,
                        dummy_columns = ['gallery_number','department','object_name',"culture","credit_line","medium"],
                        weight_columns = ['country', 
                                          'tags', 
                                          'classification', 
                                          'credit_line', 
                                          'medium', 
                                          'accessionyear',
                                          'artist_role',
                                          'artist_alpha_sort', 
                                          'artist_nationality', 
                                          'artist_begin_date', 
                                          'artist_end_date', 
                                          'artist_gender', 'portfolio'],
                        drop_whence_weighed = False,
                        one_if_exists = ['object_wikidata_url', ]
                    ):
    ''' 
    inputs: gonna need data frame, the rest can be left as default or modified as deemed necessary
    
    process:
            drops and object id, these are unique keys with no significance (using object number)
            cleans accessionyear, replacing nan with 0, and breaking strings into the first four (year)
                if group to decade is true, it'll floor the year and return the beginning decade number (1963 -> 1960)
            goes through gallery_number and fills nan with 0, encodes any strings to ints (starting at 1000), turns any low counts into 999
            fills object_name nan with Misc or classification if available
            fills title nan with Unknown
            turns portfolio into boolean (has or does not have)
            if consolidate list is not empyt:
                    it'll go through the list of features and turn any low values counts (threshold given) into "other") to help clean the data
            drop dynasty reign and period
            turns dummy list into dummy columns
    
    returns: modified dataframe
    '''

    ## ascension year, turning unknown into 0 so that they're not removed, may be able to impute with logical leaps/unknown value in doing so (case by case basis)
    df["accessionyear"] = df["accessionyear"].replace("NaN",0).fillna(0).astype("string").str[:4].astype(int)
    if group_to_decade:
        df["accessionyear"] = df["accessionyear"]//10*10

    # gallery number (name), placing na into none group for lack of better info. only 4 are string, remapping to encoded number
    df["gallery_number"] = df["gallery_number"].fillna(0)
    for i,each in enumerate(df[df["gallery_number"].fillna(0).astype(str).str.contains(r"[a-z]")]["gallery_number"].value_counts().index):
        #print(i,each)
        df["gallery_number"] = df["gallery_number"].replace({each:(1000+i)})
    df["gallery_number"]  = df["gallery_number"].astype(int)
    gal_count = df["gallery_number"].value_counts(normalize=True)
    idx = gal_count[gal_count.lt(consolidate_threshold)].index
    df.loc[df["gallery_number"].isin(idx), "gallery_number"] = 999
    

    ## replacing nan with misc category for object name
    df["object_name"] = np.where(df["object_name"].isna()==True,df["classification"],df["object_name"])
    df["object_name"] = df["object_name"].fillna("Misc")

    ## replacing nan with unknown for title name
    df["title"] = df["title"].fillna("Unknown")

    ## turning portfolio into boolean (all values seem to be individual, only 4 in target)
    df["portfolio"] = np.where(df.portfolio.isna(),False,True)

    #culture seems to be worth keeping at the moment

    # fix pipes in country field
    df["country"] = fix_pipes(df["country"])

    # run fields that should be 1 if something even just exists through this ;)
    for field in one_if_exists:
        df = make_1_if_exists(df, field)

    # And this runs the list of weight columns through the weighing process
    df = weighted_fields(df, 'is_highlight', weight_columns)

    # if you set drop_whence_weigh to True then it'll drop them, elswise keeps
    if drop_whence_weighed: df.drop(columns= weight_columns, inplace=True)

    ## consolidates low value counts
    if len(consolidate_list)>0:
        for each in consolidate_list:
            df[each] = df[each].fillna("Unknown")
            counts = df[each].value_counts(normalize=True)
            idx = counts[counts.lt(consolidate_threshold)].index
            df.loc[df[each].isin(idx), each] = 'Other'

    #get dummies for X subsets
    if len(dummy_columns)>0:
        temp_df = df.copy()
        df = pd.get_dummies(df, columns=dummy_columns, drop_first=False)
        ##adding columns back in for use in explore
        df = pd.concat([df,temp_df[dummy_columns]],axis=1)
    
    return df, dummy_columns


def split_tvt_stratify(df, target, dummy_columns):
    """
    takes in a dataframe, splits it into 60, 20, 20, 
    and seperates out the x variables and y (target) as new df/series
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123,stratify = df[target])

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123,stratify = train_validate[target])

    train,validate,test = get_kmeans_cluster_features(train,validate,test,target)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    #drop out dummy_columns
    X_train.drop(columns=dummy_columns,inplace=True)
    X_validate.drop(columns=dummy_columns,inplace=True)
    X_test.drop(columns=dummy_columns,inplace=True)

    print(f"df -> {df.shape}")
    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test


def fix_pipes(series_in_frame):
    output = series_in_frame.fillna('None').str.split('|')
    output = output.apply(lambda x: x[0])
    return output

def split_data(df, strat_by, rand_st=123):
    '''
    Takes in: a pd.DataFrame()
          and a column to stratify by  ;dtype(str)
          and a random state           ;if no random state is specifed defaults to [123]
          
      return: train, validate, test    ;subset dataframes
    '''
    # split the training and validation data off from the test data
    # the test data will be .2 of the dataset
    train, test = train_test_split(df, test_size=.2, 
                               random_state=rand_st, stratify=df[strat_by])
    train, validate = train_test_split(train, test_size=.25, 
                 random_state=rand_st, stratify=train[strat_by])
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')


    return train, validate, test


def initalize_museum(new= False):
    ''' 
    initalizes for consistency, no input, copy outputs
    '''
    
    if new == True:
        
        target = "is_highlight"
        df = get_art()
        df = filter_artist_cols(df)
        df,dummy_columns = first_part_clean(df)
        X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test = split_tvt_stratify(df,target,dummy_columns)

        data_list = [X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test, df, target]

        pd.to_pickle(data_list, 'data_list')

        return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test, df, target
    else:
        filename = "data_list"
        # if file is available locally, read it
        if os.path.isfile(filename):
            data_list = pd.read_pickle(filename)
            X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test, df, target =  data_list
            return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test, df, target
        
        # if file not available locally, acquire data from source csv
        # and write it as pickle locally for future use
        else:
            print('''\n
            There is no chached file available, pleas run again with Parameter: new = True
            The results will be cached for faster access in the future.
                        Timed at 1.9s vs the 15.2s to get anew

            Thank you for you patience.
            ''')
            return np.arange(11)


def weighted_fields(df, target, fields_to_weigh: list):
    '''
    takes in the df, the target variable, and a list of field(s) to weigh

    using the training set it weighs the field for frequency of

    occurrences both in and out of the target set

    subtracts the frequency in the non-target set from that in the target set

    NaNs are dropped from the result and the weights are mapped using the original field

    anytyhing not in the weighted set is automatically zero

    Returns: the df with a new field weighted accordingly
    '''
    # fillna
    for x in fields_to_weigh:
        df[x] = df[x].fillna('None')
    # Split the data
    train, validate, test = split_data(df, strat_by= target)
    # loop through each field in the list of fields_to_weigh
    for x in fields_to_weigh:

        # get the value_counts for in target and out of target values in the fields_to_weigh list
        field_true_target = train[train[target] == True][x].value_counts()
        field_false_target = train[~train[target] == True][x].value_counts()
    # Divide the resultant values by their sums
        field_true_target = field_true_target / field_true_target.sum()

        field_false_target = field_false_target / field_false_target.sum()
    # Make them DataFrames
        field_true_target = pd.DataFrame(field_true_target)

        field_false_target = pd.DataFrame(field_false_target)
    # create a new combined field
        field_true_target['combined'] = field_true_target - field_false_target

    # only use values above zero
        field_weights = field_true_target.combined[field_true_target.combined > 0]

    # make a new field name with weighed added on to it
        new_field_name = x + '_weighed'

    # use the new_field_name and map the weights in
        df[new_field_name] = df[x].map(lambda x: 0 if x not in field_weights.index else field_weights.loc[x])

# return the DataFrame with the new field in it
    return df

def make_1_if_exists(df, col_to_use):
    '''
    
    '''
    # fillna with zeros
    df[col_to_use] = df[col_to_use].fillna(0)

    new_col = col_to_use + '_1_or_0'

    df[new_col] = df[col_to_use].map(lambda x: 0 if x == 0 else 1)

    return df

def start():
    print('''
    X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test, df, target = initalize_museum()
    
    ''')
    

def get_kmeans_cluster_features(train,validate,test,target):
    ''' 
    takes in your three datasets to apply the new featuers to as well as applying to the base datasets
    a dictionary (iterable lists with the last being the order of clustering (function to auto later))
    '''
    import warnings
    warnings.filterwarnings("ignore")
    dict_to_cluster={}
    dict_to_cluster.update({"cluster_strong_yes":[ 'department_The Libraries', 
                                            'department_European Paintings', 
                                            'department_Robert Lehman Collection', 
                                            'department_The Cloisters', 
                                            'department_The American Wing', 
                                            'department_Musical Instruments', 
                                            'department_Modern and Contemporary Art',
                                            "is_timeline_work", 
                                            'object_name_Painting', 
                                            'culture_American', 
                                            "medium_None"
                                            ],
                        "cluster_strong_no":[  'department_Drawings and Prints', 
                                        'department_Asian Art', 
                                        'department_European Sculpture and Decorative Arts', 
                                        'object_name_Print', 
                                        'department_Greek and Roman Art',
                                        'credit_line_The Jefferson R. Burdick Collection, Gift of Jefferson R. Burdick',
                                        'object_name_Kylix fragment',
                                        "portfolio"
                                    ],
                            })

    from sklearn.cluster import KMeans
    #list has to be same len as clusters, sorts clusters into high and low eff
    threshold_list=[.75,.001]

    for i in list(dict_to_cluster):
        #set features
        #print(i,dict_to_cluster[i])
        #X1 = train[train[i]==1][dict_to_cluster[i]]
        #X2 = validate[validate[i]==1][dict_to_cluster[i]]
        #X3 = test[test[i]==1][dict_to_cluster[i]]
        X1 = train[dict_to_cluster[i]]
        X2 = validate[dict_to_cluster[i]]
        X3 = test[dict_to_cluster[i]]

        kmeans_scaled = KMeans(n_clusters=21,random_state=123)
        kmeans_scaled.fit(X1)

        X1["cluster"] = kmeans_scaled.predict(X1)
        X2["cluster"] = kmeans_scaled.predict(X2)
        X3["cluster"] = kmeans_scaled.predict(X3)

        train[f"{i}"] = X1["cluster"]
        validate[f"{i}"] = X2["cluster"]
        test[f"{i}"] = X3["cluster"]

    for num,i in enumerate(list(dict_to_cluster)):
        series1 = train.groupby([f"{i}"])[target].mean()
        series2 = validate.groupby([f"{i}"])[target].mean()
        series3 = test.groupby([f"{i}"])[target].mean()
        if num == 0:
            train[f"{i}"] = np.where(train[f"{i}"].isin(series1[series1>.75].index.tolist()),1,0)
            validate[f"{i}"] = np.where(validate[f"{i}"].isin(series2[series2>.75].index.tolist()),1,0)
            test[f"{i}"] = np.where(test[f"{i}"].isin(series3[series3>.75].index.tolist()),1,0)
        if num == 1:
            train[f"{i}"] = np.where(train[f"{i}"].isin(series1[series1<.001].index.tolist()),1,0)
            validate[f"{i}"] = np.where(validate[f"{i}"].isin(series2[series2<.001].index.tolist()),1,0)
            test[f"{i}"] = np.where(test[f"{i}"].isin(series3[series3<.001].index.tolist()),1,0)

    return train,validate,test