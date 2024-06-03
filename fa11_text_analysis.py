
# coding: utf-8

# <a id="home"></a>
# ![Final Lesson Exercise](images/Banner_FEX.png)

# # Lesson #11: Text Analysis

# ## About this assignment
# In this assignment, you will explore quotes from the `Simpsons` cartoon.<br/>
# 
# This time you will practice a classification of a text analysis problem, including:
# * Finding information using regular expressions and simple text manipulations.
# * Vectorization and full classification pipeline for the classification problem.
# 
# The end goal: classify `who said` a text quotation. 

# In[1]:


# --------------------------- RUN THIS CODE CELL -------------------------------------

# --------
# imports and setup 
# Use the following libraries for the assignment, when needed:
# --------
import re
import os

import pandas as pd
import numpy as np

import sklearn 
from sklearn import preprocessing, metrics, naive_bayes, pipeline, feature_extraction
from sklearn.feature_extraction import text
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline


from sklearn import neighbors, tree, ensemble, naive_bayes, svm
# *** KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# <a id="dataset_desc"></a>
# [Go back to the beginning of the assignment](#home)
# ## The raw dataset - the Simpsons
# In this assignment, you will explore information regarding the "Simpsons" cartoon.<br/>
# The raw dataset, includes sentences quotes by different figures from the Simpson cartoons.
# 
# Such as:<br>
# `Homer Simpson` said: "**Okay, here we go...**"

# ## 1. Read the raw data and extract information
# In this section you will perform the following actions:<br />
# * Extract person-name and quoted text, using regular expressions and simple text manipulations.
# * Read a text file and extract information into a dataframe.

# ### 1.a. Find "what was said" and "who said it" - part 1
# 
# In this section you will deal with  quotations from "Simpsons" episodes.<br/>
# 
# For a given text such as:<br />
#     `Homer Simpson` said: "**Okay, here we go...**"<br />
# You need to extract the following:<br />
# * Extract the person who is mentioned as saying the quoted text, in the above example, you should extract:<br />
#   `Homer Simpson`
# * Extract the quoted text, <u>after removing the double quotation marks(")</u>. In the above example, you should extract:<br />
#   **Okay, here we go...**
# * Note that you could expect the double quotation mark (") will appear only before and after the quoted<br />
#   text and no where else.
# 
# You could expect the input text to look similar to the following pattern:<br />
# `Person name` said: "**quoted text**"<br />
# 
# _Variations of the above pattern_:<br />
# * The colon punctuation mark (:) is optional, and and might not appear in the input sentence.
# * Instead of the word _said_ you could also expect the words: _answered_, _responded_, _replied_
# 
# An _additional pattern_:<br /> 
# * `Person name`: "**quoted text**"
#    * For Example:<br />
#    `Marge Simpson`: "**Here, take this**." 
#    
# **Important Note**: if the pattern is not found, return None values for both the <br />
#   extracted person name and the extracted text. 
# 
# ----------
# 
# <u>**Text files** with the different quotations</u> (one in each line), could be found in the [`data`](data) folder (click to open):<br />
# * A [`sample of 10 quotations`](data/simpsons_dataset_8_speakers_ptrn_10_lns.txt)
# * A [`sample of 100 quotations`](data/simpsons_dataset_8_speakers_ptrn_100_lns.txt)
# * Other quotation text file could be found in the [`data`](data) folder.

# In[2]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[3]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
'''
What do you need to do?

Complete the 'person_quotation_pattern_extraction' function to extract 
     the person quoted and the quoted text, from a given input 'raw_text' as explained above.

The returned values:
- extracted_person_name - The extracted person name, as appearing in the patterns explained above
- extracted_quotation   - The extracted quoted text (withot the surrounding quotation marks).

* Important Note: if the pattern is not found, return None values for both the <br />
                  extracted person name and the extracted text. 
------
The return statement should look similar to the following statement:
return extracted_person_name, extracted_quotation 
'''
def person_quotation_pattern_extraction(raw_text):
    ###
    ### YOUR CODE HERE
    ###
    text_patt = r'"(.*?)"'
    name_patt = r'(\w*\.*\s*[A-Z][a-z]+\s[A-Z][a-z]+)\s*(answered|:|said|responded|replied)'
    
    extracted_quotation = re.findall(text_patt, raw_text)
    extracted_person_name = re.findall(name_patt, raw_text)
    
    if extracted_quotation and extracted_person_name:
        extracted_quotation = extracted_quotation[0]
        extracted_person_name = extracted_person_name[0][0].strip()
        return extracted_person_name, extracted_quotation
    else:
        return None, None


# In[4]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:




# In[5]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.a. --- Test your implementation:
'''
print ("Test 1 - Testing the implementation of the 'person_quotation_pattern_extraction' method ...\n")

in_text = 'Marge Simpson responded: "Homer, please."'

try:
    person_name, extracted_text = person_quotation_pattern_extraction(in_text)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert person_name is not None and extracted_text is not None, "Missing 'person_name' or 'extracted_text' returned values"
    
print ("Good Job!\nYou've passed the 1st test for the 'person_quotation_pattern_extraction' function :-)")


# In[6]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.a. --- Test your implementation:
'''
print ("Test 2 - Testing the implementation of the 'person_quotation_pattern_extraction' method ...\n")

in_text = 'Marge Simpson responded: "Homer, please."'

try:
    person_name, extracted_text = person_quotation_pattern_extraction(in_text)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert person_name == 'Marge Simpson', "Wrong extracted value for 'person_name', try again"
assert extracted_text == '''Homer, please.''', "Wrong extracted value for 'extracted_text', try again"

print ("Good Job!\nYou've passed the 2nd test for the 'person_quotation_pattern_extraction' function :-)")


# In[7]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.a. --- Test your implementation:
'''
print ("Test 3 - Testing the implementation of the 'person_quotation_pattern_extraction' method ...\n")
print ("====> The following is a hidden test ... ")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 1.b. Find "what was said" and "who said it" - part 2
# 
# In this section you will continue to deal with quotations from "Simpsons" episodes.<br/>
# 
# This time, the quoted texts will come before the 
# For a given text such as:<br />
#     "**Oh man, that guy's tough to love.**", `Bart Simpson` replied.<br />
# You need to extract the following:<br />
# * Extract the person who is mentioned as saying the quoted text, in the above example, you should extract:<br />
#   `Bart Simpson`
# * Extract the quoted text, <u>after removing the double quotation marks(")</u>. In the above example, you should extract:<br />
#   ****Oh man, that guy's tough to love.**.**
# * Note that you could expect the double quotation mark (") will appear only before and after the quoted<br />
#   text and no where else.
# 
# You could expect the input text to look similar to the following pattern:<br />
# "**quoted text**", `Person name` said.<br />
# 
# _Variations of the above pattern_:<br />
# * The comma punctuation mark (,) is optional, and might not appear in the input sentence.
# * Instead of the word _said_ you could also expect the words: _answered_, _responded_, _replied_
#    
# **Important Note**: if the pattern is not found, return None values for both the <br />
#   extracted person name and the extracted text. 
#   
# ----------
# 
# <u>**Text files** with the different quotations</u> (one in each line), could be found in the [`data`](data) folder (click to open):<br />
# * A [`sample of 10 quotations`](data/simpsons_dataset_8_speakers_ptrn_10_lns.txt)
# * A [`sample of 100 quotations`](data/simpsons_dataset_8_speakers_ptrn_100_lns.txt)
# * Other quotation text file could be found in the [`data`](data) folder.

# In[8]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[9]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
'''
What do you need to do?

Complete the 'quotation_person_pattern_extraction' function to extract 
     the person quoted and the quoted text, from a given input 'raw_text' as explained above.

     Note: In this function you should expect the quoted text to appear before the person name.
     
The returned values:
- extracted_person_name - The extracted person name, as appearing in the patterns explained above
- extracted_quotation   - The extracted quoted text (withot the surrounding quotation marks).

* Important Note: if the pattern is not found, return None values for both the <br />
                  extracted person name and the extracted text. 
------
The return statement should look similar to the following statement:
return extracted_person_name, extracted_quotation 
'''
def quotation_person_pattern_extraction(raw_text):
    ###
    ### YOUR CODE HERE
    ###
    text_patt = r'"(.*?)"'
    name_patt = r'(\w*\.*\s*[A-Z][a-z]+\s[A-Z][a-z]+)\s*(answered|:|said|responded|replied)'
    
    extracted_quotation = re.findall(text_patt, raw_text)
    extracted_person_name = re.findall(name_patt, raw_text)
    
    if extracted_quotation and extracted_person_name:
        extracted_quotation = extracted_quotation[0]
        extracted_person_name = extracted_person_name[0][0].strip()
        return extracted_person_name, extracted_quotation
    else:
        return None, None


# In[10]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:



# In[11]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.b. --- Test your implementation:
'''
print ("Test 1 - Testing the implementation of the 'quotation_person_pattern_extraction' method ...\n")

in_text = '''"Oh man, that guy's tough to love.", Bart Simpson replied.''' 

try:
    person_name, extracted_text = quotation_person_pattern_extraction(in_text)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert person_name is not None and extracted_text is not None, "Missing 'person_name' or 'extracted_text' returned values"

print ("Good Job!\nYou've passed the 1st test for the 'quotation_person_pattern_extraction' function :-)")


# In[12]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.b. --- Test your implementation:
'''
print ("Test 2 - Testing the implementation of the 'quotation_person_pattern_extraction' method ...\n")

in_text = '''"Oh man, that guy's tough to love.", Bart Simpson replied.''' 

try:
    person_name, extracted_text = quotation_person_pattern_extraction(in_text)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

    assert person_name == 'Bart Simpson', "Wrong extracted value for 'person_name', try again"
    assert extracted_text == '''Oh man, that guy's tough to love.''', "Wrong extracted value for 'extracted_text', try again"

print ("Good Job!\nYou've passed the 2nd test for the 'quotation_person_pattern_extraction' function :-)")


# In[13]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.b. --- Test your implementation:
'''
print ("Test 3 - Testing the implementation of the 'quotation_person_pattern_extraction' method ...\n")

print ("====> The following is a hidden test ... ")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 1.c. Transfer raw data to dataframe
# In this section you will read lines from a text file and <br/>
# extract the quoted person name and quoted text from each line.
# 
# <u>Use both patterns</u>, which you implemented in the above methods to try to extract the person name and quoted text.<br/>
# 
# Remarks:
# * Hint: Use the returned object from the 'open' method to get the file handler.
# * Each line you read is expected to contain a new-line in the end of the line. Remove the new-line as following:
#     * line_cln = line.strip()
# * There are the options for each line (assume one of these three options):
#     1. The first set of patterns, for which the person name appears before the quoted text.
#     + The second set of patterns, for which the quoted text appears before the person.
#     + Empty lines. 
# 
# 

# In[14]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[15]:


'''
What do you need to do?

Complete the 'transfer_raw_text_to_dataframe' function to return a dataframe
   with the extracted person name and text as explained above. 
The information is expected to be extracted from the lines of the given 'filename' file.
* Use the above implemented 'person_quotation_pattern_extraction' and 
  'quotation_person_pattern_extraction' methods, the the two pattern sets.

The returned dataframe should include two columns:
- 'person_name' - containing the extracted person name for each line.
- 'extracted_text' - containing the extracted quoted text for each line.

The returned values:
- dataframe - The dataframe with the extracted information as described above.

* Important Note: if a line does not contain any quotation pattern, no information should be saved in the
                corresponding row in the dataframe. 
------
------
The return statement should look similar to the following statement:
return dataframe
'''
def transfer_raw_text_to_dataframe(filename):
    ###
    ### YOUR CODE HERE
    ###
    per_n = []
    extr_text = []
    
    with open(filename, "r") as file:
        for line in file:
            person_name, extracted_text = quotation_person_pattern_extraction(line)
            if person_name and extracted_text:
                per_n.append(person_name)
                extr_text.append(extracted_text)
    
    df = {'person_name': per_n, 'extracted_text': extr_text}
    return pd.DataFrame(df)


# In[16]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_10_lns.txt'


dataframe = transfer_raw_text_to_dataframe(file_name)
dataframe



# In[17]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.c. --- Test your implementation:
'''
print ("Test 1 - Testing the implementation of the 'transfer_raw_text_to_dataframe' method ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_10_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'transfer_raw_text_to_dataframe' function :-)")


# In[18]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.c. --- Test your implementation:
'''
print ("Test 2 - Testing the implementation of the 'transfer_raw_text_to_dataframe' method ...\n")

cols=sorted(['person_name', 'extracted_text'])
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_10_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    cols_in_df = sorted(list(dataframe.columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert dataframe is not None , "Missing 'dataframe' returned value"

assert dataframe.shape == (10, 2), 'Wrong shape for returned dataframe'
np.testing.assert_array_equal(cols_in_df, cols, 'wrong columns in returned dataframe')

print ("Good Job!\nYou've passed the 2nd test for the 'transfer_raw_text_to_dataframe' function :-)")


# In[19]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
1.c. --- Test your implementation:
'''
print ("Test 3 - Testing the implementation of the 'transfer_raw_text_to_dataframe' method ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_10_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    person_names = dataframe['person_name']
    person_names = [person for person in person_names if person is not None and person.strip()]
    unique_names = np.unique(person_names)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert len(person_names) == 10, 'Missing person names in returned dataframe'
print (unique_names)
print ("Good Job!\nYou've passed the 3rd test for the 'transfer_raw_text_to_dataframe' function :-)")
dataframe.head()


# ## 2. Auxiliary classification pipeline methods
# * Create simple pipeline
# * fit the classification pipeline
# * predict new test examples
# * evaluate performance

# ### 2.a. Create a simple pipeline 
# Create a simple pipeline to contain the following:
# * Vectorizer - a CountVectorizer object without any parameters
# * Classifier - a MultinomialNB classifier

# In[20]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[21]:


'''
What do you need to do?

Complete the 'create_simple_pipeline' function to return a simple classification
   pipeline object, which contains only the above components.
------
The return statement should look similar to the following statement:
return clf_pipeline
'''
def create_simple_pipeline():
    ###
    ### YOUR CODE HERE
    ###
    clf_pipeline=Pipeline([
        ('vect',CountVectorizer()),('clf',MultinomialNB())])
    return clf_pipeline


# In[22]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:




# In[23]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
2.a. --- Test your implementation:
'''
print ("Test - Testing the implementation of the 'create_simple_pipeline' method ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_10_lns.txt'

try:
    clf_pipeline = create_simple_pipeline()
    clf_pipeline_steps = clf_pipeline.steps
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert clf_pipeline is not None and clf_pipeline_steps is not None, 'Missing or corrupt returned pipeline object'
assert len(clf_pipeline_steps) == 2, 'Wrong number of steps in returned pipeline object' 

print ("Good Job!\nYou've passed the test for the 'create_simple_pipeline' function :-)")


# ### 2.b. Fit a pipeline on train dataframe
# Use the simple pipeline classification object to fit (train),<br />
# it on the input 'df_train' dataframe.
# 
# The input 'df_train' includes two columns:
# * 'person_name' - acting as the category, which you need to classify
# * 'extracted_text' - the raw text, repesenting a sentence (or a few short sentences), 
#    which are associated to the person, called 'person_name').
#    
# Use the df_train['extracted_text'] to get the X_train raw data.<br/>
# Use the df_train['person_name'] to get the y_train categories.<br/>
# 
# **Note**: No return value is needed in this function

# In[24]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[25]:


'''
What do you need to do?

Complete the 'fit' function to train and fit the given 'clf_pipeline' object.
      The training should be done on the input 'df_train' dataframe.

For detailed explanation see the above explanation.

* Note: No return value is needed in this function
'''
def fit(clf_pipeline, df_train):
    ###
    ### YOUR CODE HERE
    ###
    clf_pipeline.fit(df_train['extracted_text'],df_train['person_name'])


# In[26]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:




# ### 2.c. Predict test example
# Use the trained given simple pipeline to predict new test examples.<br />
# 
# The input `x_test` includes a pandas series of the 'extracted_text'.<br />
# This refers to the same type of data as the 'extracted_text' column in the df_train dataframe,<br />
#  described and used above.

# In[27]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[29]:


'''
What do you need to do?

Complete the 'predict' function to return the predicted values for the test
    'x_test' series as described above, using the trained given 'clf_trained_pipeline' pipeline.

For detailed explanation see the above explanation
------
The return statement should look similar to the following statement:
return y_predicted
'''
def predict(clf_trained_pipeline, x_test):
    ###
    ### YOUR CODE HERE
    ###
    return clf_trained_pipeline.predict(x_test)


# In[30]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:




# In[31]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
2.b., 2.c. --- Test your implementation:
'''
print ("Test 1 - Testing the implementation of the 'fit' and 'predict' methods ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_1k_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    # raw dataset is already shuffled
    df_train = dataframe.iloc[:800,:]
    df_test = dataframe.iloc[800:,:]
    clf_pipeline = create_simple_pipeline()
    fit(clf_pipeline, df_train)
    y_predicted = predict(clf_pipeline, df_test['extracted_text'])
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert y_predicted is not None , "Missing the 'y_predicted' series returned value"

print ("Good Job!\nYou've passed the 1st test for the 'fit' and 'predict' methods :-)")


# In[32]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
2.b., 2.c. --- Test your implementation:
'''
print ("Test 2 - Testing the implementation of the 'fit' and 'predict' methods ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_1k_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    # raw dataset is already shuffled
    df_train = dataframe.iloc[:800,:]
    df_test = dataframe.iloc[800:,:]
    clf_pipeline = create_simple_pipeline()
    fit(clf_pipeline, df_train)
    y_predicted = predict(clf_pipeline, df_test['extracted_text'])
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
assert y_predicted[0] == 'Homer Simpson', 'Wrong predicted value'
assert y_predicted[22] == 'Marge Simpson', 'Wrong predicted value'

print ("Good Job!\nYou've passed the 2nd test for the 'fit' and 'predict' methods :-)")


# ### 2.d. Evaluate trained classification pipeline
# Use the accuracy measure to evaluate the trained pipeline.<br />

# In[33]:


# --------------------------- RUN THIS CODE CELL (AFTER YOUR IMPLEMENTATION) -------------------------------------
# (OPTIONAL IMPLEMENTATION CELL) add some assitant code or use this
# cell code for you exploration, if needed:
###
### YOUR CODE HERE
###


# In[34]:


'''
What do you need to do?

Complete the 'evaluate_accuracy' function to return the accuracy value of trained 
    classification pipeline.
    
The evaluation will be performed between the actual values - 'y_test'
    and the predicted values 'y_predicted'.

-----
The return statement should look similar to the following statement:
return evaluation_val
'''
def evaluate_accuracy(y_test, y_predicted):
    ###
    ### YOUR CODE HERE
    ###
    return accuracy_score(y_test,y_predicted)


# In[35]:


# --------------------------- (AFTER YOUR IMPLEMENTATION if used) RUN THIS CODE CELL  ------------------------------------ 
# This section is FOR YOUR ASSISTANCE ONLY It will not be checked.
# Add assistance tests here IF NEEDED:



# In[36]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
2.d. --- Test your implementation:
'''
print ("Test 1 - Testing the implementation of the 'evaluate_accuracy' method ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_1k_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    # raw dataset is already shuffled
    df_train = dataframe.iloc[:800,:]
    df_test = dataframe.iloc[800:,:]
    y_test = df_test['person_name']
    clf_pipeline = create_simple_pipeline()
    fit(clf_pipeline, df_train)
    y_predicted = predict(clf_pipeline, df_test['extracted_text'])
    evaluation_val = evaluate_accuracy(y_test,y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'evaluate_accuracy' method :-)")


# In[37]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
2.d. --- Test your implementation:
'''
print ("Test 2 - Testing the implementation of the 'evaluate_accuracy' method ...\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_1k_lns.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    # raw dataset is already shuffled
    df_train = dataframe.iloc[:800,:]
    df_test = dataframe.iloc[800:,:]
    y_test = df_test['person_name']
    clf_pipeline = create_simple_pipeline()
    fit(clf_pipeline, df_train)
    y_predicted = predict(clf_pipeline, df_test['extracted_text'])
    evaluation_val = evaluate_accuracy(y_test,y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert evaluation_val is not None , "Missing the 'y_predicted' series returned value"
assert evaluation_val == 0.39 , "Wrong accuracy value"

print ("Good Job!\nYou've passed the 2nd test for the 'evaluate_accuracy' method :-)")


# In[38]:


# --------------------------- RUN THIS TEST CODE CELL -------------------------------------
'''
2.d. --- Test your implementation:
'''
print ("Test 3 - Testing the implementation of the 'evaluate_accuracy' method ...\n")

print ("The four most frequent figures :\n")
print ('- Homer Simpson')
print ('- Marge Simpson')
print ('- Bart Simpson')
print ('- Lisa Simpson')
print ("We will test with a mass of examples, but take only the frequent\n")

cols=['person_name', 'extracted_text']
file_name = '.' + os.sep + 'data' + os.sep + 'simpsons_dataset_8_speakers_ptrn_40k.txt'

try:
    dataframe = transfer_raw_text_to_dataframe(file_name)
    dataframe_freq = dataframe[dataframe['person_name'].isin(['Homer Simpson', 'Marge Simpson','Bart Simpson','Lisa Simpson'])]
    # raw dataset is already shuffled
    df_train = dataframe_freq.iloc[:20000,:]
    df_test = dataframe_freq.iloc[20000:,:]
    y_test = df_test['person_name']
    clf_pipeline = create_simple_pipeline()
    fit(clf_pipeline, df_train)
    y_predicted = predict(clf_pipeline, df_test['extracted_text'])
    evaluation_val = evaluate_accuracy(y_test,y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:') 
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert np.round(evaluation_val,2) ==  0.5 , "Wrong accuracy value"

print ("Good Job!\nYou've passed the 3rd test for the 'evaluate_accuracy' method :-)")


# In[ ]:





# In[ ]:




