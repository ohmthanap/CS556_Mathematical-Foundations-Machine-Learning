#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendations HW

# **Name:** Thanapoom Phatthanaphan
# 
# **ID:** 20011296

# **Collaboration Policy:** Homeworks will be done individually: each student must hand in their own answers. Use of partial or entire solutions obtained from others or online is strictly prohibited.

# **Late Policy:** Late submission have a penalty of 2\% for each passing hour. 

# **Submission format:** Successfully complete the Movie Lens recommender as described in this jupyter notebook. Submit a `.py` and an `.ipynb` file for this notebook. You can go to `File -> Download as ->` to download a .py version of the notebook. 
# 
# **Only submit one `.ipynb` file and one `.py` file.** The `.ipynb` file should have answers to all the questions. Do *not* zip any files for submission. 

# **Download the dataset from here:** https://grouplens.org/datasets/movielens/1m/

# In[1]:


# Import all the required libraries
import numpy as np
import pandas as pd


# ## Reading the Data
# Now that we have downloaded the files from the link above and placed them in the same directory as this Jupyter Notebook, we can load each of the tables of data as a CSV into Pandas. Execute the following, provided code.

# In[2]:


# Read the dataset from the two files into ratings_data and movies_data
#NOTE: if you are getting a decode error, add "encoding='ISO-8859-1'" as an additional argument
#      to the read_csv function
column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings, engine='python')
column_list_movies = ["MovieID","Title","Genres"]
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies, engine='python', encoding = 'latin-1')
column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users, engine='python')


# `ratings_data`, `movies_data`, `user_data` corresponds to the data loaded from `ratings.dat`, `movies.dat`, and `users.dat` in Pandas.

# ## Data analysis

# We now have all our data in Pandas - however, it's as three separate datasets! To make some more sense out of the data we have, we can use the Pandas `merge` function to combine our component data-frames. Run the following code:

# In[3]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)
data


# Next, we can create a pivot table to match the ratings with a given movie title. Using `data.pivot_table`, we can aggregate (using the average/`mean` function) the reviews and find the average rating for each movie. We can save this pivot table into the `mean_ratings` variable. 

# In[4]:


mean_ratings=data.pivot_table('Ratings','Title',aggfunc='mean')
mean_ratings


# Now, we can take the `mean_ratings` and sort it by the value of the rating itself. Using this and the `head` function, we can display the top 15 movies by average rating.

# In[5]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],aggfunc='mean')
top_15_mean_ratings = mean_ratings.sort_values(by = 'Ratings',ascending = False).head(15)
top_15_mean_ratings


# Let's adjust our original `mean_ratings` function to account for the differences in gender between reviews. This will be similar to the same code as before, except now we will provide an additional `columns` parameter which will separate the average ratings for men and women, respectively.

# In[6]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
mean_ratings


# We can now sort the ratings as before, but instead of by `Rating`, but by the `F` and `M` gendered rating columns. Print the top rated movies by male and female reviews, respectively.

# In[7]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)

mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_female_ratings.head(15))

top_male_ratings = mean_ratings.sort_values(by='M', ascending=False)
print(top_male_ratings.head(15))


# In[8]:


mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]


# Let's try grouping the data-frame, instead, to see how different titles compare in terms of the number of ratings. Group by `Title` and then take the top 10 items by number of reviews. We can see here the most popularly-reviewed titles.

# In[9]:


ratings_by_title=data.groupby('Title').size()
ratings_by_title.sort_values(ascending=False).head(10)


# Similarly, we can filter our grouped data-frame to get all titles with a certain number of reviews. Filter the dataset to get all movie titles such that the number of reviews is >= 2500.

# ## Question 1

# Create a ratings matrix using Numpy. This matrix allows us to see the ratings for a given movie and user ID. The element at location $[i,j]$ is a rating given by user $i$ for movie $j$. Print the **shape** of the matrix produced.  
# 
# Additionally, choose 3 users that have rated the movie with MovieID "**1377**" (Batman Returns). Print these ratings, they will be used later for comparison.
# 
# 
# **Notes:**
# - Do *not* use `pivot_table`.
# - A ratings matrix is *not* the same as `ratings_data` from above.
# - The ratings of movie with MovieID $i$ are stored in the ($i$-1)th column (index starts from 0)  
# - Not every user has rated every movie. Missing entries should be set to 0 for now.
# - If you're stuck, you might want to look into `np.zeros` and how to use it to create a matrix of the desired shape.
# - Every review lies between 1 and 5, and thus fits within a `uint8` datatype, which you can specify to numpy.

# In[10]:


# Create the desired matrix which
# the number of rows is the number of users
# and the number of columns is the number of movies
ratings = np.zeros((max(user_data['UserID']), max(movies_data['MovieID'])), dtype=np.uint8)

# Add movie rating of each user in the matrix
for i in data.itertuples():
    ratings[i.UserID - 1, i.MovieID - 1] = i.Ratings

print(ratings)


# In[11]:


# Print the shape
print(ratings.shape)


# In[12]:


# Store and print ratings for Batman Returns
ratings_batman = np.array([ratings[9, 1376],ratings[12, 1376], ratings[17, 1376] ])
print(ratings_batman)


# ## Question 2

# Normalize the ratings matrix (created in **Question 1**) using Z-score normalization. While we can't use `sklearn`'s `StandardScaler` for this step, we can do the statistical calculations ourselves to normalize the data.
# 
# Before you start:
# - Your first step should be to get the average of every *column* of the ratings matrix (we want an average by title, not by user!).
# - Make sure that the mean is calculated considering only non-zero elements. If there is a movie which is rated only by 10 users, we get its mean rating using (sum of the 10 ratings)/10 and **NOT** (sum of 10 ratings)/(total number of users)
# - All of the missing values in the dataset should be replaced with the average rating for the given movie. This is a complex topic, but for our case replacing empty values with the mean will make it so that the absence of a rating doesn't affect the overall average, and it provides an "expected value" which is useful for computing correlations and recommendations in later steps.
# - In our matrix, 0 represents a missing rating.
# - Next, we want to subtract the average from the original ratings thus allowing us to get a mean of 0 in every *column*. It may be very close but not exactly zero because of the limited precision `float`s allow.
# - Lastly, divide this by the standard deviation of the *column*.
# 
# - Not every MovieID is used, leading to zero columns. This will cause a divide by zero error when normalizing the matrix. Simply replace any NaN values in your normalized matrix with 0.

# In[13]:


# Find average of every column of the ratings matrix
column_mean = []
for i in range(ratings.shape[1]):
    mean = np.mean(ratings[:, i][ratings[:, i] != 0])
    column_mean.append(mean) 

# Replace missing values (0 value) with mean in each column of the matrix
normalized_matrix = np.array(ratings, dtype=float)
normalized_matrix = np.where(normalized_matrix == 0, column_mean, normalized_matrix)

# Subtract the average from the original ratings
normalized_matrix -= column_mean

# Divide by the standard deviation of the column
ratings = ratings.astype('float64')
ratings[ratings == 0] = np.nan
column_std = np.nanstd(ratings, axis=0)
normalized_matrix /= column_std

# Replace NaN value as 0 value
normalized_matrix[np.isnan(normalized_matrix)] = 0
print(normalized_matrix)


# ## Question 3

# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix from the previous question. Perform the process using numpy, and along the way print the shapes of the $U$, $S$, and $V$ matrices you calculated.

# In[14]:


# Compute the SVD of the normalised matrix
U, s, VT = np.linalg.svd(normalized_matrix)
S = np.zeros((normalized_matrix.shape[0], normalized_matrix.shape[1]))
S[:normalized_matrix.shape[1], :normalized_matrix.shape[1]] = np.diag(s)
svd = U@S@VT
print(svd)


# In[15]:


# Print the shapes
print(U.shape)
print(S.shape)
print(VT.shape)


# ## Question 4

# Reconstruct four rank-k rating matrix $R_k$, where $R_k = U_kS_kV_k^T$ for k = [100, 1000, 2000, 3000]. Using each of $R_k$ make predictions for the 3 users selected in Question 1, for the movie with ID 1377 (Batman Returns). Compare the original ratings with the predicted ratings.

# In[16]:


# Reconstruct four rank-k rating matrix
for k in [100, 1000, 2000, 3000]:
    r_k = U[:, :k]@S[:k, :k]@VT[:k, :]
    
    # Rescale the reconstructed data matrix back to the original scale
    r_k *= column_std
    r_k += column_mean
    
    # Compare predicted ratings with the original ratings    
    ratings_batman_predict = r_k[[9, 12, 17], 1376]
    print(f"\nPredictions for the 3 users rated Batman Returns for k = {k}")
    print(ratings_batman_predict)
    print(ratings_batman)


# ## Question 5

# ### Cosine Similarity
# Cosine similarity is a metric used to measure how similar two vectors are. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. Cosine similarity is high if the angle between two vectors is 0, and the output value ranges within $cosine(x,y) \in [0,1]$. $0$ means there is no similarity (perpendicular), where $1$ (parallel) means that both the items are 100% similar.
# 
# $$ cosine(x,y) = \frac{x^T y}{||x|| ||y||}  $$

# **Based on the reconstruction rank-1000 rating matrix $R_{1000}$ and the cosine similarity,** sort the movies which are most similar. You will have a function `top_movie_similarity` which sorts data by its similarity to a movie with ID `movie_id` and returns the top $n$ items, and a second function `print_similar_movies` which prints the titles of said similar movies. Return the top 5 movies for the movie with ID `1377` (*Batman Returns*)
# 
# Note: While finding the cosine similarity, there are a few empty columns which will have a magnitude of **zero** resulting in NaN values. These should be replaced by 0, otherwise these columns will show most similarity with the given movie. 

# In[17]:


# Sort the movies based on cosine similarity
def top_movie_similarity(data, movie_id, top_n=5):
    # Movie id starts from 1
    # Use the calculation formula above to compute Cosine Similarity
    x = data
    y = data[:, movie_id - 1]
    magnitude_x = np.linalg.norm(x)
    magnitude_y = np.linalg.norm(y)
    dot_product = np.dot(x.T, y)
    cosine_similarity = dot_product/(magnitude_x * magnitude_y)
    
    # descending sort
    top_indices_similar_movies = np.argsort(-cosine_similarity)[0:top_n + 1]
    
    return top_indices_similar_movies


# Print the top 5 movies for Batman Returns
def print_similar_movies(movie_titles, top_indices_similar_movies):
    print('Most Similar movies: ')
    for i, movie in enumerate(top_indices_similar_movies):
        print(f"{i+1}. {movie_titles[movie_titles['MovieID'] == movie + 1]['Title'].values[0]}")

        
movie_id = 1377
r_1000 = U[:, :1000]@S[:1000, :1000]@VT[:1000, :]

# Compute Cosine Similarity based on rank-1000 rating matrix
top_indices_similar_movies = top_movie_similarity(r_1000, movie_id)
print_similar_movies(movies_data, top_indices_similar_movies)


# ## Question 6

# ### Movie Recommendations
# Using the same process from Question 5, write `top_user_similarity` which sorts data by its similarity to a user with ID `user_id` and returns the top result. Then find the MovieIDs of the movies that this similar user has rated most highly, but that `user_id` has not yet seen. Find at least 5 movie recommendations for the user with ID `5954` and print their titles.
# 
# Hint: To check your results, find the genres of the movies that the user likes and compare with the genres of the recommended movies.

# In[18]:


#Sort users based on cosine similarity
def top_user_similarity(data, user_id):
    # Use the calculation formula above to compute Cosine Similarity
    x = data
    y = data[user_id - 1, :]
    magnitude_x = np.linalg.norm(x)
    magnitude_y = np.linalg.norm(y)
    dot_product = np.dot(x, y.T)
    cosine_similarity = dot_product/(magnitude_x * magnitude_y)
    
    # descending sort
    top_indices_similar_user = np.argsort(-cosine_similarity)[1]
    return top_indices_similar_user


user_id = 5954

# Create array of movies that user has never seen
movie_user_seen = data[data['UserID'] == user_id]['MovieID'].values
movie_user_not_seen = movies_data[~movies_data['MovieID'].isin(movie_user_seen)]['MovieID'].values

# Compute Cosine Similarity based on rank-1000 rating matrix
top_indices_similar_user = top_user_similarity(r_1000, user_id)

# Find top rated movies of similar user
top_rated_movie = data[(data['UserID'] == top_indices_similar_user) & data['MovieID'].isin(movie_user_not_seen)][['Title', 'Ratings']].values
top_rated_movie = top_rated_movie[np.argsort(-top_rated_movie[:, 1])]

# Print movie recommendations
print('Movie recommendations: ')
for i, movie in enumerate(top_rated_movie[0:10]):
    print(f"{i+1}. {movie[0]}")

