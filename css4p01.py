# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:30:50 2024

@author: welman
"""

import numpy as np
import pandas as pd

file_path = 'movie_dataset.csv'
movies_df = pd.read_csv(file_path)

#QUESTION 1: What is the highest rated movie in the dataset?
highest_rated_movie = movies_df.loc[movies_df['Rating'].idxmax()]

print("Highest Rated Movie:")
print("Title:", highest_rated_movie['Title'])
print("Rating:", highest_rated_movie['Rating'])

#QUESTION 2: What is the average revenue of all movies in the dataset? 
movies_df.describe(include=np.number)
movies_df.describe(include=['O'])
print('movies_df')

print("Before Cleaning - Missing Values:")
print(movies_df.isnull().sum())

movies_df_cleaned = movies_df.dropna()

print("\nAfter Cleaning - Missing Values:")
print(movies_df_cleaned.isnull().sum())

if 'Revenue (Millions)' in movies_df.columns:
    
    average_revenue = movies_df['Revenue (Millions)'].mean()
    print(f"The average revenue of all movies in the dataset is: ${average_revenue:.2f}")
else:
    print("The 'Revenue (Millions)' column is not present in the dataset.")

#QUESTION 3: To calculate the average revenue of movies from 2015 to 2017 in a dataset:

if 'Year' in movies_df.columns and 'Revenue (Millions)' in movies_df.columns:
    filtered_movies = movies_df[(movies_df['Year'] >= 2015) & (movies_df['Year'] <= 2017)]

    average_revenue = filtered_movies['Revenue (Millions)'].mean()

    print(f"The average revenue of movies from 2015 to 2017 is: ${average_revenue:.2f}")
else:
    print("The 'Year' or 'Revenue (Millions)' column is not present in the dataset.")

#QUESTION 4: How many movies were released in the year 2016?

if 'Year' in movies_df.columns:
    movies_2016 = movies_df[movies_df['Year'] == 2016]

    num_movies_2016 = len(movies_2016)

    print(f"The number of movies released in the year 2016 is: {num_movies_2016}")
else:
    print("The 'Year' column is not present in the dataset.")

#QUESTION 5: How many movies were directed by Christopher Nolan?

if 'Director' in movies_df.columns:
    nolan_movies = movies_df[movies_df['Director'] == 'Christopher Nolan']

    num_nolan_movies = len(nolan_movies)

    print(f"The number of movies directed by Christopher Nolan is: {num_nolan_movies}")
else:
    print("The 'Director' column is not present in the dataset.")

#QUESTION 6: How many movies in the dataset have a rating of at least 8.0?

if 'Rating' in movies_df.columns:
    high_rated_movies = movies_df[movies_df['Rating'] >= 8.0]

    num_high_rated_movies = len(high_rated_movies)

    print(f"The number of movies with a rating of at least 8.0 is: {num_high_rated_movies}")
else:
    print("The 'Rating' column is not present in the dataset.")

#QUESTION 7: What is the median rating of movies directed by Christopher Nolan?

if 'Director' in movies_df.columns and 'Rating' in movies_df.columns:
    nolan_movies = movies_df[movies_df['Director'] == 'Christopher Nolan']

    median_rating_nolan = nolan_movies['Rating'].median()

    print(f"The median rating of movies directed by Christopher Nolan is: {median_rating_nolan}")
else:
    print("The 'Director' or 'Rating' column is not present in the dataset.")

#QUESTION 8:Find the year with the highest average rating?

if 'Year' in movies_df.columns and 'Rating' in movies_df.columns:
    average_ratings_by_year = movies_df.groupby('Year')['Rating'].mean()

    highest_average_rating_year = average_ratings_by_year.idxmax()
    highest_average_rating = average_ratings_by_year.max()

    print(f"The year with the highest average rating is {highest_average_rating_year} with an average rating of {highest_average_rating:.2f}")
else:
    print("The 'Year' or 'Rating' column is not present in the dataset.")


#QUESTION 9: What is the percentage increase in number of movies made between 2006 and 2016?

if 'Year' in movies_df.columns:
    movies_2006 = movies_df[movies_df['Year'] == 2006]
    movies_2016 = movies_df[movies_df['Year'] == 2016]

    num_movies_2006 = len(movies_2006)
    num_movies_2016 = len(movies_2016)

    percentage_increase = ((num_movies_2016 - num_movies_2006) / num_movies_2006) * 100

    print(f"The percentage increase in the number of movies between 2006 and 2016 is: {percentage_increase:.2f}%")
else:
    print("The 'Year' column is not present in the dataset.")


#QUESTION 10: Find the most common actor in all the movies?

if 'Actors' in movies_df.columns:
    all_actors = movies_df['Actors'].str.split(', ')
    flat_actors_list = [actor for sublist in all_actors.dropna() for actor in sublist]
    actors_series = pd.Series(flat_actors_list)
    most_common_actor = actors_series.mode()[0]

    print(f"The most common actor in all the movies is: {most_common_actor}")
else:
    print("The 'Actors' column is not present in the dataset.")
    
#QUESTION 11: How many unique genres are there in the dataset?

if 'Genre' in movies_df.columns:
    all_genres = movies_df['Genre'].str.split(', ')
    flat_genres_list = [genre for sublist in all_genres.dropna() for genre in sublist]
    num_unique_genres = len(set(flat_genres_list))

    print(f"The number of unique genres in the dataset is: {num_unique_genres}")
else:
    print("The 'Genre' column is not present in the dataset.")

#QUESTION 12: Do a correlation of the numerical features, what insights can you deduce? Mention at least 5 insights. And what advice can you give directors to produce better movies?
#assuming normal distribution with cleaned data

from scipy.stats import pearsonr
import pandas as pd

numerical_features = movies_df.select_dtypes(include=['float64', 'int64'])

if not numerical_features.empty:
    movies_df_cleaned = numerical_features.dropna()

    correlation_matrix, p_values = pd.DataFrame(), pd.DataFrame()

    for col1 in movies_df_cleaned.columns:
        correlation_matrix[col1], p_values[col1] = zip(*[(pearsonr(movies_df_cleaned[col1], movies_df_cleaned[col2])) for col2 in movies_df_cleaned.columns])

    print("Correlation Matrix:")
    print(correlation_matrix)
    print("\nP-values:")
    print(p_values)

    significance_matrix = p_values < 0.05
    print("\nStatistical Significance (p < 0.05):")
    print(significance_matrix)

else:
    print("No numerical features in the dataset.")

#References/Resources used: *Besides course notes;https://github.com/lakshanagv/Complete-guide-to-data-analysis-using-Python---IMDB-movies-data/blob/main/Quick%20guide%20to%20Data%20Analysis%20using%20Pandas.ipynb; https://pandas.pydata.org/docs/index.html;https://www.kaggle.com/code/trentpark/data-analysis-basics-imdb-dataset ;https://medium.com/@nitin.data1997/imdb-movie-analysis-exploring-movie-data-a-python-analysis-ecf29eab9417