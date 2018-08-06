Test datasets "u.data" and "ratings.dat" can be correspondingly downloaded from https://grouplens.org/datasets/movielens/100k and https://grouplens.org/datasets/movielens/1m
"u.data" and "ratings.dat" should put in the same directory with the python files.
"main.py" is the instance program that can be run in commend lines.
Other python files are libraries of different algorithms implemented by us to support the instance.
Note: only pandas and numpy are necessary

main.py can be run in such commends(python main.py parameters): python main.py algorithm mood dataset fraction/userId

For each parameter:

 (1)algorithm:  'uu' => user-user CF under KNN
                'ii' => item-item CF under KNN
                'mf' => basic matrix factorization
                'mf+' => matrix facctorization with bias

 (2)mood:       'eva' => to evaluate the algorithm and print the RMSE
                'app' => to apply the recommender system, output top 5 movies for a specific user input

 (3)dataset:    'small' => the 100k dataset u.data
                'big' => the 1M dataset ratings.dat
 
 (4)fraction/userId:
If the mood is 'eva', the fourth parameter is fraction. Fraction can be any float from 0 to 1 as the fraction of training data and test data. For example, "0.1" => 90% of the dataset is training data, 10% is test data. The program will output the RMSE.
If the mood is 'app', the fourth parameter is userId. UserId could be any interger from 1 to the last user. For example, "235" => as a wrapped up recommender syster, the program will print top 5 books to recommend to user 235.

    



