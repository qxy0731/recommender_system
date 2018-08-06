import sys
from evaluation import *
from uucf_iicf import *
from matrix_factorization import *

parameters = sys.argv[1:]

dataset_dic = {"big":("ratings.dat","::"), "small":("u.data","\t")}

# try:
dataset = parameters[2]
data = pd.read_csv(dataset_dic[dataset][0], sep=dataset_dic[dataset][1], encoding="latin-1",header=None,engine='python')
data.columns = ["user_id","item_id","rating","timestamp"]
ratings_matrix = data.pivot(index="user_id", columns="item_id",values="rating")
ratings_matrix = ratings_matrix.fillna(0).astype(np.int64)
print("Running.......")
# user-user
if parameters[0] == 'uu':
    if parameters[1] == 'eva':
        fraction = float(parameters[3])
        test_data, ratings_matrix= devide_train_test(ratings_matrix,fraction)
        estimated_data = wrap_user(test_data,ratings_matrix,50)
        print("user-user RMSE: {}\n".format(evaluate_rmse(test_data=test_data, estimated_data=estimated_data)))
    elif parameters[1] == 'app':
        userId = int(parameters[3])
        assert(userId >= 1 and userId < ratings_matrix.shape[0])
        print("Top five movies for {} are: ".format(userId),end='')
        print(userbestfive(userId,ratings_matrix,50))
# item-item
elif parameters[0] == 'ii':
    if parameters[1] == 'eva':
        fraction = float(parameters[3])
        test_data, ratings_matrix= devide_train_test(ratings_matrix,fraction)
        estimated_data = wrap_item(test_data,ratings_matrix,50)
        print("item-item RMSE: {}\n".format(evaluate_rmse(test_data=test_data, estimated_data=estimated_data)))
    elif parameters[1] == 'app':
        userId = int(parameters[3])
        assert(userId >= 1 and userId < ratings_matrix.shape[0])
        print("Top five movies for {} are: ".format(userId),end='')
        print(itembestfive(userId,ratings_matrix,50))
# basic mf
elif parameters[0] == 'mf':
    if parameters[1] == 'eva':
        fraction = float(parameters[3])
        test_data, ratings_matrix= devide_train_test(ratings_matrix,fraction)
        estimated_data = estimate_MF(test_data,ratings_matrix)
        print("Basic MF RMSE: {}\n".format(evaluate_rmse(test_data=test_data, estimated_data=estimated_data)))
    elif parameters[1] == 'app':
        userId = int(parameters[3])
        assert(userId >= 1 and userId < ratings_matrix.shape[0])
        print("Top five movies for {} are: ".format(userId),end='')
        print(recommandation_by_MF(ratings_matrix,userId))
# mf with bias
elif parameters[0] == 'mf+':
    if parameters[1] == 'eva':
        fraction = float(parameters[3])
        test_data, ratings_matrix= devide_train_test(ratings_matrix,fraction)
        estimated_data = estimate_MF_bias(test_data,ratings_matrix)
        print("Basic MF RMSE: {}\n".format(evaluate_rmse(test_data=test_data, estimated_data=estimated_data)))
    elif parameters[1] == 'app':
        userId = int(parameters[3])
        assert(userId >= 1 and userId < ratings_matrix.shape[0])
        print("Top five movies for {} are: ".format(userId),end='')
        print(recommandation_by_MF_bias(ratings_matrix,userId))
# except:
#     print("wrong parameters")
