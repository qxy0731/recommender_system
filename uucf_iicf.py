import pandas as pd
import numpy as np

def userBaseRating(id,item,matrix,k):
    need_compare = matrix.loc[matrix.index==id]
    baseline = np.array(need_compare).astype(np.float64)
    if np.std(baseline) == 0.0:
        baseline[0][0] = baseline[0][0]+0.001
    result_list = []
    user_list = []

    item_index = item - 1
    for i in range(0, len(matrix)):
        each_line = np.array([matrix.iloc[i]]).astype(np.float64)
        if np.std(each_line) == 0.0:
            each_line[0][0] = each_line[0][0]+0.001
        if matrix.iloc[i].tolist()[item_index]!=0:
            result_list.append([matrix.index[i],np.corrcoef(each_line,baseline)[0,1],matrix.iloc[i].tolist()[item_index]])

    result_list = sorted(result_list, key=lambda x : x[1])
    result_list.reverse()
    result_list = result_list[1:]

    if len(result_list)<=k:
        for i in range(0,len(result_list)):
            user_list.append(result_list[i])
    else:
        for i in range(0,k):
            user_list.append(result_list[i])

    curr_sum = 0
    curr_count = 0

    for i in range(0,len(user_list)):
        curr_sum += user_list[i][1]*user_list[i][2]
        curr_count += user_list[i][1]
    if curr_count==0:
        return 0
    else:
        final_result = curr_sum/curr_count
        return final_result


def itemBaseRating(id,item,matrix,k):
    matrix = matrix.T
    need_compare = matrix.loc[matrix.index==item]
    baseline = np.array(need_compare).astype(np.float64)
    if np.std(baseline) == 0.0:
        baseline[0][0] = baseline[0][0]+0.001
    result_list = []
    m_result = []
    item_list = []
    id_index = id - 1

    for i in range(0, len(matrix)):
        each_line = np.array([matrix.iloc[i]]).astype(np.float64)
        if np.std(each_line) == 0.0:
            each_line[0][0] = each_line[0][0]+0.001
        if matrix.iloc[i].tolist()[id_index]!=0:
            result_list.append([matrix.index[i],np.corrcoef(each_line,baseline)[0,1],matrix.iloc[i].tolist()[id_index]])

    result_list = sorted(result_list, key=lambda x : x[1])
    result_list.reverse()
    result_list = result_list[1:]

    if len(result_list)<=k:
        for i in range(0,len(result_list)):
            item_list.append(result_list[i])
    else:
        for i in range(0,k):
            item_list.append(result_list[i])

    curr_sum = 0
    curr_count = 0

    for i in range(0,len(item_list)):
        curr_sum += item_list[i][1]*item_list[i][2]
        curr_count += item_list[i][1]

    if curr_count==0:
        return 0
    else:
        final_result = curr_sum/curr_count
    return final_result

def userbestfive(id,matrix,k):
    result = []
    baseline = matrix.iloc[matrix.index==id].values.tolist()[0]
    for i in range(0,len(baseline)):
        if baseline[i]!=0:
            rating = userBaseRating(id,matrix.columns[i],matrix,k)
            result.append([rating,matrix.columns[i]])
    result = sorted(result, key=lambda x : x[0])
    result.reverse()
    final_result = []
    if len(result)<=5:
        for i in range(0,len(result)):
            final_result.append(result[i][1])
    else:
        for i in range(0,5):
            final_result.append(result[i][1])
    return final_result


def itembestfive(id,matrix,k):
    result = []
    baseline = matrix.iloc[matrix.index==id].values.tolist()[0]
    for i in range(0,len(baseline)):
        if baseline[i]!=0:
            rating = itemBaseRating(id,matrix.columns[i],matrix,k)
            result.append([rating,matrix.columns[i]])
    result = sorted(result, key=lambda x : x[0])
    result.reverse()
    final_result = []
    if len(result)<=5:
        for i in range(0,len(result)):
            final_result.append(result[i][1])
    else:
        for i in range(0,5):
            final_result.append(result[i][1])
    return final_result


def wrap_user(test_data,matrix,k=25):
    final_result = {}
    count = 0
    print("-----------nil={}----user_start-----------------".format(k))

    for each in test_data:
        print("{}/{}".format(count,len(test_data)))
        final_result[each] = userBaseRating(each[0],each[1],matrix,k)
        count += 1
    print("-----------nil={}----user_done-----------------".format(k))
    return final_result


def wrap_item(test_data,matrix,k=25):
    final_result = {}
    count = 0
    print("-----------nil={}----item_start-----------------".format(k))
    for each in test_data:
        print("{}/{}".format(count,len(test_data)))
        final_result[each] = itemBaseRating(each[0],each[1],matrix,k)
        count += 1
    print("-----------nil={}----item_done-----------------".format(k))
    return final_result
