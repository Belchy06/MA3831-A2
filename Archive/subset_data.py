import pandas as pd


def get_subset_based_on_title(dataset):
    # resource has more than 1 associated course
    subset1 = dataset.groupby(["COMBINED_TITLE"]).count().sort_values(["COURSENAME"], ascending=False).reset_index()[
        ['COMBINED_TITLE', 'COURSENAME']]
    ss1 = list(set(subset1[subset1['COURSENAME'] >= 2]['COMBINED_TITLE']))
    subset_based_on_title = dataset[dataset['COMBINED_TITLE'].isin(ss1)]
    return subset_based_on_title


def get_subset_based_on_course(dataset):
    # Course has more than 4 associated resources
    subset2 = dataset.groupby(["COURSENAME"]).count().sort_values(["COMBINED_TITLE"], ascending=False).reset_index()[
        ['COURSENAME', 'COMBINED_TITLE']]
    ss2 = list(set(subset2[subset2['COMBINED_TITLE'] >= 5]['COURSENAME']))
    subset_based_on_course = dataset[dataset['COURSENAME'].isin(ss2)]
    return subset_based_on_course


if __name__ == '__main__':
    # merged_data = pd.read_csv('merged_dat.csv')
    #
    # x = merged_data.sort_values(by='COURSENAME').copy()
    # x.drop_duplicates(keep='first', inplace=True)
    # x.to_csv("final.csv", index=False, encoding='utf-8-sig')

    data = pd.read_csv('final.csv')
    print('Original: {}'.format(len(data)))
    title_sub = get_subset_based_on_title(data.copy())
    # title_sub.to_csv("title_subset.csv", index=False, encoding='utf-8-sig')
    print('Title subset: {}'.format(len(title_sub)))
    course_sub = get_subset_based_on_course(data.copy())
    # course_sub.to_csv("course_subset.csv", index=False, encoding='utf-8-sig')
    print('Course subset: {}'.format(len(course_sub)))
