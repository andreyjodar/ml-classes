from dataset import data_set

FNAME = 'review-dataset2/student/student-mat.csv'

if __name__ == '__main__':
    data = data_set(FNAME)
    for key, value in data.items():
        print(key)
    fname = FNAME.split('/')
    fname = fname[-1]
    print('fname -->', fname)