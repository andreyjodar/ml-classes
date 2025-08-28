def cross_validation(clfs, xtr, ytr):
    intervalo = len(ytr) / 5
    result = []