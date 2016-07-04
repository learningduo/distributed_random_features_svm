#ifndef SVM_NORMALIZER_H
#define SVM_NORMALIZER_H

#include <armadillo>

using namespace arma;
using namespace std;

vec fro_norm(123);



mat normRange(mat x, int desired_max, int desired_min) {
    int desired_range = desired_max - desired_min;

    mat x_new = x;
    for (int i = 0; i < x.n_cols; i++) {
        double orig_range = x_new.col(i).max() - x_new.col(i).min();
        double m = x_new.col(i).min();
        double range = x_new.col(i).max() - m;
        x_new.col(i) = (x_new.col(i) - m) / range;
        x_new.col(i) = (x_new.col(i) * desired_range) + desired_min;
    }
    return x_new;
}

mat normMinMax(mat x) {
    mat x_new = x;
    for (int i = 0; i < x.n_cols; i++) {
        x_new.col(i) = (x_new.col(i) - x_new.col(i).min())/(x_new.col(i).max() - x_new.col(i).min());
    }
    return x_new;
}

mat normZ(mat x) {
    mat train_mean = mean(x, 0);
    mat train_std = stddev(x, 0);
    mat mean_mat = repmat(train_mean, x.n_rows, 1);

    mat x_new = x;
    for (int i = 0; i < x.n_cols; i++) {
        if (train_std(i) != 0)
            x_new.col(i) -= train_mean(i);
            x_new.col(i) /= train_std(i);
    }
    return x_new;
}

mat normTest(mat x) {
    mat x_new = x;
    for (int i = 0; i < x.n_cols; i++) {
        x_new.col(i) /= fro_norm(i);
    }
    return x_new;
}

mat normMat(mat x) {
    mat x_new = x;
    for (int i = 0; i < x.n_cols; i++) {
        x_new.col(i) /= norm(x_new.col(i), "fro");
        fro_norm(i) = norm(x_new.col(i), "fro");
    }
    return x_new;
}


mat normCol(mat x) {
    mat x_new = x;

    for (int i = 0; i < x.n_cols; i++) {
        if(norm(x_new.col(i), "fro") != 0)
            x_new.col(i) /= norm(x_new.col(i), "fro");
    }
    return x_new;
}



#endif //SVM_NORMALIZER_H
