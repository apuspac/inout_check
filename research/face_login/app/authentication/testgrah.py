from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus #pydotplusに変更

if __name__ == "__main__":

    #irisデータの読み込み
    iris = load_iris()

    #決定木学習
    clf = tree.DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)

    #決定木モデルの書き出し
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  #pydotplusに変更
    graph.write_png("iris.png")
